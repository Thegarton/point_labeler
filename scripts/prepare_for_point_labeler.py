#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from point_labeler_bridge import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    IDENTITY_KITTI_POSE,
    classes_from_metadata,
    fmt_float,
    load_csv_frame,
    load_ins_pose_index,
    load_metadata,
    load_semantic_mask,
    read_semantic_classes,
    write_identity_calib,
    write_labels_xml,
)


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    litept_output_dir = Path(args.litept_output_dir).expanduser().resolve() if args.litept_output_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    classes_yaml = Path(args.classes_yaml).expanduser().resolve() if args.classes_yaml else None
    ins_path = resolve_ins_path(csv_dir, args.ins_path)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {csv_dir}")

    pose_index = load_ins_pose_index(ins_path) if ins_path is not None else None

    velodyne_dir = out_dir / "velodyne"
    labels_dir = out_dir / "labels"
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    write_identity_calib(out_dir / "calib.txt")
    class_definitions, class_source = resolve_class_definitions(
        litept_output_dir=litept_output_dir,
        classes_yaml=classes_yaml,
        csv_files=csv_files,
    )
    write_labels_xml(out_dir / "labels.xml", class_definitions)

    manifest: dict = {
        "version": 1,
        "height": args.height,
        "width": args.width,
        "class_source": class_source,
        "classes": [{"name": name, "id": label_id} for name, label_id in class_definitions],
        "ins_path": str(ins_path) if ins_path is not None else None,
        "frames": [],
    }
    poses: list[list[float]] = []

    for csv_path in csv_files:
        frame_id = csv_path.stem
        frame = load_csv_frame(csv_path, frame_id=frame_id, height=args.height, width=args.width)

        metadata_path = litept_output_dir / frame_id / "metadata.json" if litept_output_dir is not None else None
        metadata = load_metadata(metadata_path) if metadata_path is not None else {}
        ego_pose = metadata.get("ego_pose")
        if ego_pose is None and pose_index is not None:
            if frame.timestamp_us is None:
                raise ValueError(f"Cannot match INS pose for {frame_id}: frame timestamp_us is missing")
            ego_pose = pose_index.nearest(frame.timestamp_us).to_jsonable()

        mask_path = litept_output_dir / frame_id / "semantic_mask.npy" if litept_output_dir is not None else None
        if mask_path is not None and mask_path.exists():
            mask = load_semantic_mask(mask_path, height=args.height, width=args.width)
        else:
            mask = np.zeros((args.height, args.width), dtype=np.uint16)

        frame.points_flat.astype(np.float32, copy=False).tofile(velodyne_dir / f"{frame_id}.bin")
        mask.reshape(-1).astype(np.uint32, copy=False).tofile(labels_dir / f"{frame_id}.label")

        kitti_pose = ego_pose.get("kitti_pose") if ego_pose else IDENTITY_KITTI_POSE
        if len(kitti_pose) != 12:
            raise ValueError(f"ego_pose.kitti_pose for {frame_id} must have 12 values, got {len(kitti_pose)}")
        poses.append([float(value) for value in kitti_pose])

        manifest["frames"].append(
            {
                "frame_id": frame_id,
                "source_csv": str(csv_path),
                "source_metadata": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
                "source_semantic_mask": str(mask_path) if mask_path is not None and mask_path.exists() else None,
                "velodyne": str(velodyne_dir / f"{frame_id}.bin"),
                "label": str(labels_dir / f"{frame_id}.label"),
                "timestamp_us": frame.timestamp_us,
                "timestamp_s": frame.timestamp_s,
                "timestamp_u": frame.timestamp_u,
                "point_count": int(frame.points_flat.shape[0]),
                "shape": [args.height, args.width],
                "frame_meta": frame.meta,
                "ego_pose": ego_pose,
                "source_metadata_payload": metadata,
            }
        )

    (out_dir / "poses.txt").write_text(
        "".join(" ".join(fmt_float(value) for value in pose) + "\n" for pose in poses),
        encoding="utf-8",
    )
    (out_dir / "bridge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_settings_hint(out_dir)

    print(json.dumps({"frames": len(csv_files), "out_dir": str(out_dir), "ins_path": str(ins_path) if ins_path else None}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV/LitePT semantic masks to a point_labeler KITTI-like dataset.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--litept-output-dir", default=None)
    parser.add_argument("--classes-yaml", default=None, help="Fallback class config. If omitted, class_names are read from LitePT metadata.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ins-path", default=None)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    return parser.parse_args()


def resolve_ins_path(csv_dir: Path, ins_path: str | None) -> Path | None:
    if ins_path:
        return resolve_ins_file(Path(ins_path).expanduser().resolve())
    candidate = csv_dir / "ins"
    return resolve_ins_file(candidate) if candidate.exists() else None


def resolve_ins_file(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ""}])
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise FileNotFoundError(f"No INS file found in {path}")
        raise ValueError(f"Multiple INS files found in {path}; pass --ins-path explicitly")
    raise FileNotFoundError(f"INS path does not exist: {path}")


def resolve_class_definitions(
    *,
    litept_output_dir: Path | None,
    classes_yaml: Path | None,
    csv_files: list[Path],
) -> tuple[list[tuple[str, int]], str | None]:
    if litept_output_dir is not None:
        for csv_path in csv_files:
            metadata_path = litept_output_dir / csv_path.stem / "metadata.json"
            metadata = load_metadata(metadata_path)
            classes = classes_from_metadata(metadata)
            if classes:
                return classes, str(metadata_path)

    if classes_yaml is not None:
        return read_semantic_classes(classes_yaml), str(classes_yaml)

    raise ValueError("Could not infer classes from LitePT metadata. Pass --classes-yaml as a fallback.")


def write_settings_hint(out_dir: Path) -> None:
    labels_xml = out_dir / "labels.xml"
    text = "\n".join(
        [
            f"labels file: {labels_xml}",
            "point size: 4",
            "render points as spheres: true",
            "tile size: 100.0",
            "max scans: 500",
            "min range: 0.0",
            "max range: 200.0",
            "",
        ]
    )
    (out_dir / "settings.cfg.example").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

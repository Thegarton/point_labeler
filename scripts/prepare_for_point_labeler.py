#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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
    parse_image_size,
    read_kitti_pose_values,
    read_semantic_classes,
    write_identity_calib,
    write_labels_xml,
)
from precompute_point_rgb import precompute_point_rgb

DEFAULT_TILE_SIZE = 100.0
DEFAULT_MAX_RANGE = 200.0
TILE_MARGIN = 20.0
RANGE_MARGIN = 10.0
IGNORE_ID = 255

DRIVING_V1_MERGE_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("TRUCK_BUS", ("Truck", "Bus", "Other Vehicle")),
    ("Cyclist", ("Bicyclist", "Bicycle")),
    ("motorcycle", ("Motorcyclist", "Motorcycle")),
    ("ground", ("Walkable", "Sidewalk", "Lane Marker", "Road", "Curb")),
    ("Thin vertical bar", ("Pole", "Tree Trunk")),
]

DRIVING_V1_TARGET_ORDER = [
    "Car",
    "TRUCK_BUS",
    "Cyclist",
    "motorcycle",
    "Pedestrian",
    "Sign",
    "Traffic Light",
    "Thin vertical bar",
    "Construction Cone",
    "Building",
    "Vegetation",
    "ground",
    "Other Ground",
]

DRIVING_V1_REQUIRED_SOURCE_NAMES = {
    source_name.casefold() for _, source_names in DRIVING_V1_MERGE_GROUPS for source_name in source_names
}


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir).expanduser().resolve()
    litept_output_dir = Path(args.litept_output_dir).expanduser().resolve() if args.litept_output_dir else None
    out_dir = Path(args.out_dir).expanduser().resolve()
    classes_yaml = Path(args.classes_yaml).expanduser().resolve() if args.classes_yaml else None
    pose_calib_file = Path(args.pose_calib_file).expanduser().resolve() if args.pose_calib_file else None
    rgb_calib_file = Path(args.rgb_calib_file or args.calib_file).expanduser().resolve() if (args.rgb_calib_file or args.calib_file) else None
    ins_path = resolve_ins_path(csv_dir, args.ins_path)

    csv_files = sorted(csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {csv_dir}")

    pose_index = load_ins_pose_index(ins_path) if ins_path is not None else None

    velodyne_dir = out_dir / "velodyne"
    labels_dir = out_dir / "labels"
    image_dir = out_dir / "image_2"
    velodyne_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    write_pose_calibration(out_dir / "calib.txt", calib_file=pose_calib_file)
    copied_rgb_calib = copy_rgb_calibration(out_dir=out_dir, calib_file=rgb_calib_file, calibration_type=args.type)
    class_definitions, class_source = resolve_class_definitions(
        litept_output_dir=litept_output_dir,
        classes_yaml=classes_yaml,
        csv_files=csv_files,
    )
    class_merge = build_class_merge(class_definitions, preset=args.class_merge_preset)
    label_remap = class_merge["source_id_to_target_id"]
    class_definitions = class_merge["classes"]
    labels_xml = out_dir / "labels.xml"
    if args.overwrite_labels_xml or not labels_xml.exists():
        write_labels_xml(labels_xml, class_definitions)

    manifest: dict = {
        "version": 1,
        "height": args.height,
        "width": args.width,
        "pose_mode": args.pose_mode,
        "visualization_axis_mode": args.visualization_axis_mode,
        "class_source": class_source,
        "class_merge": class_merge["metadata"],
        "source_classes": class_merge["source_classes"],
        "classes": [{"name": name, "id": label_id} for name, label_id in class_definitions],
        "ins_path": str(ins_path) if ins_path is not None else None,
        "frames": [],
    }
    max_abs_xy = 0.0
    max_range = 0.0

    for csv_path in csv_files:
        frame_id = csv_path.stem
        frame = load_csv_frame(csv_path, frame_id=frame_id, height=args.height, width=args.width)
        frame_max_abs_xy, frame_max_range = frame_extent(frame.points_flat)
        max_abs_xy = max(max_abs_xy, frame_max_abs_xy)
        max_range = max(max_range, frame_max_range)

        metadata_path = litept_output_dir / frame_id / "metadata.json" if litept_output_dir is not None else None
        metadata = load_metadata(metadata_path) if metadata_path is not None else {}
        pose_path = litept_output_dir / frame_id / "pose.txt" if litept_output_dir is not None else None
        ego_pose = metadata.get("ego_pose")
        if ego_pose is None and pose_path is not None and pose_path.exists():
            ego_pose = ego_pose_from_kitti_pose_file(pose_path, timestamp_us=frame.timestamp_us)
        if ego_pose is None and pose_index is not None:
            if frame.timestamp_us is None:
                raise ValueError(f"Cannot match INS pose for {frame_id}: frame timestamp_us is missing")
            ego_pose = pose_index.nearest(frame.timestamp_us).to_jsonable()

        mask_path = litept_output_dir / frame_id / "semantic_mask.npy" if litept_output_dir is not None else None
        if mask_path is not None and mask_path.exists():
            mask = load_semantic_mask(mask_path, height=args.height, width=args.width)
            mask = remap_semantic_mask(mask, label_remap)
        else:
            mask = np.zeros((args.height, args.width), dtype=np.uint16)

        frame.points_flat.astype(np.float32, copy=False).tofile(velodyne_dir / f"{frame_id}.bin")
        mask.reshape(-1).astype(np.uint32, copy=False).tofile(labels_dir / f"{frame_id}.label")
        image_path = copy_frame_image(csv_dir=csv_dir, image_dir=image_dir, frame_id=frame_id)

        if ego_pose is not None and len(ego_pose.get("kitti_pose", [])) != 12:
            raise ValueError(f"ego_pose.kitti_pose for {frame_id} must have 12 values, got {len(ego_pose.get('kitti_pose', []))}")

        manifest["frames"].append(
            {
                "frame_id": frame_id,
                "source_csv": str(csv_path),
                "source_metadata": str(metadata_path) if metadata_path is not None and metadata_path.exists() else None,
                "source_semantic_mask": str(mask_path) if mask_path is not None and mask_path.exists() else None,
                "source_pose": str(pose_path) if pose_path is not None and pose_path.exists() else None,
                "velodyne": str(velodyne_dir / f"{frame_id}.bin"),
                "label": str(labels_dir / f"{frame_id}.label"),
                "image": str(image_path) if image_path is not None else None,
                "timestamp_us": frame.timestamp_us,
                "timestamp_s": frame.timestamp_s,
                "timestamp_u": frame.timestamp_u,
                "point_count": int(frame.points_flat.shape[0]),
                "shape": [args.height, args.width],
                "frame_meta": frame.meta,
                "ego_pose": ego_pose,
                "visualization_pose": None,
                "source_metadata_payload": metadata,
            }
        )

    poses = visualization_poses_for(args.pose_mode, manifest["frames"], args.visualization_axis_mode)
    for frame_manifest, visualization_pose in zip(manifest["frames"], poses):
        frame_manifest["visualization_pose"] = visualization_pose

    (out_dir / "poses.txt").write_text(
        "".join(" ".join(fmt_float(value) for value in pose) + "\n" for pose in poses),
        encoding="utf-8",
    )
    (out_dir / "bridge_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    settings_text = settings_content(out_dir, max_abs_xy=max_abs_xy, max_range=max_range)
    (out_dir / "settings.cfg").write_text(settings_text, encoding="utf-8")
    (out_dir / "settings.cfg.example").write_text(settings_text, encoding="utf-8")

    rgb_summary = None
    if args.precompute_rgb:
        rgb_summary = precompute_point_rgb(
            dataset_dir=out_dir,
            camera_id=args.camera_id,
            overwrite=args.overwrite_rgb,
            calibration_type=args.type,
            calib_file=copied_rgb_calib,
            calibration_image_size=parse_image_size(args.rgb_calibration_image_size),
        )

    print(
        json.dumps(
            {
                "frames": len(csv_files),
                "out_dir": str(out_dir),
                "ins_path": str(ins_path) if ins_path else None,
                "rgb_manifest": str(out_dir / "point_rgb" / "point_rgb_manifest.json") if rgb_summary is not None else None,
            },
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CSV/LitePT semantic masks to a point_labeler KITTI-like dataset.")
    parser.add_argument("--csv-dir", required=True)
    parser.add_argument("--litept-output-dir", default=None)
    parser.add_argument("--classes-yaml", default=None, help="Fallback class config. If omitted, class_names are read from LitePT metadata.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ins-path", default=None)
    parser.add_argument("--pose-calib-file", default=None, help="Optional calib.txt for labeler poses. Defaults to identity Tr.")
    parser.add_argument("--calib-file", default=None, help="Deprecated alias for --rgb-calib-file.")
    parser.add_argument("--rgb-calib-file", default=None, help="KITTI-style camera calibration used only for RGB projection.")
    parser.add_argument(
        "--rgb-calibration-image-size",
        default=None,
        help="WIDTHxHEIGHT image size for --rgb-calib-file projection matrices, e.g. 3840x2160.",
    )
    parser.add_argument("--type", choices=["koide", "factory"], default=None, help="Calibration type saved as rgb_calib_<type>.txt.")
    parser.add_argument("--precompute-rgb", action="store_true", help="Precompute point_rgb/*.rgb from image_2 and RGB calibration.")
    parser.add_argument("--camera-id", default="P2", help="KITTI camera projection matrix used for RGB precompute.")
    parser.add_argument("--overwrite-rgb", action="store_true", help="Overwrite existing point_rgb/*.rgb files.")
    parser.add_argument("--overwrite-labels-xml", action="store_true")
    parser.add_argument(
        "--class-merge-preset",
        choices=["auto", "none", "driving_v1"],
        default="auto",
        help="Merge source semantic ids before writing point_labeler labels. auto applies driving_v1 to the full driving taxonomy.",
    )
    parser.add_argument("--pose-mode", choices=["relative_ego", "local_identity"], default="relative_ego")
    parser.add_argument(
        "--visualization-axis-mode",
        choices=["ego_y_forward", "kitti_x_forward"],
        default="ego_y_forward",
        help="Axis convention for poses.txt only. ego_y_forward maps ego +Y motion to KITTI/Velodyne +X forward.",
    )
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


def write_pose_calibration(out_path: Path, *, calib_file: Path | None) -> None:
    if calib_file is None:
        write_identity_calib(out_path)
        return
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibration file does not exist: {calib_file}")
    shutil.copy2(calib_file, out_path)


def copy_rgb_calibration(*, out_dir: Path, calib_file: Path | None, calibration_type: str | None) -> Path | None:
    if calib_file is None:
        return None
    if not calib_file.exists():
        raise FileNotFoundError(f"RGB calibration file does not exist: {calib_file}")
    filename = f"rgb_calib_{calibration_type}.txt" if calibration_type else "rgb_calib.txt"
    out_path = out_dir / filename
    shutil.copy2(calib_file, out_path)
    return out_path


def copy_frame_image(*, csv_dir: Path, image_dir: Path, frame_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = csv_dir / f"{frame_id}{suffix}"
        if candidate.exists():
            image_dir.mkdir(parents=True, exist_ok=True)
            out = image_dir / f"{frame_id}{suffix}"
            shutil.copy2(candidate, out)
            return out
    return None


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


def build_class_merge(classes: list[tuple[str, int]], *, preset: str) -> dict:
    source_classes = [{"name": name, "id": label_id} for name, label_id in classes]
    if preset == "none" or (preset == "auto" and not should_apply_driving_v1_merge(classes)):
        return {
            "classes": classes,
            "source_id_to_target_id": {label_id: label_id for _, label_id in classes},
            "source_classes": source_classes,
            "metadata": {
                "preset": preset,
                "applied": False,
                "groups": [],
                "source_id_to_target_id": {label_id: label_id for _, label_id in classes},
            },
        }
    if preset not in {"auto", "driving_v1"}:
        raise ValueError(f"Unsupported class merge preset: {preset}")
    return build_driving_v1_class_merge(classes, requested_preset=preset)


def should_apply_driving_v1_merge(classes: list[tuple[str, int]]) -> bool:
    names = {name.casefold() for name, _ in classes}
    return DRIVING_V1_REQUIRED_SOURCE_NAMES.issubset(names)


def build_driving_v1_class_merge(classes: list[tuple[str, int]], *, requested_preset: str) -> dict:
    source_to_target_name = {
        source_name.casefold(): target_name
        for target_name, source_names in DRIVING_V1_MERGE_GROUPS
        for source_name in source_names
    }
    target_names_in_source_order: list[str] = []
    source_target_names: dict[int, str] = {}
    for source_name, source_id in classes:
        if source_id == IGNORE_ID or source_name.casefold() == "ignore":
            continue
        target_name = source_to_target_name.get(source_name.casefold(), source_name)
        source_target_names[source_id] = target_name
        if target_name not in target_names_in_source_order:
            target_names_in_source_order.append(target_name)

    target_order = [
        name for name in DRIVING_V1_TARGET_ORDER if name in target_names_in_source_order
    ] + [
        name for name in target_names_in_source_order if name not in DRIVING_V1_TARGET_ORDER
    ]
    target_name_to_id: dict[str, int] = {}
    source_id_to_target_id: dict[int, int] = {}
    merged_classes: list[tuple[str, int]] = []

    for target_id, target_name in enumerate(target_order):
        target_name_to_id[target_name] = target_id
        merged_classes.append((target_name, target_id))

    for source_id, target_name in source_target_names.items():
        source_id_to_target_id[source_id] = target_name_to_id[target_name]

    ignore_sources = [(name, label_id) for name, label_id in classes if label_id == IGNORE_ID or name.casefold() == "ignore"]
    if ignore_sources:
        merged_classes.append(("ignore", IGNORE_ID))
        for _, source_id in ignore_sources:
            source_id_to_target_id[source_id] = IGNORE_ID

    groups = []
    for target_name, source_names in DRIVING_V1_MERGE_GROUPS:
        matched = [
            {"name": source_name, "id": source_id}
            for source_name, source_id in classes
            if source_name.casefold() in {name.casefold() for name in source_names}
        ]
        if matched:
            groups.append({"target": target_name, "target_id": target_name_to_id.get(target_name), "sources": matched})

    return {
        "classes": merged_classes,
        "source_id_to_target_id": source_id_to_target_id,
        "source_classes": [{"name": name, "id": label_id} for name, label_id in classes],
        "metadata": {
            "preset": requested_preset,
            "applied": True,
            "groups": groups,
            "source_id_to_target_id": source_id_to_target_id,
        },
    }


def remap_semantic_mask(mask: np.ndarray, source_id_to_target_id: dict[int, int]) -> np.ndarray:
    out = np.asarray(mask, dtype=np.uint16).copy()
    for source_id in np.unique(mask):
        source_int = int(source_id)
        if source_int not in source_id_to_target_id:
            continue
        out[mask == source_id] = np.uint16(source_id_to_target_id[source_int])
    return out


def ego_pose_from_kitti_pose_file(path: Path, *, timestamp_us: int | None) -> dict:
    kitti_pose = read_kitti_pose_values(path)
    matrix = kitti_pose_to_matrix(kitti_pose)
    return {
        "timestamp_us": timestamp_us,
        "source_timestamp_us": timestamp_us,
        "delta_us": 0,
        "translation": matrix[:3, 3].astype(float).tolist(),
        "rotation_matrix": matrix[:3, :3].astype(float).tolist(),
        "kitti_pose": kitti_pose,
        "source": str(path),
    }


def visualization_poses_for(pose_mode: str, frames: list[dict], axis_mode: str) -> list[list[float]]:
    if pose_mode == "local_identity":
        return [list(IDENTITY_KITTI_POSE) for _ in frames]
    if pose_mode == "relative_ego":
        return relative_ego_poses(frames, axis_mode=axis_mode)
    raise ValueError(f"Unsupported pose mode: {pose_mode}")


def relative_ego_poses(frames: list[dict], *, axis_mode: str) -> list[list[float]]:
    valid: list[tuple[int, int, np.ndarray]] = []
    for index, frame in enumerate(frames):
        ego_pose = frame.get("ego_pose")
        if not ego_pose:
            continue
        kitti_pose = ego_pose.get("kitti_pose")
        if not kitti_pose:
            continue
        timestamp = ego_pose.get("timestamp_us", ego_pose.get("source_timestamp_us", frame.get("timestamp_us")))
        if timestamp is None:
            timestamp = index
        valid.append((int(timestamp), index, kitti_pose_to_matrix(kitti_pose)))

    if not valid:
        return [list(IDENTITY_KITTI_POSE) for _ in frames]

    _, _, first_pose = min(valid, key=lambda item: (item[0], item[1]))
    origin = np.linalg.inv(first_pose)
    matrices_by_index = {index: matrix for _, index, matrix in valid}

    poses: list[list[float]] = []
    for index in range(len(frames)):
        matrix = matrices_by_index.get(index)
        if matrix is None:
            poses.append(list(IDENTITY_KITTI_POSE))
            continue
        poses.append(matrix_to_kitti_pose(convert_visualization_axes(origin @ matrix, axis_mode)))
    return poses


def convert_visualization_axes(matrix: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode == "kitti_x_forward":
        return matrix
    if axis_mode == "ego_y_forward":
        basis = np.eye(4, dtype=np.float64)
        basis[:3, :3] = np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return basis @ matrix @ np.linalg.inv(basis)
    raise ValueError(f"Unsupported visualization axis mode: {axis_mode}")


def kitti_pose_to_matrix(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (12,):
        raise ValueError(f"KITTI pose must have 12 values, got {arr.shape}")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :4] = arr.reshape(3, 4)
    return matrix


def matrix_to_kitti_pose(matrix: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(matrix, dtype=np.float64)[:3, :4].reshape(-1)]


def frame_extent(points_flat: np.ndarray) -> tuple[float, float]:
    xyz = np.asarray(points_flat[:, :3], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    if not np.any(finite):
        return 0.0, 0.0
    xyz = xyz[finite]
    max_abs_xy = float(np.max(np.abs(xyz[:, :2])))
    max_range = float(np.max(np.linalg.norm(xyz, axis=1)))
    return max_abs_xy, max_range


def settings_content(out_dir: Path, *, max_abs_xy: float, max_range: float) -> str:
    labels_xml = out_dir / "labels.xml"
    tile_size = max(DEFAULT_TILE_SIZE, 2.0 * max_abs_xy + TILE_MARGIN)
    max_range_value = max(DEFAULT_MAX_RANGE, max_range + RANGE_MARGIN)
    text = "\n".join(
        [
            f"labels file: {labels_xml}",
            "point size: 4",
            "render points as spheres: true",
            "shade point spheres: true",
            "show intensity: false",
            "allow velodyne only: false",
            "point cloud source: velodyne",
            "csv image width: 1920",
            "csv image height: 1536",
            f"tile size: {tile_size:.3f}",
            "max scans: 500",
            "min range: 0.0",
            f"max range: {max_range_value:.3f}",
            "",
        ]
    )
    return text


if __name__ == "__main__":
    main()

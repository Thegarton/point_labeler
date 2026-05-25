#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from point_labeler_bridge import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    labels_to_mask,
    load_label_file,
    write_kitti_pose_values,
)


def main() -> None:
    args = parse_args()
    labeler_dir = Path(args.labeler_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = labeler_dir / "bridge_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing bridge manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    height = int(args.height or manifest.get("height") or DEFAULT_HEIGHT)
    width = int(args.width or manifest.get("width") or DEFAULT_WIDTH)

    exported = 0
    for frame in manifest.get("frames", []):
        frame_id = str(frame["frame_id"])
        label_path = labeler_dir / "labels" / f"{frame_id}.label"
        labels = load_label_file(label_path, height=height, width=width)
        mask = labels_to_mask(labels, height=height, width=width)

        frame_out = out_dir / frame_id
        frame_out.mkdir(parents=True, exist_ok=True)
        mask_path = frame_out / "semantic_mask.npy"
        np.save(mask_path, mask.astype(np.uint16, copy=False))

        metadata = {
            "frame_id": frame_id,
            "semantic_mask": str(mask_path),
            "confidence_mask": None,
            "pose": None,
            "pseudo_label_version": args.pseudo_label_version,
            "provenance": args.provenance,
        }

        ego_pose = frame.get("ego_pose")
        if ego_pose is not None:
            metadata["ego_pose"] = ego_pose
            pose_path = frame_out / "pose.txt"
            write_kitti_pose_values(pose_path, ego_pose["kitti_pose"])
            metadata["pose"] = str(pose_path)

        (frame_out / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        exported += 1

    print(json.dumps({"exported_frames": exported, "out_dir": str(out_dir)}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export point_labeler .label files back to LitePT semantic masks.")
    parser.add_argument("--labeler-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--pseudo-label-version", default="manual_point_labeler_v0")
    parser.add_argument("--provenance", default="human_corrected")
    return parser.parse_args()


if __name__ == "__main__":
    main()

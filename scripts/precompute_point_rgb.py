#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from point_labeler_bridge import (
    project_rgb_to_points,
    read_image_rgb,
    read_kitti_calibration,
    read_velodyne_bin,
    select_camera_projection,
    select_lidar_to_camera,
    write_point_rgb,
)


def main() -> None:
    args = parse_args()
    summary = precompute_point_rgb(
        dataset_dir=Path(args.dataset_dir).expanduser().resolve(),
        camera_id=args.camera_id,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def precompute_point_rgb(*, dataset_dir: Path, camera_id: str = "P2", overwrite: bool = False) -> dict:
    calib_path = dataset_dir / "calib.txt"
    velodyne_dir = dataset_dir / "velodyne"
    image_dir = dataset_dir / "image_2"
    rgb_dir = dataset_dir / "point_rgb"

    calibration = read_kitti_calibration(calib_path)
    projection = select_camera_projection(calibration, camera_id)
    tr = select_lidar_to_camera(calibration)

    frames = []
    for points_path in sorted(velodyne_dir.glob("*.bin")):
        frame_id = points_path.stem
        rgb_path = rgb_dir / f"{frame_id}.rgb"
        image_path = find_frame_image(image_dir, frame_id)
        if image_path is None:
            frames.append({"frame_id": frame_id, "status": "missing_image", "rgb": None})
            continue
        if rgb_path.exists() and not overwrite:
            frames.append({"frame_id": frame_id, "status": "exists", "rgb": str(rgb_path)})
            continue

        points = read_velodyne_bin(points_path)
        image = read_image_rgb(image_path)
        rgb, projection_summary = project_rgb_to_points(points, image, projection, tr)
        write_point_rgb(rgb_path, rgb)
        frames.append(
            {
                "frame_id": frame_id,
                "status": "created",
                "points": projection_summary["points"],
                "projected": projection_summary["projected"],
                "behind_camera": projection_summary["behind_camera"],
                "out_of_image": projection_summary["out_of_image"],
                "image": str(image_path),
                "rgb": str(rgb_path),
            }
        )

    summary = {"version": 1, "dataset_dir": str(dataset_dir), "camera_id": camera_id, "frames": frames}
    rgb_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "point_rgb_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def find_frame_image(image_dir: Path, frame_id: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        path = image_dir / f"{frame_id}{suffix}"
        if path.exists():
            return path
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute per-point RGB colors from camera images and KITTI calib.txt.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--camera-id", default="P2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

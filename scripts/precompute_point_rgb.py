#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from point_labeler_bridge import (
    parse_image_size,
    project_rgb_to_points,
    read_calibration_image_size,
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
        calibration_type=args.type,
        calib_file=Path(args.calib_file).expanduser().resolve() if args.calib_file else None,
        calibration_image_size=parse_image_size(args.calibration_image_size),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def precompute_point_rgb(
    *,
    dataset_dir: Path,
    camera_id: str = "P2",
    overwrite: bool = False,
    calibration_type: str | None = None,
    calib_file: Path | None = None,
    calibration_image_size: tuple[int, int] | None = None,
) -> dict:
    calib_path = resolve_rgb_calib_path(dataset_dir=dataset_dir, calibration_type=calibration_type, calib_file=calib_file)
    velodyne_dir = dataset_dir / "velodyne"
    image_dir = dataset_dir / "image_2"
    rgb_dir = dataset_dir / "point_rgb"

    calibration = read_kitti_calibration(calib_path)
    projection = select_camera_projection(calibration, camera_id)
    tr = select_lidar_to_camera(calibration)
    calibration_image_size = calibration_image_size or read_calibration_image_size(calib_path, camera_id=camera_id)

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
        rgb, projection_summary = project_rgb_to_points(
            points,
            image,
            projection,
            tr,
            calibration_image_size=calibration_image_size,
        )
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

    summary = {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "camera_id": camera_id,
        "calibration_type": calibration_type,
        "calib_file": str(calib_path),
        "calibration_image_size": list(calibration_image_size) if calibration_image_size is not None else None,
        "frames": frames,
    }
    rgb_dir.mkdir(parents=True, exist_ok=True)
    (rgb_dir / "point_rgb_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def resolve_rgb_calib_path(*, dataset_dir: Path, calibration_type: str | None, calib_file: Path | None) -> Path:
    if calib_file is not None:
        if not calib_file.exists():
            raise FileNotFoundError(f"RGB calibration file does not exist: {calib_file}")
        return calib_file

    candidates: list[Path] = []
    if calibration_type:
        candidates.extend(
            [
                dataset_dir / f"rgb_calib_{calibration_type}.txt",
                dataset_dir / f"{calibration_type}_calib.txt",
            ]
        )
    candidates.extend([dataset_dir / "rgb_calib.txt", dataset_dir / "calib.txt"])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No RGB calibration file found. Pass --calib-file, or place rgb_calib.txt/rgb_calib_<type>.txt "
        "inside the dataset directory."
    )


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
    parser.add_argument("--type", choices=["koide", "factory"], default=None, help="Calibration type used to resolve rgb_calib_<type>.txt.")
    parser.add_argument("--calib-file", default=None, help="Explicit KITTI-style RGB calibration file.")
    parser.add_argument(
        "--calibration-image-size",
        default=None,
        help="WIDTHxHEIGHT image size for the camera projection matrix, e.g. 3840x2160. "
        "If omitted, image_size/S_XX is read from the calibration file when present.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"


def run_script(script: str, *args: str) -> None:
    env = {**os.environ, "PYTHONPATH": str(SCRIPTS)}
    subprocess.run([sys.executable, str(SCRIPTS / script), *args], check=True, env=env, capture_output=True, text=True)


def write_classes(path: Path) -> None:
    path.write_text(
        "semantic_classes:\n"
        "  background: 0\n"
        "  car: 2\n"
        "  ignore: 255\n",
        encoding="utf-8",
    )


def write_metadata(path: Path, ego_pose: dict | None = None) -> None:
    metadata = {
        "frame_id": path.parent.name,
        "label_space": "litept_waymo_semseg",
        "litept_dataset": "waymo",
        "class_names": ["Car", "Truck", "Road"],
        "num_classes": 3,
        "ignore_index": 255,
    }
    if ego_pose is not None:
        metadata["ego_pose"] = ego_pose
    path.write_text(json.dumps(metadata), encoding="utf-8")


def write_xyz_csv(path: Path, *, height: int, width: int, timestamp_s: int = 10, timestamp_u: int = 260_000) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("x,y,z,intensity,timestamp_s,timestamp_u\n")
        for i in range(height * width):
            f.write(f"{i},{i + 1},{i + 2},0.5,{timestamp_s},{timestamp_u}\n")


def test_roundtrip_preserves_mask_and_existing_ego_pose(tmp_path: Path):
    height, width = 2, 3
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    write_xyz_csv(csv_dir / "frame_000.csv", height=height, width=width)
    (csv_dir / "frame_000.jpg").write_bytes(b"fake-jpeg")
    litept = tmp_path / "litept"
    frame_out = litept / "frame_000"
    frame_out.mkdir(parents=True)
    mask = np.array([[0, 2, 255], [2, 0, 255]], dtype=np.uint16)
    np.save(frame_out / "semantic_mask.npy", mask)
    ego_pose = {
        "timestamp_us": 10_260_000,
        "source_timestamp_us": 10_300_000,
        "delta_us": 40_000,
        "translation": [4.0, 5.0, 6.0],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "kitti_pose": [1.0, 0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 1.0, 6.0],
    }
    write_metadata(frame_out / "metadata.json", ego_pose=ego_pose)

    labeler_dir = tmp_path / "labeler"
    run_script(
        "prepare_for_point_labeler.py",
        "--csv-dir",
        str(csv_dir),
        "--litept-output-dir",
        str(litept),
        "--out-dir",
        str(labeler_dir),
        "--height",
        str(height),
        "--width",
        str(width),
    )

    assert (labeler_dir / "velodyne" / "frame_000.bin").exists()
    assert (labeler_dir / "labels" / "frame_000.label").exists()
    assert (labeler_dir / "image_2" / "frame_000.jpg").read_bytes() == b"fake-jpeg"
    pose_values = [float(x) for x in (labeler_dir / "poses.txt").read_text(encoding="utf-8").strip().split()]
    assert pose_values == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    manifest = json.loads((labeler_dir / "bridge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["pose_mode"] == "relative_ego"
    assert manifest["frames"][0]["ego_pose"] == ego_pose
    assert manifest["frames"][0]["image"] == str(labeler_dir / "image_2" / "frame_000.jpg")
    assert manifest["frames"][0]["visualization_pose"] == pose_values
    settings = (labeler_dir / "settings.cfg").read_text(encoding="utf-8")
    assert f"labels file: {labeler_dir / 'labels.xml'}" in settings
    assert "render points as spheres: true" in settings

    corrected = tmp_path / "corrected"
    run_script(
        "export_from_point_labeler.py",
        "--labeler-dir",
        str(labeler_dir),
        "--out-dir",
        str(corrected),
    )

    np.testing.assert_array_equal(np.load(corrected / "frame_000" / "semantic_mask.npy"), mask)
    metadata = json.loads((corrected / "frame_000" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["ego_pose"] == ego_pose
    assert metadata["class_names"] == ["Car", "Truck", "Road"]
    assert metadata["confidence_mask"] is None
    assert metadata["manual_reviewed"] is True
    assert metadata["pose"] == str(corrected / "frame_000" / "pose.txt")
    assert len((corrected / "frame_000" / "pose.txt").read_text(encoding="utf-8").strip().split()) == 12


def test_prepare_builds_ego_pose_from_ins_when_metadata_is_missing(tmp_path: Path):
    height, width = 2, 2
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    write_xyz_csv(csv_dir / "frame_000.csv", height=height, width=width)
    ins_dir = csv_dir / "ins"
    ins_dir.mkdir()
    (ins_dir / "ins_0.csv").write_text(
        "secs\tnsecs\tattitude_X\tattitude_Y\tattitude_Z\tlatitude\tlongitude\televation\tutmPosition_X\tutmPosition_Y\tutmPosition_Z\n"
        "10\t100000000\t0\t0\t0\t55.0\t37.0\t150.0\t1\t2\t3\n"
        "10\t300000000\t0\t0\t0\t55.0\t37.0\t150.0\t4\t5\t6\n",
        encoding="utf-8",
    )
    litept = tmp_path / "litept"
    frame_out = litept / "frame_000"
    frame_out.mkdir(parents=True)
    write_metadata(frame_out / "metadata.json")

    labeler_dir = tmp_path / "labeler"
    run_script(
        "prepare_for_point_labeler.py",
        "--csv-dir",
        str(csv_dir),
        "--litept-output-dir",
        str(litept),
        "--out-dir",
        str(labeler_dir),
        "--height",
        str(height),
        "--width",
        str(width),
    )

    manifest = json.loads((labeler_dir / "bridge_manifest.json").read_text(encoding="utf-8"))
    ego_pose = manifest["frames"][0]["ego_pose"]
    assert ego_pose["source_timestamp_us"] == 10_300_000
    assert ego_pose["delta_us"] == 40_000
    assert ego_pose["translation"] == [4.0, 5.0, 6.0]


def test_prepare_does_not_overwrite_existing_labels_xml_without_flag(tmp_path: Path):
    height, width = 2, 2
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    write_xyz_csv(csv_dir / "frame_000.csv", height=height, width=width)
    litept = tmp_path / "litept"
    frame_out = litept / "frame_000"
    frame_out.mkdir(parents=True)
    write_metadata(frame_out / "metadata.json")

    labeler_dir = tmp_path / "labeler"
    labeler_dir.mkdir()
    custom_xml = "<config><label><id>7</id><name>custom</name><color>1 2 3</color><root>mine</root></label></config>\n"
    (labeler_dir / "labels.xml").write_text(custom_xml, encoding="utf-8")

    run_script(
        "prepare_for_point_labeler.py",
        "--csv-dir",
        str(csv_dir),
        "--litept-output-dir",
        str(litept),
        "--out-dir",
        str(labeler_dir),
        "--height",
        str(height),
        "--width",
        str(width),
    )

    assert (labeler_dir / "labels.xml").read_text(encoding="utf-8") == custom_xml


def test_prepare_writes_relative_ego_poses_from_earliest_timestamp(tmp_path: Path):
    height, width = 1, 1
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    write_xyz_csv(csv_dir / "frame_000.csv", height=height, width=width, timestamp_s=20, timestamp_u=0)
    write_xyz_csv(csv_dir / "frame_001.csv", height=height, width=width, timestamp_s=10, timestamp_u=0)

    litept = tmp_path / "litept"
    for frame_id, timestamp_us, ty in (("frame_000", 20_000_000, 4.0), ("frame_001", 10_000_000, 1.0)):
        frame_out = litept / frame_id
        frame_out.mkdir(parents=True)
        ego_pose = {
            "timestamp_us": timestamp_us,
            "source_timestamp_us": timestamp_us,
            "delta_us": 0,
            "translation": [0.0, ty, 0.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "kitti_pose": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, ty, 0.0, 0.0, 1.0, 0.0],
        }
        write_metadata(frame_out / "metadata.json", ego_pose=ego_pose)

    labeler_dir = tmp_path / "labeler"
    run_script(
        "prepare_for_point_labeler.py",
        "--csv-dir",
        str(csv_dir),
        "--litept-output-dir",
        str(litept),
        "--out-dir",
        str(labeler_dir),
        "--height",
        str(height),
        "--width",
        str(width),
    )

    pose_values = [float(x) for x in (labeler_dir / "poses.txt").read_text(encoding="utf-8").strip().split()]
    manifest = json.loads((labeler_dir / "bridge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["visualization_axis_mode"] == "ego_y_forward"
    assert pose_values == [
        1.0,
        0.0,
        0.0,
        3.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]

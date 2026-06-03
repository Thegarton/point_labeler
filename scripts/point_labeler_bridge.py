from __future__ import annotations

import bisect
import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_HEIGHT = 192
DEFAULT_WIDTH = 480
CHANNELS = 4
IDENTITY_KITTI_POSE = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]

DEFAULT_ELEVATION_DEG = (7.85, -12.85)
DEFAULT_AZIMUTH_DEG = (60.0, -60.0)


@dataclass(frozen=True)
class LoadedFrame:
    frame_id: str
    points_range: np.ndarray
    points_flat: np.ndarray
    timestamp_us: int | None
    timestamp_s: int | None
    timestamp_u: int | None
    meta: dict[str, Any]


@dataclass(frozen=True)
class InsPose:
    timestamp_us: int
    source_timestamp_us: int
    delta_us: int
    translation: list[float]
    rotation_matrix: list[list[float]]
    kitti_pose: list[float]
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InsSample:
    timestamp_us: int
    translation: list[float]
    rotation_matrix: list[list[float]]
    latitude: float | None
    longitude: float | None
    elevation: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None


class InsPoseIndex:
    def __init__(self, samples: list[InsSample]) -> None:
        if not samples:
            raise ValueError("INS pose index is empty")
        self.samples = sorted(samples, key=lambda sample: sample.timestamp_us)
        self.timestamps = [sample.timestamp_us for sample in self.samples]

    def nearest(self, timestamp_us: int) -> InsPose:
        pos = bisect.bisect_left(self.timestamps, timestamp_us)
        candidates: list[InsSample] = []
        if pos < len(self.samples):
            candidates.append(self.samples[pos])
        if pos > 0:
            candidates.append(self.samples[pos - 1])
        sample = min(candidates, key=lambda candidate: abs(candidate.timestamp_us - timestamp_us))
        delta_us = int(sample.timestamp_us - timestamp_us)
        return InsPose(
            timestamp_us=int(timestamp_us),
            source_timestamp_us=int(sample.timestamp_us),
            delta_us=delta_us,
            translation=sample.translation,
            rotation_matrix=sample.rotation_matrix,
            kitti_pose=flatten_kitti_pose(sample.rotation_matrix, sample.translation),
            latitude=sample.latitude,
            longitude=sample.longitude,
            elevation=sample.elevation,
            yaw=sample.yaw,
            pitch=sample.pitch,
            roll=sample.roll,
        )


def load_csv_frame(path: Path, *, frame_id: str, height: int, width: int) -> LoadedFrame:
    rows, fieldnames = read_csv_rows(path)
    if has_columns(fieldnames, ("x", "y", "z", "intensity")):
        return load_xyz_csv(rows, path=path, frame_id=frame_id, height=height, width=width)

    dis_cols = tuple(f"dis{i}" for i in range(height))
    int_cols = tuple(f"int{i}" for i in range(height))
    if has_columns(fieldnames, dis_cols) and has_columns(fieldnames, int_cols):
        return load_raw_packet_csv(rows, path=path, frame_id=frame_id, height=height, width=width)

    raise ValueError(
        f"Unsupported CSV layout in {path}. Expected x,y,z,intensity or raw packet dis0..dis{height - 1}."
    )


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header")
        fieldnames = tuple(clean_key(name) for name in reader.fieldnames)
        rows = [{clean_key(key): value for key, value in row.items()} for row in reader]
    return rows, fieldnames


def load_xyz_csv(
    rows: list[dict[str, str]], *, path: Path, frame_id: str, height: int, width: int
) -> LoadedFrame:
    xyz_i: list[list[float]] = []
    timestamp_s = None
    timestamp_u = None
    for row in rows:
        xyz_i.append([to_float(row["x"]), to_float(row["y"]), to_float(row["z"]), to_float(row["intensity"])])
        if timestamp_s is None and row.get("timestamp_s") not in (None, ""):
            timestamp_s = int(float(row["timestamp_s"]))
        if timestamp_u is None and row.get("timestamp_u") not in (None, ""):
            timestamp_u = int(float(row["timestamp_u"]))

    arr = np.asarray(xyz_i, dtype=np.float32)
    expected = height * width
    if arr.shape != (expected, CHANNELS):
        raise ValueError(f"CSV {path} has shape {arr.shape}, expected {(expected, CHANNELS)}")

    arr = arr.reshape(height, width, CHANNELS)
    timestamp_us = timestamp_to_us(timestamp_s, timestamp_u)
    return LoadedFrame(
        frame_id=frame_id,
        points_range=arr,
        points_flat=arr.reshape(-1, CHANNELS),
        timestamp_us=timestamp_us,
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        meta={"source_format": "csv", "csv_layout": "xyz"},
    )


def load_raw_packet_csv(
    rows: list[dict[str, str]], *, path: Path, frame_id: str, height: int, width: int
) -> LoadedFrame:
    if len(rows) != width:
        raise ValueError(f"Raw packet CSV {path} has {len(rows)} rows, expected {width} azimuth packets")

    rows = sort_raw_rows(rows)
    dis_cols = tuple(f"dis{i}" for i in range(height))
    int_cols = tuple(f"int{i}" for i in range(height))
    ref_cols = tuple(f"ref{i}" for i in range(height))

    distance = matrix_from_columns(rows, dis_cols)
    intensity = matrix_from_columns(rows, int_cols)
    reflectivity = matrix_from_columns(rows, ref_cols) if all(col in rows[0] for col in ref_cols) else None

    np.clip(distance, 0.0, None, out=distance)
    valid = np.isfinite(distance) & (distance > 0.0)

    elevation = np.deg2rad(np.linspace(DEFAULT_ELEVATION_DEG[0], DEFAULT_ELEVATION_DEG[1], height, dtype=np.float32))
    azimuth = np.deg2rad(np.linspace(DEFAULT_AZIMUTH_DEG[0], DEFAULT_AZIMUTH_DEG[1], width, dtype=np.float32))
    cos_el = np.cos(elevation)[None, :]
    sin_el = np.sin(elevation)[None, :]
    cos_az = np.cos(azimuth)[:, None]
    sin_az = np.sin(azimuth)[:, None]

    x = distance * cos_el * cos_az
    y = distance * cos_el * sin_az
    z = distance * sin_el
    strength = reflectivity if reflectivity is not None else intensity

    arr = np.stack([x, y, z, strength], axis=-1).transpose(1, 0, 2).astype(np.float32, copy=False)
    arr[~valid.T, :3] = 0.0
    arr[~valid.T, 3] = 0.0

    first = rows[0] if rows else {}
    timestamp_s = optional_int(first.get("timeStampData_s"))
    timestamp_u = optional_int(first.get("timeStampData_u"))
    timestamp_us = timestamp_to_us(timestamp_s, timestamp_u)
    return LoadedFrame(
        frame_id=frame_id,
        points_range=arr,
        points_flat=arr.reshape(-1, CHANNELS),
        timestamp_us=timestamp_us,
        timestamp_s=timestamp_s,
        timestamp_u=timestamp_u,
        meta={
            "source_format": "csv",
            "csv_layout": "raw_packet",
            "distance_channel": "dis",
            "strength_channel": "ref" if reflectivity is not None else "int",
            "elevation_deg": list(DEFAULT_ELEVATION_DEG),
            "azimuth_deg": list(DEFAULT_AZIMUTH_DEG),
        },
    )


def load_semantic_mask(path: Path, *, height: int, width: int) -> np.ndarray:
    mask = np.load(path, allow_pickle=False)
    arr = np.asarray(mask)
    if arr.shape != (height, width):
        raise ValueError(f"Semantic mask {path} has shape {arr.shape}, expected {(height, width)}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Semantic mask {path} dtype must be integer, got {arr.dtype}")
    return arr.astype(np.uint16, copy=False)


def labels_to_mask(labels: np.ndarray, *, height: int, width: int) -> np.ndarray:
    flat = np.asarray(labels, dtype=np.uint32)
    expected = height * width
    if flat.shape != (expected,):
        raise ValueError(f"Labels have shape {flat.shape}, expected {(expected,)}")
    return (flat & np.uint32(0xFFFF)).astype(np.uint16, copy=False).reshape(height, width)


def load_label_file(path: Path, *, height: int, width: int) -> np.ndarray:
    labels = np.fromfile(path, dtype=np.uint32)
    expected = height * width
    if labels.shape != (expected,):
        raise ValueError(f"Label file {path} has {labels.size} labels, expected {expected}")
    return labels


def write_kitti_pose_values(path: Path, kitti_pose: list[float]) -> None:
    if len(kitti_pose) != 12:
        raise ValueError(f"KITTI pose must have 12 values, got {len(kitti_pose)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(fmt_float(value) for value in kitti_pose) + "\n", encoding="utf-8")


def write_identity_calib(path: Path) -> None:
    path.write_text("Tr: " + " ".join(fmt_float(value) for value in IDENTITY_KITTI_POSE) + "\n", encoding="utf-8")


def read_kitti_calibration(path: Path) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        name, values = raw_line.split(":", 1)
        tokens = [x for x in values.strip().split() if x]
        if len(tokens) != 12:
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :4] = np.asarray([float(x) for x in tokens], dtype=np.float64).reshape(3, 4)
        matrices[name.strip()] = matrix
    if not matrices:
        raise ValueError(f"No 3x4 KITTI calibration matrices found in {path}")
    return matrices


def select_camera_projection(calibration: dict[str, np.ndarray], camera_id: str) -> np.ndarray:
    if camera_id in calibration:
        return calibration[camera_id]
    for fallback in ("P2", "P0", "P1", "P3"):
        if fallback in calibration:
            return calibration[fallback]
    raise KeyError(f"Calibration has no {camera_id} matrix and no P0/P1/P2/P3 fallback")


def select_lidar_to_camera(calibration: dict[str, np.ndarray]) -> np.ndarray:
    if "Tr" in calibration:
        return calibration["Tr"]
    if "T_lidar_to_camera" in calibration:
        return calibration["T_lidar_to_camera"]
    raise KeyError("Calibration has no Tr lidar_to_camera matrix")


def project_rgb_to_points(
    points_xyzi: np.ndarray,
    image_rgb: np.ndarray,
    projection: np.ndarray,
    tr: np.ndarray,
    *,
    calibration_image_size: tuple[int, int] | None = None,
) -> tuple[np.ndarray, dict]:
    points = np.asarray(points_xyzi, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points_xyzi must have shape [N,>=3], got {points.shape}")
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image_rgb must have shape [H,W,3], got {image.shape}")

    projection = scale_camera_projection(
        projection,
        calibration_image_size=calibration_image_size,
        target_image_shape=image.shape[:2],
    )
    points_h = np.ones((points.shape[0], 4), dtype=np.float64)
    points_h[:, :3] = points[:, :3]
    camera_h = (tr @ points_h.T).T
    camera_depth = camera_h[:, 2]
    projected = (projection @ camera_h.T).T

    rgb = np.zeros((points.shape[0], 3), dtype=np.uint8)
    valid_depth = camera_depth > 0.0
    valid_projected = valid_depth & (np.abs(projected[:, 2]) > 1e-9)
    u = np.full((points.shape[0],), -1, dtype=np.int64)
    v = np.full((points.shape[0],), -1, dtype=np.int64)
    u[valid_projected] = np.rint(projected[valid_projected, 0] / projected[valid_projected, 2]).astype(np.int64)
    v[valid_projected] = np.rint(projected[valid_projected, 1] / projected[valid_projected, 2]).astype(np.int64)

    height, width = image.shape[:2]
    inside = valid_projected & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    rgb[inside] = image[v[inside], u[inside]]
    summary = {
        "points": int(points.shape[0]),
        "projected": int(np.count_nonzero(inside)),
        "behind_camera": int(np.count_nonzero(~valid_depth)),
        "out_of_image": int(np.count_nonzero(valid_projected & ~inside)),
        "image_size": [int(width), int(height)],
        "calibration_image_size": list(calibration_image_size) if calibration_image_size is not None else None,
    }
    return rgb, summary


def scale_camera_projection(
    projection: np.ndarray,
    *,
    calibration_image_size: tuple[int, int] | None,
    target_image_shape: tuple[int, int],
) -> np.ndarray:
    matrix = np.asarray(projection, dtype=np.float64).copy()
    if calibration_image_size is None:
        return matrix

    source_width, source_height = calibration_image_size
    target_height, target_width = target_image_shape
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"calibration_image_size must be positive, got {calibration_image_size}")

    matrix[0, :] *= float(target_width) / float(source_width)
    matrix[1, :] *= float(target_height) / float(source_height)
    return matrix


def parse_image_size(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    normalized = value.strip().lower().replace(",", "x").replace("\t", "x").replace(" ", "x")
    tokens = [token for token in normalized.split("x") if token]
    if len(tokens) != 2:
        raise ValueError(f"Image size must be WIDTHxHEIGHT, got {value!r}")
    width, height = (int(float(token)) for token in tokens)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image size must be positive, got {value!r}")
    return width, height


def read_calibration_image_size(path: Path, *, camera_id: str | None = None) -> tuple[int, int] | None:
    generic_keys = {"image_size", "calibration_image_size", "source_image_size", "rgb_image_size", "resolution"}
    camera_keys = {key.lower() for key in camera_size_keys(camera_id)}
    generic_size = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        name, values = raw_line.split(":", 1)
        key = name.strip()
        tokens = [x for x in values.strip().split() if x]
        if len(tokens) < 2:
            continue
        value = f"{tokens[0]}x{tokens[1]}"
        if key.lower() in camera_keys:
            return parse_image_size(value)
        if key.lower() in generic_keys:
            generic_size = parse_image_size(value)

    return generic_size


def camera_size_keys(camera_id: str | None) -> set[str]:
    if not camera_id:
        return set()
    suffix = camera_id[1:] if camera_id.startswith("P") else camera_id
    suffix = suffix.zfill(2)
    return {
        f"S_{suffix}",
        f"S_rect_{suffix}",
        f"image_size_{camera_id}",
        f"calibration_image_size_{camera_id}",
    }


def read_velodyne_bin(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size % CHANNELS != 0:
        raise ValueError(f"Point cloud {path} has {arr.size} floats, not divisible by {CHANNELS}")
    return arr.reshape(-1, CHANNELS)


def write_point_rgb(path: Path, rgb: np.ndarray) -> None:
    arr = np.asarray(rgb, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"RGB point colors must have shape [N,3], got {arr.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)


def read_image_rgb(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if raw.startswith(b"P6"):
        return read_ppm_rgb(raw, source=path)
    try:
        import cv2  # type: ignore

        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is not None:
            return image_bgr[:, :, ::-1].astype(np.uint8, copy=False)
    except ImportError:
        pass
    try:
        from PIL import Image  # type: ignore

        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    except ImportError as exc:
        raise ImportError("Reading JPG/PNG images requires opencv-python or Pillow") from exc


def read_ppm_rgb(raw: bytes, *, source: Path) -> np.ndarray:
    offset = 0

    def next_token() -> bytes:
        nonlocal offset
        while offset < len(raw) and raw[offset:offset + 1].isspace():
            offset += 1
        if offset < len(raw) and raw[offset:offset + 1] == b"#":
            while offset < len(raw) and raw[offset:offset + 1] not in {b"\n", b"\r"}:
                offset += 1
            return next_token()
        start = offset
        while offset < len(raw) and not raw[offset:offset + 1].isspace():
            offset += 1
        return raw[start:offset]

    magic = next_token()
    if magic != b"P6":
        raise ValueError(f"Unsupported PPM magic in {source}: {magic!r}")
    width = int(next_token())
    height = int(next_token())
    max_value = int(next_token())
    if max_value != 255:
        raise ValueError(f"Only 8-bit PPM images are supported, got max value {max_value}")
    if offset < len(raw) and raw[offset:offset + 1].isspace():
        offset += 1
    expected = width * height * 3
    pixels = raw[offset:offset + expected]
    if len(pixels) != expected:
        raise ValueError(f"PPM {source} has {len(pixels)} pixel bytes, expected {expected}")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()


def load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_ins_pose_index(path: Path) -> InsPoseIndex:
    rows = read_ins_rows(path)
    origin = lat_lon_origin(rows)
    return InsPoseIndex([row_to_ins_sample(row, origin=origin) for row in rows])


def read_ins_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first_line = f.readline()
        delimiter = "\t" if "\t" in first_line else ","
        f.seek(0)
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"INS file {path} has no header")
        return [{clean_key(key): value for key, value in row.items()} for row in reader]


def row_to_ins_sample(row: dict[str, str], *, origin: tuple[float, float, float] | None) -> InsSample:
    secs = required_int(row, "secs")
    nsecs = required_int(row, "nsecs")
    timestamp_us = secs * 1_000_000 + nsecs // 1000
    lat = optional_float(row.get("latitude"))
    lon = optional_float(row.get("longitude"))
    elevation = optional_float(row.get("elevation"))
    translation = translation_from_row(row, lat=lat, lon=lon, elevation=elevation, origin=origin)
    roll = optional_float(row.get("attitude_X"))
    pitch = optional_float(row.get("attitude_Y"))
    yaw = optional_float(row.get("attitude_Z"))
    rotation = rotation_matrix_zyx(roll or 0.0, pitch or 0.0, yaw or 0.0)
    return InsSample(
        timestamp_us=timestamp_us,
        translation=translation,
        rotation_matrix=rotation.tolist(),
        latitude=lat,
        longitude=lon,
        elevation=elevation,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
    )


def translation_from_row(
    row: dict[str, str],
    *,
    lat: float | None,
    lon: float | None,
    elevation: float | None,
    origin: tuple[float, float, float] | None,
) -> list[float]:
    utm = [
        optional_float(row.get("utmPosition_X")) or 0.0,
        optional_float(row.get("utmPosition_Y")) or 0.0,
        optional_float(row.get("utmPosition_Z")) or 0.0,
    ]
    if np.linalg.norm(np.asarray(utm, dtype=np.float64)) > 1e-6:
        return utm
    if origin is not None and lat is not None and lon is not None:
        return lat_lon_to_local_enu(lat, lon, elevation or origin[2], origin)
    return [0.0, 0.0, elevation or 0.0]


def lat_lon_origin(rows: list[dict[str, str]]) -> tuple[float, float, float] | None:
    for row in rows:
        lat = optional_float(row.get("latitude"))
        lon = optional_float(row.get("longitude"))
        elevation = optional_float(row.get("elevation"))
        if lat is not None and lon is not None:
            return lat, lon, elevation or 0.0
    return None


def lat_lon_to_local_enu(lat: float, lon: float, elevation: float, origin: tuple[float, float, float]) -> list[float]:
    origin_lat, origin_lon, origin_elevation = origin
    earth_radius_m = 6_378_137.0
    east = math.radians(lon - origin_lon) * earth_radius_m * math.cos(math.radians(origin_lat))
    north = math.radians(lat - origin_lat) * earth_radius_m
    up = elevation - origin_elevation
    return [east, north, up]


def rotation_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def flatten_kitti_pose(rotation: list[list[float]], translation: list[float]) -> list[float]:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = np.asarray(translation, dtype=np.float64)
    return pose[:3, :].reshape(-1).tolist()


def write_labels_xml(path: Path, classes: list[tuple[str, int]]) -> None:
    root = ET.Element("config")
    for name, label_id in classes:
        label = ET.SubElement(root, "label")
        ET.SubElement(label, "id").text = str(label_id)
        ET.SubElement(label, "name").text = name
        ET.SubElement(label, "description").text = ""
        ET.SubElement(label, "color").text = " ".join(str(value) for value in label_color(name, label_id))
        ET.SubElement(label, "root").text = "manual"
        ET.SubElement(label, "macro").text = ""
        ET.SubElement(label, "category").text = name

    indent_xml(root)
    tree = ET.ElementTree(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def classes_from_metadata(metadata: dict[str, Any]) -> list[tuple[str, int]]:
    class_names = metadata.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        return []

    classes = [(str(name), idx) for idx, name in enumerate(class_names)]
    ignore_index = metadata.get("ignore_index")
    if ignore_index is not None:
        ignore_id = int(ignore_index)
        if all(label_id != ignore_id for _, label_id in classes):
            classes.append(("ignore", ignore_id))
    return sorted(classes, key=lambda item: item[1])


def read_semantic_classes(path: Path) -> list[tuple[str, int]]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return read_semantic_classes_without_yaml(path)

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    semantic = payload.get("semantic_classes", {})
    return sorted(((str(name), int(label_id)) for name, label_id in semantic.items()), key=lambda item: item[1])


def read_semantic_classes_without_yaml(path: Path) -> list[tuple[str, int]]:
    classes: list[tuple[str, int]] = []
    in_semantic = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if line.startswith("semantic_classes:"):
            in_semantic = True
            continue
        if in_semantic and line and not line.startswith(" "):
            break
        if in_semantic:
            stripped = line.strip()
            if ":" not in stripped:
                continue
            name, value = stripped.split(":", 1)
            classes.append((name.strip(), int(value.strip())))
    return sorted(classes, key=lambda item: item[1])


def label_color(name: str, label_id: int) -> tuple[int, int, int]:
    lowered = name.strip().lower()
    if label_id == 0 and lowered in {"background", "unlabeled", "void"}:
        return 0, 0, 0
    if label_id == 255:
        return 120, 120, 120
    seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(name)) + label_id * 9973
    return 40 + seed % 196, 40 + (seed // 7) % 196, 40 + (seed // 37) % 196


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def has_columns(fieldnames: tuple[str, ...], required: tuple[str, ...]) -> bool:
    available = set(fieldnames)
    return all(name in available for name in required)


def sort_raw_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if all(row.get("packSeqNum") not in (None, "") for row in rows):
        return sorted(rows, key=lambda row: int(float(row["packSeqNum"])))
    if all(row.get("axIdx0") not in (None, "") for row in rows):
        return sorted(rows, key=lambda row: int(float(row["axIdx0"])))
    return rows


def matrix_from_columns(rows: list[dict[str, str]], columns: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[to_float(row.get(col, "0")) for col in columns] for row in rows], dtype=np.float32)


def timestamp_to_us(timestamp_s: int | None, timestamp_u: int | None) -> int | None:
    if timestamp_s is None or timestamp_u is None:
        return None
    return int(timestamp_s) * 1_000_000 + int(timestamp_u)


def required_int(row: dict[str, str], key: str) -> int:
    value = row.get(key)
    if value in (None, ""):
        raise ValueError(f"INS row is missing required column {key!r}")
    return int(float(value))


def optional_int(value: str | float | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def to_float(value: str | float | int | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def clean_key(key: str | None) -> str:
    return "" if key is None else key.strip()


def fmt_float(value: float) -> str:
    return f"{float(value):.12g}"

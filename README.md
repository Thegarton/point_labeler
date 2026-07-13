# HL320 Point Labeler

`point_labeler` is used here as a manual correction tool for HL320 LiDAR point labels.
The current workflow is point-wise and flat: one CSV row, one LiDAR point, one label.

This README intentionally documents the HL320 workflow only.
p
## Build

### Ubuntu

```bash
sudo apt install cmake g++ git libeigen3-dev libboost-all-dev qtbase5-dev libglew-dev

cmake -S . -B build
cmake --build build --parallel
```

The executable is written to:

```bash
./bin/labeler
```

### Windows

Use Visual Studio 2022, CMake, and vcpkg.

```powershell
cd C:\src
git clone https://github.com/microsoft/vcpkg.git
cd C:\src\vcpkg
.\bootstrap-vcpkg.bat
```

Build from **Developer PowerShell for VS 2022**:

```powershell
cd C:\src\point_labeler

cmake -S . -B build-windows `
  -G "Visual Studio 17 2022" `
  -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake `
  -DVCPKG_TARGET_TRIPLET=x64-windows `
  -DPOINT_LABELER_BUILD_TESTS=OFF

cmake --build build-windows --config Release --parallel
```

The executable is written to:

```powershell
.\bin\labeler.exe
```

If Qt DLLs are missing at startup:

```powershell
C:\src\vcpkg\installed\x64-windows\tools\Qt5\bin\windeployqt.exe .\bin\labeler.exe
```

The viewer needs a GPU/driver with OpenGL Core Profile support. Remote desktop and VMs may expose too old an OpenGL
context.

## HL320 Data Model

HL320 point tables are flat CSV files. The point order in the CSV is the source of truth.

Required columns:

```text
x y z intensity
```

Useful HL320 columns:

```text
azimuth vertical slot pixel hcell vcell Cxd Cyd
```

Important:

- `slot` and `pixel` are acquisition indices. They are not image axes and are not a reliable range-image layout.
- Do not reshape labels by `slot/pixel` for training or export.
- `Cxd` and `Cyd` are the camera-image coordinates for each LiDAR point.
- Point `i` in CSV row `i` must correspond to label `i` in `labels/<frame>.label`.
- Exported masks for this workflow are 1D arrays: `semantic_mask.npy.shape == (N,)`.

## Dataset Layout

A labeler dataset should look like this:

```text
HL320_output/
├── csv/
│   ├── 000000.csv
│   ├── 000001.csv
│   └── ...
├── velodyne/
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
├── labels/
│   ├── 000000.label
│   ├── 000001.label
│   └── ...
├── image_2/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ...
├── point_rgb/                 # optional
├── labels.xml
├── settings.cfg
├── bridge_manifest.json
├── calib.txt                  # identity is fine for this flat workflow
└── poses.txt                  # identity poses are fine for this flat workflow
```

`velodyne/<frame>.bin` is raw float32 `N x 4` in this order:

```text
x y z intensity
```

`labels/<frame>.label` is raw uint32 with exactly `N` labels.

`bridge_manifest.json` must mark this as a flat-point dataset:

```json
{
  "label_layout": "flat_points",
  "frames": [
    {
      "frame_id": "000000",
      "label_layout": "flat_points",
      "source_csv": "/path/to/csv/000000.csv",
      "point_count": 19968
    }
  ]
}
```

The exporter uses `label_layout: flat_points` to write 1D `semantic_mask.npy` files instead of old 2D range images.

## Settings

For HL320 flat data, `settings.cfg` should contain:

```text
allow velodyne only: true
point cloud source: velodyne
csv image width: 1920
csv image height: 1536
```

Notes:

- Use `point cloud source: velodyne` for manual labeling when the dataset already contains converted `.bin` files.
- Keep the matching original CSV files in `csv/` so the image viewer and external visualization scripts can use `Cxd/Cyd`.
- `allow velodyne only: true` lets the tool run even with identity/missing real poses and calibration.
- Reduce `max scans` if too many identity-pose scans are loaded together.

## Labels

The current HL320 taxonomy is defined in `labels.xml`. Example classes:

```text
0   background
2   CAR
5   PEDESTRIAN
10  traffic_sign
11  roadblock
13  tire
15  traffic_cone
18  adhesive_noise
19  Long_distance_noise
20  underground_noise
21  Multiple_Distances
255 ignore
```

Use `background` for ordinary visible scene points that should participate in training.
Use `ignore` only for points that should not contribute to training loss.

## Viewing And Editing

Start the tool:

```bash
./bin/labeler
```

Then select the dataset directory, for example:

```text
/home/a60116606/git_repo/point_labeler/HL320_output_sam3_manual_104
```

Useful controls:

- `W/A/S/D`: move the camera.
- Hold `Shift` while moving: faster camera movement.
- Visuals tab `camera RGB`: show precomputed/projected camera colors when `point_rgb/` is available.
- Visuals tab `intensity`: color by intensity using the plasma gradient.
- Visuals tab `sphere shadows`: toggle sphere shading while keeping points rendered as spheres.

The image viewer uses the CSV projection columns `Cxd/Cyd` to draw LiDAR points over `image_2/<frame>.jpg`.
It does not use `slot/pixel` as a camera or range-image coordinate system.

## Export Corrected Labels

After manual correction, export labels back to the flat LitePT/fine-tune layout.

Create a script such as:

```bash
#!/bin/bash
set -e

POINT_LABELER_ROOT="/home/a60116606/git_repo/point_labeler"
LABELER_DIR="$POINT_LABELER_ROOT/HL320_output_sam3_manual_104"
EXPORT_DIR="$LABELER_DIR/export"

cd "$POINT_LABELER_ROOT/scripts"

python3 export_from_point_labeler.py \
  --labeler-dir "$LABELER_DIR" \
  --out-dir "$EXPORT_DIR" \
  --pseudo-label-version "manual_point_labeler_hl320_v0" \
  --provenance "human_corrected" \
  --manual-reviewed
```

Run it:

```bash
bash run_export_hl320.sh
```

Expected output:

```text
HL320_output_sam3_manual_104/export/
├── 000000/
│   ├── semantic_mask.npy      # 1D uint16, shape (N,)
│   └── metadata.json
├── 000001/
│   ├── semantic_mask.npy
│   └── metadata.json
└── ...
```

The exported `metadata.json` is generated from the current `labels.xml`, so class ids, names, and classes added in the
UI are preserved.s
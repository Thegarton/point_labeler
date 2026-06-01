# Point Cloud Labeling Tool

 Tool for labeling of a single point clouds or a stream of point clouds. 
 
<img src="https://user-images.githubusercontent.com/11506664/63230808-340d5680-c212-11e9-8902-bc08f0f64dc8.png" width=500>

 Given the poses of a KITTI point cloud dataset, we load tiles of overlapping point clouds. Thus, multiple point clouds are labeled at once in a certain area. 

## Features
 - Support for KITTI Vision Benchmark Point Clouds.
 - Human-readable label description files in xml allow to define label names, ids, and colors.
 - Modern OpenGL shaders for rendering of even millions of points.
 - Tools for labeling of individual points and polygons.
 - Filtering of labels makes it easy to label even complicated structures with ease.

## Dependencies

* Eigen >= 3.2
* boost >= 1.54
* QT >= 5.2
* OpenGL Core Profile >= 4.0
 
## Build
  
On Ubuntu 22.04/20.04, the dependencies can be installed from the package manager:
```bash
sudo apt install cmake g++ git libeigen3-dev libboost-all-dev qtbase5-dev libglew-dev
```

Then, build the project, change to the cloned directory and use the following commands:
```bash
cmake -S . -B build
cmake --build build
```

Alternatively, you can also use the "classical" cmake build procedure:
```bash
mkdir build && cd build
cmake ..
make -j5
```


Now the project root directory (e.g. `~/point_labeler`) should contain a `bin` directory containing the labeler.


## Usage


In the `bin` directory, just run `./labeler` to start the labeling tool. 

The labeling tool allows to label a sequence of point clouds in a tile-based fashion, i.e., the tool loads all scans overlapping with the current tile location.
Thus, you will always label the part of the scans that overlaps with the current tile.


In the `settings.cfg` files you can change the following options:

<pre>

tile size: 100.0   # size of a tile (the smaller the less scans get loaded.)
max scans: 500    # number of scans to load for a tile. (should be maybe 1000), but this is currently very memory-consuming.
min range: 0.0    # minimum distance of points to consider.
max range: 50.0   # maximum distance of points in the point cloud.
add car points: true # add points at the origin of the sensor possibly caused by the car itself. Default: false.

</pre>

## CSV/LitePT bridge

CSV frames and LitePT semantic masks can be converted to the KITTI-like layout expected by this tool:

```bash
python3 scripts/prepare_for_point_labeler.py \
  --csv-dir /path/to/data/2026_01_12_13_16_21_frames_192 \
  --litept-output-dir /path/to/output/2026_01_12_13_16_21_frames_192/litept_waymo \
  --out-dir /path/to/labeler_dataset
```

The generated dataset contains `velodyne/`, `labels/`, `poses.txt`, `calib.txt`, `labels.xml`,
`bridge_manifest.json`, `settings.cfg`, and `settings.cfg.example`. Use the generated `labels file` setting so the UI
uses the same semantic ids as the LitePT masks. Class names are read from the LitePT per-frame
`metadata.json` `class_names` field; `--classes-yaml` is only a fallback when metadata is missing.
If the CSV directory contains synced camera frames named `frame_id.jpg`, `frame_id.jpeg`, or `frame_id.png`,
they are copied into `image_2/` and recorded in `bridge_manifest.json`.
For labeler visualization, `poses.txt` is written relative to the earliest ego pose. The default
`--visualization-axis-mode ego_y_forward` maps ego/INS `+Y` motion to KITTI/Velodyne `+X` forward;
use `--visualization-axis-mode kitti_x_forward` if your pose matrices are already in KITTI axes.

After manual editing and saving in the labeler, export corrected masks back to the LitePT layout:

```bash
python3 scripts/export_from_point_labeler.py \
  --labeler-dir /path/to/labeler_dataset \
  --out-dir /path/to/corrected_litept_out
```

If existing LitePT metadata contains `ego_pose`, it is preserved. Otherwise, when an INS file is
available, the converter matches each frame to the nearest INS row by timestamp and writes `ego_pose`
metadata plus KITTI-style `pose.txt` on export. By default it looks for either `<csv-dir>/ins` as a
single file or one `*.csv`/`*.tsv` file inside the `<csv-dir>/ins/` directory; use `--ins-path` to
override that.

## Camera RGB point coloring

To color points from the synchronized camera image in the UI, prepare a real KITTI-style `calib.txt`
with a camera projection matrix such as `P2` and LiDAR-to-camera `Tr`, then precompute point colors:

```bash
python3 scripts/prepare_for_point_labeler.py \
  --csv-dir /path/to/data \
  --litept-output-dir /path/to/litept_or_fused_masks \
  --calib-file /path/to/calib.txt \
  --precompute-rgb \
  --out-dir /path/to/labeler_dataset
```

This writes `point_rgb/<frame_id>.rgb` as raw `uint8` RGB triplets aligned with `velodyne/<frame_id>.bin`.
In the labeler UI, enable `camera RGB` in the Visuals tab to switch from class/remission coloring to image
RGB coloring. Points behind the camera, outside the image, or missing RGB data are shown as black.

You can also precompute RGB for an existing labeler dataset:

```bash
python3 scripts/precompute_point_rgb.py \
  --dataset-dir /path/to/labeler_dataset \
  --camera-id P2 \
  --overwrite
```




 
## Folder structure

When loading a dataset, the data must be organized as follows:

<pre>
point cloud folder
├── velodyne/             -- directory containing ".bin" files with Velodyne point clouds.   
├── labels/   [optional]  -- label directory, will be generated if not present.  
├── image_2/  [optional]  -- directory containing ".png" files from the color   camera.  
├── calib.txt             -- calibration of velodyne vs. camera. needed for projection of point cloud into camera.  
└── poses.txt             -- file containing the poses of every scan.
</pre>

 

## Documentation

See the [wiki](https://github.com/jbehley/point_labeler/wiki) for more information on the usage and other details.


 ## Citation

If you're using the tool in your research, it would be nice if you cite our [paper](https://arxiv.org/abs/1904.01416):

```
@inproceedings{behley2019iccv,
    author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
     title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
 booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
      year = {2019}
}
```

We used the tool to label SemanticKITTI, which contains overall over 40.000 scans organized in 20 sequences. 

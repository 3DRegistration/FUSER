
## Benchmark Dataset Demonstration

Prepare the benchmark folders before running evaluation. The repository expects the following directory layouts.

### ScanNet

Each ScanNet scene should be organized as:

```text
<SCANNET_ROOT>/
  scene0197_01/
    PointCloud/
      cloud_bin_0.ply
      cloud_bin_1.ply
      ...
      pose_0.txt
      pose_1.txt
      ...
      gt.log
```

Notes:

- `gt.log` is required for evaluation.
- the current loader uses the first `30` `cloud_bin_*.ply` files in each scene
- the number of `pose_*.txt` files must match the number of point clouds used
- evaluation reads ground-truth pairwise trajectories from `PointCloud/gt.log`


### 3DMatchFrame / ArkitScenes

Each frame-based benchmark scene should be organized as:

```text
<FRAME_BENCHMARK_ROOT>/
  <scene_id>/
    PointCloud/
      cloud_bin_0.ply
      cloud_bin_1.ply
      ...
      gt.log
    Depth/
      cloud_bin_0.depth.png
      cloud_bin_1.depth.png
      ...
    Intrinsic/
      cloud_bin_0.txt
      cloud_bin_1.txt
      ...
    Extrinsic/
      cloud_bin_0.txt
      cloud_bin_1.txt
      ...
    sequence_meta.json
```

Notes:

- `PointCloud`, `Depth`, `Intrinsic`, and `Extrinsic` are all expected to exist
- the number of files in these subdirectories should be aligned per scene
- evaluation reads ground-truth pairwise trajectories from `PointCloud/gt.log`

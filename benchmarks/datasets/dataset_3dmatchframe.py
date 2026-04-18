import json
import re
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch

from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset

CLOUD_BIN_PATTERN = re.compile(r"cloud_bin_(\d+)")

@dataclass
class BenchmarkSceneRecord:
    scene_id: str
    point_cloud_paths: list
    extrinsic_paths: list
    depth_paths: list
    intrinsic_paths: list

def cloud_bin_sort_key(path):
    match = CLOUD_BIN_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Unexpected benchmark filename: {path}")
    return int(match.group(1))


class ThreeDMatchFrame(Dataset):
    def __init__(
        self,
        data_root,
        voxel_size=0.03,
        n_max_points=15000,
        scene_ids=None,
        seed=12345,
    ):
        super().__init__()
        self.data_root = Path(data_root).expanduser().resolve()
        self.voxel_size = float(voxel_size)
        self.n_max_points = int(n_max_points)
        self.scene_ids = scene_ids
        self.seed = int(seed)
        self.records = self._build_records()

    def _load_sequence_meta(self, scene_root):
        meta_path = scene_root / "sequence_meta.json"
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _build_records(self):
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"3DMatch benchmark dataset root not found: {self.data_root}")

        records = []
        for scene_id in self.scene_ids:
            scene_root = self.data_root / scene_id

            point_cloud_paths = sorted((scene_root / "PointCloud").glob("cloud_bin_*.ply"), key=cloud_bin_sort_key)
            extrinsic_paths = sorted((scene_root / "Extrinsic").glob("cloud_bin_*.txt"), key=cloud_bin_sort_key)
            depth_paths = sorted((scene_root / "Depth").glob("cloud_bin_*.*"), key=cloud_bin_sort_key)
            intrinsic_paths = sorted((scene_root / "Intrinsic").glob("cloud_bin_*.txt"), key=cloud_bin_sort_key)
            assert len(point_cloud_paths) == len(extrinsic_paths) == len(depth_paths) == len(intrinsic_paths)

            records.append(
                BenchmarkSceneRecord(
                    scene_id=scene_id,
                    point_cloud_paths=point_cloud_paths,
                    extrinsic_paths=extrinsic_paths,
                    depth_paths=depth_paths,
                    intrinsic_paths=intrinsic_paths,
                )
            )

        if not records:
            raise RuntimeError(f"No valid 3DMatchFrame scenes found under {self.data_root}")
        return records

    def __len__(self):
        return len(self.records)

    def _load_point_cloud(self, path, sample_seed):
        point_cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(point_cloud.points, dtype=np.float32)
        if points.size == 0:
            raise ValueError(f"Empty point cloud: {path}")

        points = torch.from_numpy(points)
        _, unique_indices = ME.utils.sparse_quantize(
            torch.floor(points / self.voxel_size).contiguous(),
            return_index=True,
        )
        points = points[unique_indices].float()

        if points.shape[0] > self.n_max_points:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(sample_seed)
            selected = torch.randperm(points.shape[0], generator=generator)[: self.n_max_points]
            points = points[selected]

        return points

    def _load_pose(self, path):
        return torch.from_numpy(np.loadtxt(path, dtype=np.float32).reshape(4, 4)).float()

    def _build_relative_pose_dict(self, pose_seq):
        relative_pose = {}
        for i in range(len(pose_seq) - 1):
            for j in range(i + 1, len(pose_seq)):
                relative_pose[f"{j}=>{i}"] = torch.linalg.inv(pose_seq[i]) @ pose_seq[j]
        return relative_pose

    def __getitem__(self, index):
        record = self.records[index]
        pcd_seq = []
        pose_seq = []
        for view_idx, (point_cloud_path, extrinsic_path) in enumerate(zip(record.point_cloud_paths, record.extrinsic_paths)):
            sample_seed = self.seed + index * 1000 + view_idx
            pcd_seq.append(self._load_point_cloud(point_cloud_path, sample_seed))
            pose_seq.append(self._load_pose(extrinsic_path))

        return {
            "pcd_seq": pcd_seq,
            "pose_seq": pose_seq,
            "relative_pose": self._build_relative_pose_dict(pose_seq),
            "feat_seq": [torch.ones_like(points[:, [0]]) for points in pcd_seq],
            "coord_seq": [torch.floor(points / self.voxel_size) for points in pcd_seq],
            "num_views": len(pcd_seq),
            "index": index,
            "s_id": record.scene_id,
            "voxel_size": self.voxel_size
        }

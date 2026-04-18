import torch
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import Dataset

@dataclass
class SceneRecord:
    scene_id: str
    gt_log_path: str
    point_cloud_paths: list
    pose_paths: list
    frame_ids: list

class ScanNet(Dataset):
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

    def _build_records(self):
        if not self.data_root.is_dir():
            raise FileNotFoundError(f"ScanNet benchmark dataset root not found: {self.data_root}")

        records = []
        for scene_id in self.scene_ids:
            scene_root = self.data_root / scene_id / "PointCloud"
            gt_log_path = scene_root / "gt.log"
            if not gt_log_path.is_file():
                raise FileNotFoundError(f"No gt file found: {gt_log_path}")

            point_cloud_paths = sorted(
                scene_root.glob("cloud_bin_*.ply"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )[:30]
            if not point_cloud_paths:
                raise FileNotFoundError(f"No point clouds found under {scene_root}")

            pose_paths = sorted(
                scene_root.glob("pose_*.txt"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )

            frame_ids = [int(path.stem.split("_")[-1]) for path in point_cloud_paths]

            assert len(frame_ids) == len(point_cloud_paths) == len(pose_paths) == 30
            records.append(SceneRecord(scene_id=scene_id, gt_log_path=gt_log_path, point_cloud_paths=point_cloud_paths, pose_paths=pose_paths, frame_ids=frame_ids))

        if not records:
            raise RuntimeError(f"No valid ScanNet scenes found under {self.data_root}")

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

    def _build_relative_pose_dict(self, pose_seq):
        relative_pose = {}
        for i in range(len(pose_seq) - 1):
            for j in range(i + 1, len(pose_seq)):
                relative_pose[f"{j}=>{i}"] = torch.linalg.inv(pose_seq[i]) @ pose_seq[j]
        return relative_pose

    def _load_pose(self, path):
        return torch.from_numpy(np.loadtxt(path, dtype=np.float32).reshape(4, 4)).float()

    def __getitem__(self, index):
        record = self.records[index]
        pcd_seq = []
        pose_seq = []
        for view_idx, point_cloud_path in enumerate(record.point_cloud_paths):
            sample_seed = self.seed + index * 1000 + view_idx
            pcd_seq.append(self._load_point_cloud(point_cloud_path, sample_seed))
            pose_seq.append(self._load_pose(record.pose_paths[view_idx]))

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

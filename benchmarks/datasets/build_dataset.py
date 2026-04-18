from .dataset_specs import get_dataset_spec
import MinkowskiEngine as ME

def collate_fn(examples):
    pcd_seqs = [example["pcd_seq"] for example in examples]
    feat_seqs = [example["feat_seq"] for example in examples]
    coord_seqs = [example["coord_seq"] for example in examples]
    coord_seqs_mink, feat_seqs_mink = ME.utils.sparse_collate(sum(coord_seqs, []), sum(feat_seqs, []))
    return {
        "pcd_seqs": pcd_seqs,
        "coord_seqs_mink": coord_seqs_mink,
        "feat_seqs_mink": feat_seqs_mink,
        "s_ids": [example["s_id"] for example in examples],
        "relative_pose_seqs": [example["relative_pose"] for example in examples],
        "gt_pose_seqs": [example["pose_seq"] for example in examples],
        "voxel_size": [example["voxel_size"] for example in examples]
    }

def build_dataset(cfg):
    spec = get_dataset_spec(cfg.data.benchmark_name)
    if cfg.data.benchmark_name == "ScanNet":
        from .dataset_scannet import ScanNet
        dataset = ScanNet(
            data_root=spec["data_root"],
            voxel_size=spec["voxel_size"],
            n_max_points=spec["n_max_points"],
            scene_ids=spec["scene_ids"],
            seed=cfg.seed,
        )
    elif cfg.data.benchmark_name == "3DMatchFrame":
        from .dataset_3dmatchframe import ThreeDMatchFrame
        dataset = ThreeDMatchFrame(
            data_root=spec["data_root"],
            voxel_size=spec["voxel_size"],
            n_max_points=spec["n_max_points"],
            scene_ids=spec["scene_ids"],
            seed=cfg.seed,
        )
    elif cfg.data.benchmark_name == "ArkitScenes":
        from .dataset_arkitscenes import ArkitScenes
        dataset = ArkitScenes(
            data_root=spec["data_root"],
            voxel_size=spec["voxel_size"],
            n_max_points=spec["n_max_points"],
            scene_ids=spec["scene_ids"],
            seed=cfg.seed,
        )
    else:
        raise NotImplementedError

    return dataset, collate_fn
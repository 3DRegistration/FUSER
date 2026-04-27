import argparse
import json
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
try:
    import MinkowskiEngine as ME
except ImportError:
    ME = None

if __package__ and "." in __package__:
    from .cfgs.cfg import cfg as base_cfg
    from .models.model_factory import SUPPORTED_MODEL_NAMES, build_model, load_model_weights
    from .benchmarks.export import export_pred_poses
else:
    REPO_ROOT = Path(__file__).resolve().parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from cfgs.cfg import cfg as base_cfg
    from models.model_factory import SUPPORTED_MODEL_NAMES, build_model, load_model_weights
    from benchmarks.export import export_pred_poses


REPO_ROOT = Path(__file__).resolve().parent
EXAMPLES_ROOT = REPO_ROOT / "examples"
SUPPORTED_POINT_CLOUD_SUFFIXES = {".ply", ".pcd", ".pts", ".xyz", ".xyzn", ".xyzrgb"}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    key = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return key


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    text = text.strip("._-")
    return text or "sequence"


def discover_sequences(search_root: Path):
    if not search_root.is_dir():
        return {}

    discovered = {}
    for directory in sorted(search_root.rglob("*")):
        if not directory.is_dir():
            continue

        file_paths = sorted(
            [
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
            ],
            key=natural_key,
        )
        if len(file_paths) < 2:
            continue

        if directory.name == "PointCloud" and directory.parent != search_root:
            label = str(directory.parent.relative_to(search_root))
            discovered[label] = directory
        else:
            label = str(directory.relative_to(search_root))
            discovered[label] = directory

    return dict(sorted(discovered.items(), key=lambda item: item[0].lower()))


def print_available_examples():
    example_map = discover_sequences(EXAMPLES_ROOT)
    if not example_map:
        print(f"No valid example sequences were found under {EXAMPLES_ROOT}.")
        return

    print("Available example sequences:")
    for name in example_map:
        print(f"  - {name}")


def ensure_minkowski_available():
    if ME is None:
        raise ImportError(
            "MinkowskiEngine is required for demo inference but is not installed in the current environment."
        )


def resolve_sequence_dir(data_path: str, example_name: str):
    input_path = Path(data_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    if input_path.is_file():
        raise ValueError(
            f"Expected a directory containing a point cloud sequence, got a file instead: {input_path}"
        )

    if example_name:
        example_map = discover_sequences(input_path)
        if example_name not in example_map:
            available = ", ".join(example_map.keys()) if example_map else "(none)"
            raise ValueError(
                f"Example `{example_name}` was not found under {input_path}. Available sequences: {available}"
            )
        return example_name, example_map[example_name]

    direct_files = sorted(
        [
            path for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
        ],
        key=natural_key,
    )
    if len(direct_files) >= 2:
        return input_path.name, input_path

    pointcloud_dir = input_path / "PointCloud"
    pointcloud_files = sorted(
        [
            path for path in pointcloud_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
        ],
        key=natural_key,
    ) if pointcloud_dir.is_dir() else []
    if len(pointcloud_files) >= 2:
        return input_path.name, pointcloud_dir

    discovered = discover_sequences(input_path)
    if len(discovered) == 1:
        name, sequence_dir = next(iter(discovered.items()))
        return name, sequence_dir
    if len(discovered) > 1:
        available = ", ".join(discovered.keys())
        raise ValueError(
            f"Multiple valid sequences were found under {input_path}. "
            f"Please pass --example_name. Available sequences: {available}"
        )

    raise ValueError(
        f"No valid point cloud sequence was found under {input_path}. "
        f"Expected at least two files with suffixes: {sorted(SUPPORTED_POINT_CLOUD_SUFFIXES)}"
    )


def collect_point_cloud_paths(sequence_dir: Path):
    file_paths = sorted(
        [
            path for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
        ],
        key=natural_key,
    )
    if len(file_paths) < 2:
        raise ValueError(f"Sequence directory must contain at least two point clouds: {sequence_dir}")
    return file_paths


def load_point_cloud_for_model(path: Path, voxel_size: float, n_max_points: int, sample_seed: int):
    ensure_minkowski_available()
    point_cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(point_cloud.points, dtype=np.float32)
    if points.size == 0:
        raise ValueError(f"Empty point cloud: {path}")

    points = torch.from_numpy(points)
    _, unique_indices = ME.utils.sparse_quantize(
        torch.floor(points / voxel_size).contiguous(),
        return_index=True,
    )
    points = points[unique_indices].float()

    if points.shape[0] > n_max_points:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(sample_seed)
        selected = torch.randperm(points.shape[0], generator=generator)[:n_max_points]
        points = points[selected]

    return points


def build_demo_batch(point_cloud_paths, voxel_size: float, n_max_points: int):
    ensure_minkowski_available()
    pcd_seq = []
    coord_seq = []
    feat_seq = []
    point_counts = []

    for idx, path in enumerate(point_cloud_paths):
        points = load_point_cloud_for_model(
            path=path,
            voxel_size=voxel_size,
            n_max_points=n_max_points,
            sample_seed=base_cfg.seed + idx,
        )
        pcd_seq.append(points)
        coord_seq.append(torch.floor(points / voxel_size))
        feat_seq.append(torch.ones_like(points[:, [0]]))
        point_counts.append(int(points.shape[0]))

    coord_seqs_mink, feat_seqs_mink = ME.utils.sparse_collate(coord_seq, feat_seq)
    batch = {
        "pcd_seqs": [pcd_seq],
        "coord_seqs_mink": coord_seqs_mink,
        "feat_seqs_mink": feat_seqs_mink,
        "s_ids": ["demo_sequence"],
        "relative_pose_seqs": [{}],
        "gt_pose_seqs": [None],
        "voxel_size": [float(voxel_size)],
    }
    return batch, point_counts


def move_sparse_inputs_to_device(batch, device):
    batch["coord_seqs_mink"] = batch["coord_seqs_mink"].to(device)
    batch["feat_seqs_mink"] = batch["feat_seqs_mink"].to(device)
    batch["voxel_size"] = batch["voxel_size"][0]
    return batch


def run_inference(model, batch, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    batch = move_sparse_inputs_to_device(batch, device)
    with torch.no_grad():
        output = model(batch)

    peak_mem = 0.0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3
    return output, peak_mem


def make_demo_cfg(args):
    cfg = deepcopy(base_cfg)
    cfg.model.model_name = args.model_name
    cfg.model.fuser_checkpoint = args.fuser_checkpoint
    cfg.model.prior_checkpoint = args.prior_checkpoint
    cfg.model.surrogate_checkpoint = args.surrogate_checkpoint
    return cfg


def resolve_device(device_arg: str):
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def normalize_poses_to_first(camera_poses):
    if camera_poses.shape[0] == 0:
        return camera_poses
    first_inv = np.linalg.inv(camera_poses[0])
    return np.matmul(first_inv[None], camera_poses)


def apply_pose_to_points(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def build_aligned_point_cloud(aligned_scan_points):
    merged_points = np.concatenate(aligned_scan_points, axis=0)
    palette = np.asarray(
        [
            [66, 133, 244],
            [234, 67, 53],
            [251, 188, 5],
            [52, 168, 83],
            [168, 85, 247],
            [14, 165, 233],
            [249, 115, 22],
            [236, 72, 153],
        ],
        dtype=np.uint8,
    )
    colors = []
    for idx, points in enumerate(aligned_scan_points):
        color = palette[idx % len(palette)]
        colors.append(np.repeat(color[None], points.shape[0], axis=0))
    merged_colors = np.concatenate(colors, axis=0)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(merged_points.astype(np.float64))
    point_cloud.colors = o3d.utility.Vector3dVector((merged_colors / 255.0).astype(np.float64))
    return point_cloud


def save_demo_outputs(
    output_dir: Path,
    sequence_name: str,
    point_cloud_paths,
    aligned_scan_points,
    camera_poses,
    raw_camera_poses,
    prior_camera_poses,
    runtime_sec: float,
    peak_mem_gb: float,
    save_path: str,
    normalize_to_first: bool,
    model_name: str,
    checkpoint_info,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_path is None:
        aligned_ply_path = output_dir / "aligned_sequence.ply"
    else:
        aligned_ply_path = Path(save_path).expanduser().resolve()
        aligned_ply_path.parent.mkdir(parents=True, exist_ok=True)

    aligned_point_cloud = build_aligned_point_cloud(aligned_scan_points)
    success = o3d.io.write_point_cloud(str(aligned_ply_path), aligned_point_cloud)
    if not success:
        raise RuntimeError(f"Failed to write aligned point cloud to {aligned_ply_path}")

    pose_dir = export_pred_poses(
        output_dir=str(output_dir),
        scene_id=slugify(sequence_name),
        pred_pose_seq=camera_poses,
    )

    payload = {
        "sequence_name": np.asarray(sequence_name),
        "scan_names": np.asarray([path.name for path in point_cloud_paths], dtype=str),
        "camera_poses": camera_poses.astype(np.float32),
        "raw_camera_poses": raw_camera_poses.astype(np.float32),
        "aligned_points": np.concatenate(aligned_scan_points, axis=0).astype(np.float32),
        "runtime_sec": np.asarray(runtime_sec, dtype=np.float32),
        "peak_mem_gb": np.asarray(peak_mem_gb, dtype=np.float32),
    }
    if prior_camera_poses is not None:
        payload["prior_camera_poses"] = prior_camera_poses.astype(np.float32)

    npz_path = output_dir / "demo_results.npz"
    np.savez_compressed(npz_path, **payload)

    if model_name == "FUSER":
        checkpoint_summary = {"checkpoint_path": str(checkpoint_info["checkpoint_path"])}
    else:
        checkpoint_summary = {
            "prior_path": str(checkpoint_info["prior_path"]),
            "surrogate_path": str(checkpoint_info["surrogate_path"]),
        }

    summary = {
        "sequence_name": sequence_name,
        "model_name": model_name,
        "num_scans": len(point_cloud_paths),
        "normalize_to_first": normalize_to_first,
        "runtime_sec": runtime_sec,
        "peak_mem_gb": peak_mem_gb,
        "aligned_point_cloud": str(aligned_ply_path),
        "pose_dir": str(pose_dir),
        "npz_path": str(npz_path),
        "checkpoint_info": checkpoint_summary,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return aligned_ply_path, pose_dir, npz_path, summary_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FUSER or FUSER-DF on a point cloud sequence and export aligned demo results."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(EXAMPLES_ROOT),
        help="Path to a sequence directory, a scene root containing PointCloud/, or the examples root.",
    )
    parser.add_argument(
        "--example_name",
        type=str,
        default=None,
        help="Optional sequence name under --data_path when multiple example sequences exist.",
    )
    parser.add_argument(
        "--list_examples",
        action="store_true",
        help="List valid example sequences under examples/ and exit.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        choices=SUPPORTED_MODEL_NAMES,
        default="FUSER",
        help="Model to run. Choose FUSER or FUSER_DF.",
    )
    parser.add_argument(
        "--fuser_checkpoint",
        "--checkpoint",
        dest="fuser_checkpoint",
        type=str,
        default=str(REPO_ROOT / "ckpts" / "fuser.safetensors"),
        help="Checkpoint for FUSER. Also used as default prior checkpoint path.",
    )
    parser.add_argument(
        "--prior_checkpoint",
        type=str,
        default=str(REPO_ROOT / "ckpts" / "fuser.safetensors"),
        help="Prior checkpoint for FUSER_DF.",
    )
    parser.add_argument(
        "--surrogate_checkpoint",
        type=str,
        default=str(REPO_ROOT / "ckpts" / "fuser_df.safetensors"),
        help="Surrogate checkpoint for FUSER_DF.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory. Defaults to outputs/demo/<model_name>/<sequence_name>.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional custom path for the merged aligned point cloud (.ply).",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.03,
        help="Voxel size used before sparse quantization.",
    )
    parser.add_argument(
        "--n_max_points",
        type=int,
        default=30000,
        help="Maximum number of sampled points per scan after quantization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--keep_global_frame",
        action="store_true",
        help="Do not normalize the predicted poses to the first scan before exporting.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_examples:
        print_available_examples()
        return

    sequence_name, sequence_dir = resolve_sequence_dir(args.data_path, args.example_name)
    point_cloud_paths = collect_point_cloud_paths(sequence_dir)

    if args.model_name == "FUSER" and not args.fuser_checkpoint:
        raise ValueError("FUSER requires --fuser_checkpoint (alias: --checkpoint).")
    if args.model_name == "FUSER_DF" and (not args.prior_checkpoint or not args.surrogate_checkpoint):
        raise ValueError("FUSER_DF requires both --prior_checkpoint and --surrogate_checkpoint.")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (REPO_ROOT / "outputs" / "demo" / args.model_name / slugify(sequence_name)).resolve()
    )

    print(f"Sequence      : {sequence_name}")
    print(f"Sequence dir  : {sequence_dir}")
    print(f"Scans         : {len(point_cloud_paths)}")
    print(f"Model         : {args.model_name}")
    print(f"Device        : {args.device}")
    print(f"Voxel size    : {args.voxel_size}")
    print(f"Max pts/scan  : {args.n_max_points}")

    device = resolve_device(args.device)
    cfg = make_demo_cfg(args)
    model = build_model(cfg).to(device)
    checkpoint_info = load_model_weights(model, cfg)
    model.eval()

    if args.model_name == "FUSER":
        print(f"Checkpoint    : {checkpoint_info['checkpoint_path']}")
    else:
        print(f"Prior ckpt    : {checkpoint_info['prior_path']}")
        print(f"Surrogate ckpt: {checkpoint_info['surrogate_path']}")

    batch, point_counts = build_demo_batch(point_cloud_paths, args.voxel_size, args.n_max_points)
    print(f"Loaded point counts after sampling: {point_counts}")

    start_time = time.time()
    output, peak_mem_gb = run_inference(model, batch, device)
    runtime_sec = time.time() - start_time

    raw_camera_poses = output["camera_poses"][0].detach().cpu().numpy().astype(np.float32)
    camera_poses = raw_camera_poses.copy()
    if not args.keep_global_frame:
        camera_poses = normalize_poses_to_first(camera_poses)

    prior_camera_poses = None
    if "prior_global_poses" in output:
        prior_camera_poses = output["prior_global_poses"][0].detach().cpu().numpy().astype(np.float32)
        if not args.keep_global_frame:
            prior_camera_poses = normalize_poses_to_first(prior_camera_poses)

    processed_point_clouds = batch["pcd_seqs"][0]
    aligned_scan_points = []
    for points, pose in zip(processed_point_clouds, camera_poses):
        points_np = points.detach().cpu().numpy().astype(np.float32)
        aligned_scan_points.append(apply_pose_to_points(points_np, pose))

    aligned_ply_path, pose_dir, npz_path, summary_path = save_demo_outputs(
        output_dir=output_dir,
        sequence_name=sequence_name,
        point_cloud_paths=point_cloud_paths,
        aligned_scan_points=aligned_scan_points,
        camera_poses=camera_poses,
        raw_camera_poses=raw_camera_poses,
        prior_camera_poses=prior_camera_poses,
        runtime_sec=runtime_sec,
        peak_mem_gb=peak_mem_gb,
        save_path=args.save_path,
        normalize_to_first=not args.keep_global_frame,
        model_name=args.model_name,
        checkpoint_info=checkpoint_info,
    )

    print(f"Inference time: {runtime_sec:.2f}s")
    print(f"Peak GPU mem  : {peak_mem_gb:.2f} GB")
    print(f"Aligned PLY   : {aligned_ply_path}")
    print(f"Pose dir      : {pose_dir}")
    print(f"Result bundle : {npz_path}")
    print(f"Summary       : {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()


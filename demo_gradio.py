import gc
import json
import re
import sys
import time
import zipfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import gradio as gr
import matplotlib
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial.transform import Rotation

if __package__:
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
DEMO_RUNS_ROOT = REPO_ROOT / "demo_runs"
DEFAULT_FUSER_CHECKPOINT = REPO_ROOT / "ckpts" / "fuser.safetensors"
DEFAULT_SURROGATE_CHECKPOINT = REPO_ROOT / "ckpts" / "fuser_df.safetensors"

SUPPORTED_POINT_CLOUD_SUFFIXES = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
MODEL_CACHE = {}
VISUALIZATION_ROTATION_PRESETS = [
    "Original",
    "Flip X (180°)",
    "Flip Y (180°)",
    "Flip Z (180°)",
    "Rotate X +90°",
    "Rotate Y +90°",
    "Rotate Z +90°",
]


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name.lower())
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    return key


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    text = text.strip("._-")
    return text or "sequence"


def discover_example_sequences():
    if not EXAMPLES_ROOT.is_dir():
        return {}

    discovered = {}
    for directory in sorted(EXAMPLES_ROOT.rglob("*")):
        if not directory.is_dir():
            continue
        files = sorted(
            [
                path for path in directory.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
            ],
            key=natural_key,
        )
        if len(files) < 2:
            continue

        if directory.name == "PointCloud" and directory.parent != EXAMPLES_ROOT:
            label = str(directory.parent.relative_to(EXAMPLES_ROOT))
            source_dir = directory
        else:
            label = str(directory.relative_to(EXAMPLES_ROOT))
            source_dir = directory

        discovered[label] = source_dir

    return dict(sorted(discovered.items(), key=lambda item: item[0].lower()))


def preview_rows(file_paths):
    rows = []
    for idx, path in enumerate(file_paths):
        rows.append([idx, Path(path).name])
    return rows


def normalize_uploaded_files(uploaded_files):
    if not uploaded_files:
        return []

    normalized = []
    for item in uploaded_files:
        if isinstance(item, dict) and "name" in item:
            path = item["name"]
        else:
            path = getattr(item, "name", item)
        path = Path(path).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES:
            normalized.append(path)
    return sorted(normalized, key=natural_key)


def select_example_sequence(example_name):
    if not example_name:
        return None, [], "Choose an example sequence from `examples/` or upload your own point cloud files."

    example_map = discover_example_sequences()
    sequence_dir = example_map.get(example_name)
    if sequence_dir is None:
        return None, [], f"Example `{example_name}` was not found under {EXAMPLES_ROOT}."

    file_paths = sorted(
        [
            path for path in sequence_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_POINT_CLOUD_SUFFIXES
        ],
        key=natural_key,
    )
    if len(file_paths) < 2:
        return None, [], f"Example `{example_name}` does not contain enough point clouds."

    source_state = {
        "mode": "example",
        "label": example_name,
        "files": [str(path) for path in file_paths],
    }
    log_message = f"Loaded example sequence `{example_name}` with {len(file_paths)} scans. Click `Register Sequence` to run FUSER."
    return source_state, preview_rows(file_paths), log_message


def select_uploaded_sequence(uploaded_files):
    file_paths = normalize_uploaded_files(uploaded_files)
    if not file_paths:
        return None, [], "Upload two or more point cloud files to create a custom sequence."

    source_state = {
        "mode": "upload",
        "label": "uploaded_sequence",
        "files": [str(path) for path in file_paths],
    }
    log_message = f"Loaded uploaded sequence with {len(file_paths)} scans. Click `Register Sequence` to run FUSER."
    return source_state, preview_rows(file_paths), log_message


def refresh_example_choices(current_value):
    example_map = discover_example_sequences()
    choices = list(example_map.keys())
    value = current_value if current_value in choices else None
    return gr.Dropdown(choices=choices, value=value, interactive=True)


def make_demo_cfg(model_name, fuser_checkpoint, prior_checkpoint, surrogate_checkpoint):
    cfg = deepcopy(base_cfg)
    cfg.model.model_name = model_name
    cfg.model.fuser_checkpoint = str(Path(fuser_checkpoint).expanduser().resolve()) if fuser_checkpoint else None
    cfg.model.prior_checkpoint = str(Path(prior_checkpoint).expanduser().resolve()) if prior_checkpoint else None
    cfg.model.surrogate_checkpoint = str(Path(surrogate_checkpoint).expanduser().resolve()) if surrogate_checkpoint else None
    return cfg


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_bundle(model_name, fuser_checkpoint, prior_checkpoint, surrogate_checkpoint):
    cfg = make_demo_cfg(model_name, fuser_checkpoint, prior_checkpoint, surrogate_checkpoint)
    cache_key = (
        model_name,
        cfg.model.fuser_checkpoint,
        cfg.model.prior_checkpoint,
        cfg.model.surrogate_checkpoint,
    )
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    for cached_bundle in MODEL_CACHE.values():
        cached_bundle["model"] = cached_bundle["model"].cpu()
    MODEL_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = get_device()
    model = build_model(cfg).to(device)
    checkpoint_info = load_model_weights(model, cfg)
    model.eval()

    bundle = {
        "model": model,
        "cfg": cfg,
        "checkpoint_info": checkpoint_info,
        "device": device,
    }
    MODEL_CACHE[cache_key] = bundle
    return bundle


def load_point_cloud_for_model(path, voxel_size, n_max_points, sample_seed):
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


def build_demo_batch(point_cloud_paths, voxel_size, n_max_points):
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


def normalize_poses_to_first(camera_poses):
    if camera_poses.shape[0] == 0:
        return camera_poses
    first_inv = np.linalg.inv(camera_poses[0])
    return np.matmul(first_inv[None], camera_poses)


def apply_pose_to_points(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


def create_prediction_payload(
    source_state,
    point_cloud_paths,
    processed_point_clouds,
    point_counts,
    output,
    model_name,
    checkpoint_info,
    voxel_size,
    n_max_points,
    runtime_sec,
    peak_mem_gb,
):
    camera_poses = output["camera_poses"][0].detach().cpu().numpy().astype(np.float32)
    camera_poses = normalize_poses_to_first(camera_poses)

    prior_camera_poses = None
    if "prior_global_poses" in output:
        prior_camera_poses = output["prior_global_poses"][0].detach().cpu().numpy().astype(np.float32)
        prior_camera_poses = normalize_poses_to_first(prior_camera_poses)

    aligned_scan_points = []
    point_offsets = [0]
    for points, pose in zip(processed_point_clouds, camera_poses):
        points_np = points.detach().cpu().numpy().astype(np.float32)
        aligned = apply_pose_to_points(points_np, pose)
        aligned_scan_points.append(aligned)
        point_offsets.append(point_offsets[-1] + aligned.shape[0])

    aligned_points = (
        np.concatenate(aligned_scan_points, axis=0)
        if aligned_scan_points
        else np.zeros((0, 3), dtype=np.float32)
    )

    scan_names = [path.name for path in point_cloud_paths]

    if model_name == "FUSER":
        checkpoint_label = str(checkpoint_info["checkpoint_path"])
    else:
        checkpoint_label = json.dumps(
            {
                "prior_path": str(checkpoint_info["prior_path"]),
                "surrogate_path": str(checkpoint_info["surrogate_path"]),
            }
        )

    return {
        "model_name": model_name,
        "source_label": source_state["label"],
        "scan_names": scan_names,
        "camera_poses": camera_poses,
        "prior_camera_poses": prior_camera_poses,
        "aligned_points": aligned_points,
        "point_offsets": np.asarray(point_offsets, dtype=np.int64),
        "point_counts": np.asarray(point_counts, dtype=np.int64),
        "voxel_size": float(voxel_size),
        "n_max_points": int(n_max_points),
        "runtime_sec": float(runtime_sec),
        "peak_mem_gb": float(peak_mem_gb),
        "checkpoint_label": checkpoint_label,
    }


def create_run_dir(source_label):
    DEMO_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = DEMO_RUNS_ROOT / f"{timestamp}_{slugify(source_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_prediction_payload(run_dir, payload):
    save_dict = {
        "model_name": np.asarray(payload["model_name"]),
        "source_label": np.asarray(payload["source_label"]),
        "scan_names": np.asarray(payload["scan_names"], dtype=str),
        "camera_poses": payload["camera_poses"],
        "aligned_points": payload["aligned_points"],
        "point_offsets": payload["point_offsets"],
        "point_counts": payload["point_counts"],
        "voxel_size": np.asarray(payload["voxel_size"], dtype=np.float32),
        "n_max_points": np.asarray(payload["n_max_points"], dtype=np.int32),
        "runtime_sec": np.asarray(payload["runtime_sec"], dtype=np.float32),
        "peak_mem_gb": np.asarray(payload["peak_mem_gb"], dtype=np.float32),
        "checkpoint_label": np.asarray(payload["checkpoint_label"]),
    }
    if payload["prior_camera_poses"] is not None:
        save_dict["prior_camera_poses"] = payload["prior_camera_poses"]
    else:
        save_dict["prior_camera_poses"] = np.zeros((0, 4, 4), dtype=np.float32)

    predictions_path = run_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **save_dict)
    return predictions_path


def load_prediction_payload(run_dir):
    predictions_path = Path(run_dir) / "predictions.npz"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"Missing saved predictions: {predictions_path}")

    loaded = np.load(predictions_path, allow_pickle=False)
    prior = loaded["prior_camera_poses"]
    if prior.size == 0:
        prior = None

    return {
        "model_name": loaded["model_name"].item(),
        "source_label": loaded["source_label"].item(),
        "scan_names": loaded["scan_names"].astype(str).tolist(),
        "camera_poses": loaded["camera_poses"].astype(np.float32),
        "prior_camera_poses": prior.astype(np.float32) if prior is not None else None,
        "aligned_points": loaded["aligned_points"].astype(np.float32),
        "point_offsets": loaded["point_offsets"].astype(np.int64),
        "point_counts": loaded["point_counts"].astype(np.int64),
        "voxel_size": float(loaded["voxel_size"].item()),
        "n_max_points": int(loaded["n_max_points"].item()),
        "runtime_sec": float(loaded["runtime_sec"].item()),
        "peak_mem_gb": float(loaded["peak_mem_gb"].item()),
        "checkpoint_label": loaded["checkpoint_label"].item(),
    }


def get_scan_points(predictions, scan_idx):
    start = int(predictions["point_offsets"][scan_idx])
    end = int(predictions["point_offsets"][scan_idx + 1])
    return predictions["aligned_points"][start:end]


def subsample_points(points, max_points):
    if max_points <= 0 or len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[indices]


def get_selected_scan_indices(predictions, scan_filter):
    if not scan_filter or scan_filter == "All":
        return list(range(len(predictions["scan_names"])))

    try:
        scan_idx = int(scan_filter.split(":", 1)[0])
    except (TypeError, ValueError, IndexError):
        return list(range(len(predictions["scan_names"])))
    return [scan_idx]


def estimate_scene_scale(points):
    if points.size == 0:
        return 1.0
    lower = np.percentile(points, 5, axis=0)
    upper = np.percentile(points, 95, axis=0)
    scale = float(np.linalg.norm(upper - lower))
    return scale if scale > 1e-6 else 1.0


def get_visualization_rotation_transform(viz_rotation_preset):
    transform = np.eye(4, dtype=np.float32)

    if viz_rotation_preset == "Flip X (180°)":
        transform[:3, :3] = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    elif viz_rotation_preset == "Flip Y (180°)":
        transform[:3, :3] = np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    elif viz_rotation_preset == "Flip Z (180°)":
        transform[:3, :3] = np.diag([-1.0, -1.0, 1.0]).astype(np.float32)
    elif viz_rotation_preset == "Rotate X +90°":
        transform[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    elif viz_rotation_preset == "Rotate Y +90°":
        transform[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix().astype(np.float32)
    elif viz_rotation_preset == "Rotate Z +90°":
        transform[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix().astype(np.float32)

    return transform


def apply_default_view_transform(scene, merged_points, viz_rotation_preset="Original"):
    if merged_points.size == 0:
        return

    center = merged_points.mean(axis=0)
    scene_scale = estimate_scene_scale(merged_points)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = -center

    # Normalize the scene to a compact size so orbit controls feel stable.
    normalized_scale = 2.0 / max(scene_scale, 1e-6)
    scale_matrix = np.eye(4, dtype=np.float32)
    scale_matrix[:3, :3] *= normalized_scale
    transform = scale_matrix @ transform
    transform = get_visualization_rotation_transform(viz_rotation_preset) @ transform

    scene.apply_transform(transform)


def get_opengl_conversion_matrix():
    matrix = np.eye(4, dtype=np.float32)
    matrix[1, 1] = -1
    matrix[2, 2] = -1
    return matrix


def transform_points(transform, points):
    points = np.asarray(points)
    transformed = points @ transform[:3, :3].T + transform[:3, 3]
    return transformed


def compute_camera_faces(cone_shape):
    faces = []
    num_vertices = len(cone_shape.vertices)
    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        f1 = face + num_vertices
        f2 = face + 2 * num_vertices
        faces.extend(
            [
                (v1, v2, f1[1]),
                (v1, f1[0], v3),
                (f1[2], v2, v3),
                (v1, v2, f2[1]),
                (v1, f2[0], v3),
                (f2[2], v2, v3),
            ]
        )
    faces += [(c, b, a) for a, b, c in faces]
    return np.asarray(faces, dtype=np.int64)


def integrate_camera_into_scene(scene, transform, face_color, scene_scale):
    cam_width = max(scene_scale * 0.04, 0.03)
    cam_height = max(scene_scale * 0.08, 0.06)

    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix().astype(np.float32)
    rot[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    complete_transform = transform @ opengl_transform @ rot
    cone = trimesh.creation.cone(cam_width, cam_height, sections=4)

    slight_rotation = np.eye(4, dtype=np.float32)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix().astype(np.float32)

    vertices = np.concatenate(
        [
            cone.vertices,
            0.95 * cone.vertices,
            transform_points(slight_rotation, cone.vertices),
        ],
        axis=0,
    )
    vertices = transform_points(complete_transform, vertices)
    faces = compute_camera_faces(cone)

    camera_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    camera_mesh.visual.face_colors[:, :3] = face_color
    camera_mesh.visual.face_colors[:, 3] = 255
    scene.add_geometry(camera_mesh)


def predictions_to_scene(
    predictions,
    scan_filter="All",
    show_cam=True,
    max_points_per_scan=30000,
    viz_rotation_preset="Original",
):
    scene = trimesh.Scene()
    selected_indices = get_selected_scan_indices(predictions, scan_filter)

    collected_points = []
    colormap = matplotlib.colormaps.get_cmap("turbo")

    for index in selected_indices:
        scan_points = get_scan_points(predictions, index)
        scan_points = subsample_points(scan_points, int(max_points_per_scan))
        if scan_points.size == 0:
            continue

        collected_points.append(scan_points)
        rgba = colormap(index / max(len(predictions["scan_names"]) - 1, 1))
        color = np.asarray([int(255 * channel) for channel in rgba[:3]], dtype=np.uint8)
        colors = np.repeat(color[None], scan_points.shape[0], axis=0)
        point_cloud = trimesh.PointCloud(vertices=scan_points, colors=colors)
        scene.add_geometry(point_cloud)

    merged_points = (
        np.concatenate(collected_points, axis=0)
        if collected_points
        else np.zeros((0, 3), dtype=np.float32)
    )
    scene_scale = estimate_scene_scale(merged_points)

    if show_cam:
        for index in selected_indices:
            rgba = colormap(index / max(len(predictions["scan_names"]) - 1, 1))
            face_color = tuple(int(255 * channel) for channel in rgba[:3])
            integrate_camera_into_scene(scene, predictions["camera_poses"][index], face_color, scene_scale)

    apply_default_view_transform(scene, merged_points, viz_rotation_preset=viz_rotation_preset)

    return scene


def build_glb_path(run_dir, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset):
    safe_filter = str(scan_filter or "All").replace(":", "_").replace(" ", "_").replace(".", "_")
    safe_rotation = slugify(viz_rotation_preset or "Original")
    return Path(run_dir) / f"scene_{safe_filter}_cam{int(bool(show_cam))}_pts{int(max_points_per_scan)}_{safe_rotation}.glb"


def export_pose_archive(run_dir, source_label, camera_poses):
    pose_dir = export_pred_poses(
        output_dir=str(run_dir),
        scene_id=slugify(source_label),
        pred_pose_seq=camera_poses,
    )
    zip_path = Path(run_dir) / "pred_poses.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in sorted(pose_dir.rglob("*")):
            if file_path.is_file():
                zip_file.write(file_path, arcname=str(file_path.relative_to(run_dir)))
    return str(zip_path)


def build_scan_filter_dropdown(predictions, current_value="All"):
    choices = ["All"] + [f"{idx}: {name}" for idx, name in enumerate(predictions["scan_names"])]
    value = current_value if current_value in choices else "All"
    return gr.Dropdown(choices=choices, value=value, interactive=True)


def build_summary_markdown(predictions):
    checkpoint_label = predictions["checkpoint_label"]
    if predictions["model_name"] == "FUSER_DF":
        checkpoint_obj = json.loads(checkpoint_label)
        checkpoint_text = (
            f"- Prior checkpoint: `{checkpoint_obj['prior_path']}`\n"
            f"- Surrogate checkpoint: `{checkpoint_obj['surrogate_path']}`"
        )
    else:
        checkpoint_text = f"- Checkpoint: `{checkpoint_label}`"

    total_points = int(np.sum(predictions["point_counts"]))
    return (
        f"### Run Summary\n"
        f"- Model: `{predictions['model_name']}`\n"
        f"- Sequence: `{predictions['source_label']}`\n"
        f"- Scans: `{len(predictions['scan_names'])}`\n"
        f"- Sampled points: `{total_points}` total\n"
        f"- Voxel size: `{predictions['voxel_size']:.4f}`\n"
        f"- Max points per scan: `{predictions['n_max_points']}`\n"
        f"- Inference time: `{predictions['runtime_sec']:.2f}s`\n"
        f"- Peak GPU memory: `{predictions['peak_mem_gb']:.2f} GB`\n"
        f"{checkpoint_text}\n\n"
        f"All poses shown in the viewer are normalized to the first scan for stable visualization."
    )


def reconstruct_sequence(
    source_state,
    model_name,
    voxel_size,
    n_max_points,
    fuser_checkpoint,
    prior_checkpoint,
    surrogate_checkpoint,
    show_cam,
    max_points_per_scan,
    viz_rotation_preset,
):
    if not source_state or not source_state.get("files"):
        return (
            None,
            "Select an example sequence or upload a point cloud sequence first.",
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            None,
            "### Run Summary\nNo reconstruction has been run yet.",
            None,
        )

    point_cloud_paths = [Path(path) for path in source_state["files"]]
    if len(point_cloud_paths) < 2:
        return (
            None,
            "At least two point cloud files are required for multiview registration.",
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            None,
            "### Run Summary\nNo reconstruction has been run yet.",
            None,
        )

    if model_name == "FUSER" and not fuser_checkpoint:
        return (
            None,
            "FUSER requires a valid `FUSER Checkpoint` path.",
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            None,
            "### Run Summary\nNo reconstruction has been run yet.",
            None,
        )

    if model_name == "FUSER_DF" and (not prior_checkpoint or not surrogate_checkpoint):
        return (
            None,
            "FUSER-DF requires both `Prior Checkpoint` and `Surrogate Checkpoint`.",
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            None,
            "### Run Summary\nNo reconstruction has been run yet.",
            None,
        )

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        bundle = get_model_bundle(model_name, fuser_checkpoint, prior_checkpoint, surrogate_checkpoint)
        batch, point_counts = build_demo_batch(point_cloud_paths, float(voxel_size), int(n_max_points))
        processed_point_clouds = batch["pcd_seqs"][0]

        start_time = time.time()
        output, peak_mem_gb = run_inference(bundle["model"], batch, bundle["device"])
        runtime_sec = time.time() - start_time

        payload = create_prediction_payload(
            source_state=source_state,
            point_cloud_paths=point_cloud_paths,
            processed_point_clouds=processed_point_clouds,
            point_counts=point_counts,
            output=output,
            model_name=model_name,
            checkpoint_info=bundle["checkpoint_info"],
            voxel_size=float(voxel_size),
            n_max_points=int(n_max_points),
            runtime_sec=runtime_sec,
            peak_mem_gb=peak_mem_gb,
        )

        run_dir = create_run_dir(source_state["label"])
        save_prediction_payload(run_dir, payload)
        pose_zip_path = export_pose_archive(run_dir, source_state["label"], payload["camera_poses"])

        glb_path = build_glb_path(run_dir, "All", show_cam, max_points_per_scan, viz_rotation_preset)
        scene = predictions_to_scene(
            payload,
            scan_filter="All",
            show_cam=show_cam,
            max_points_per_scan=max_points_per_scan,
            viz_rotation_preset=viz_rotation_preset,
        )
        scene.export(file_obj=str(glb_path))

        log_message = (
            f"Registration finished with `{model_name}` on {len(point_cloud_paths)} scans "
            f"in {runtime_sec:.2f}s. Adjust the viewer settings on the right if needed."
        )
        summary = build_summary_markdown(payload)
        dropdown = build_scan_filter_dropdown(payload)
        return str(glb_path), log_message, dropdown, pose_zip_path, summary, str(run_dir)
    except Exception as exc:
        return (
            None,
            f"Registration failed: {exc}",
            gr.Dropdown(choices=["All"], value="All", interactive=True),
            None,
            f"### Run Summary\n`{exc}`",
            None,
        )


def update_visualization(run_dir, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset):
    if not run_dir:
        return None, "No reconstruction available yet. Click `Register Sequence` first."

    try:
        predictions = load_prediction_payload(run_dir)
        glb_path = build_glb_path(run_dir, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset)
        if not glb_path.is_file():
            scene = predictions_to_scene(
                predictions,
                scan_filter=scan_filter,
                show_cam=show_cam,
                max_points_per_scan=max_points_per_scan,
                viz_rotation_preset=viz_rotation_preset,
            )
            scene.export(file_obj=str(glb_path))
        return str(glb_path), "Visualization updated."
    except Exception as exc:
        return None, f"Visualization update failed: {exc}"


def reset_demo():
    return (
        None,
        "Choose an example sequence from `examples/` or upload your own point cloud files.",
        [],
        None,
        None,
        gr.Dropdown(choices=["All"], value="All", interactive=True),
        None,
        "### Run Summary\nNo reconstruction has been run yet.",
        gr.Dropdown(choices=list(discover_example_sequences().keys()), value=None, interactive=True),
        gr.Dropdown(choices=VISUALIZATION_ROTATION_PRESETS, value="Original", interactive=True),
        None,
    )


def build_intro_html():
    return """
    <div class="hero">
      <div class="hero-kicker">Official Interactive Demo</div>
      <h1>FUSER / FUSER-DF</h1>
      <p class="hero-subtitle">Feed-forward multiview point cloud registration with optional SE(3)<sup>N</sup> diffusion refinement.</p>
      <p class="hero-text">
        Select a point cloud sequence from <code>examples/</code> or upload your own scans.
        The demo constructs the same sparse input representation used by benchmark evaluation,
        predicts globally consistent poses, aligns the sequence, and exports a 3D viewer plus pose files.
      </p>
    </div>
    """


if __name__ == "__main__":
    initial_examples = list(discover_example_sequences().keys())

    theme = gr.themes.Base(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
    )

    with gr.Blocks(
        theme=theme,
        css="""
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap');

        .gradio-container {
            background:
                radial-gradient(circle at top left, rgba(58, 123, 213, 0.18), transparent 35%),
                radial-gradient(circle at top right, rgba(34, 211, 238, 0.14), transparent 30%),
                linear-gradient(180deg, #f6fbff 0%, #edf6fb 100%);
            font-family: 'Space Grotesk', sans-serif;
        }

        .hero {
            padding: 22px 26px;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(232,244,250,0.92));
            border: 1px solid rgba(13, 42, 66, 0.10);
            box-shadow: 0 18px 40px rgba(18, 52, 74, 0.10);
            margin-bottom: 14px;
        }

        .hero-kicker {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            background: #0f3d56;
            color: white;
            font-size: 12px;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero h1 {
            margin: 0;
            color: #08273b;
            font-size: 2.55rem;
            line-height: 1.05;
            font-weight: 700;
        }

        .hero-subtitle {
            margin: 10px 0 6px 0;
            color: #0e5f80;
            font-size: 1.05rem;
            font-weight: 500;
        }

        .hero-text {
            margin: 0;
            color: #28465a;
            line-height: 1.65;
            font-size: 0.98rem;
        }

        .mono, code, textarea, input {
            font-family: 'JetBrains Mono', monospace !important;
        }

        .status-box {
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.88);
            border: 1px solid rgba(20, 72, 96, 0.10);
            box-shadow: 0 10px 28px rgba(21, 68, 93, 0.08);
        }
        """,
    ) as demo:
        source_state = gr.State(value=None)
        run_dir_state = gr.State(value=None)

        gr.HTML(build_intro_html())

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Group(elem_classes=["status-box"]):
                    gr.Markdown("### Input Sequence")
                    with gr.Row():
                        example_dropdown = gr.Dropdown(
                            choices=initial_examples,
                            value=None,
                            label="Example Sequence",
                            info="Any directory under `examples/` containing at least two point cloud files will appear here.",
                            interactive=True,
                        )
                        refresh_examples_btn = gr.Button("Refresh", min_width=96)

                    upload_point_clouds = gr.File(
                        file_count="multiple",
                        file_types=sorted(SUPPORTED_POINT_CLOUD_SUFFIXES),
                        label="Upload Point Cloud Sequence",
                        interactive=True,
                    )

                    preview_table = gr.Dataframe(
                        headers=["Index", "File"],
                        datatype=["number", "str"],
                        value=[],
                        interactive=False,
                        label="Sequence Preview",
                        wrap=True,
                    )

                with gr.Group(elem_classes=["status-box"]):
                    gr.Markdown("### Inference Settings")
                    model_name = gr.Radio(
                        choices=list(SUPPORTED_MODEL_NAMES),
                        value="FUSER",
                        label="Model",
                        info="Choose direct feed-forward registration or diffusion-refined registration.",
                    )
                    voxel_size = gr.Slider(
                        minimum=0.01,
                        maximum=0.10,
                        value=0.03,
                        step=0.005,
                        label="Voxel Size",
                    )
                    n_max_points = gr.Slider(
                        minimum=2000,
                        maximum=30000,
                        value=30000,
                        step=1000,
                        label="Max Points Per Scan",
                    )

                    with gr.Accordion("Checkpoint Overrides", open=False):
                        fuser_checkpoint = gr.Textbox(
                            value=str(DEFAULT_FUSER_CHECKPOINT) if DEFAULT_FUSER_CHECKPOINT.is_file() else "",
                            label="FUSER Checkpoint",
                            elem_classes=["mono"],
                        )
                        prior_checkpoint = gr.Textbox(
                            value=str(DEFAULT_FUSER_CHECKPOINT) if DEFAULT_FUSER_CHECKPOINT.is_file() else "",
                            label="FUSER-DF Prior Checkpoint",
                            elem_classes=["mono"],
                        )
                        surrogate_checkpoint = gr.Textbox(
                            value=str(DEFAULT_SURROGATE_CHECKPOINT) if DEFAULT_SURROGATE_CHECKPOINT.is_file() else "",
                            label="FUSER-DF Surrogate Checkpoint",
                            elem_classes=["mono"],
                        )

            with gr.Column(scale=2):
                log_output = gr.Markdown(
                    "Choose an example sequence from `examples/` or upload your own point cloud files.",
                    elem_classes=["status-box"],
                )
                reconstruction_output = gr.Model3D(
                    label="Aligned Sequence Viewer",
                    height=560,
                    zoom_speed=0.8,
                    pan_speed=0.8,
                )

                with gr.Row():
                    submit_btn = gr.Button("Register Sequence", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

                with gr.Group(elem_classes=["status-box"]):
                    gr.Markdown("### Viewer Controls")
                    with gr.Row():
                        show_cam = gr.Checkbox(label="Show Cameras", value=True)
                        max_points_per_scan = gr.Slider(
                            minimum=500,
                            maximum=30000,
                            value=30000,
                            step=500,
                            label="Display Points Per Scan",
                        )
                    viz_rotation_preset = gr.Dropdown(
                        choices=VISUALIZATION_ROTATION_PRESETS,
                        value="Original",
                        label="Display Rotation",
                        info="Visualization-only global rotation. This does not change predicted poses.",
                        interactive=True,
                    )
                    scan_filter = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Visible Scan",
                        interactive=True,
                    )

                with gr.Row():
                    pose_archive = gr.File(label="Predicted Pose Archive (.zip)")
                summary_output = gr.Markdown(
                    "### Run Summary\nNo reconstruction has been run yet.",
                    elem_classes=["status-box"],
                )

        refresh_examples_btn.click(
            fn=refresh_example_choices,
            inputs=[example_dropdown],
            outputs=[example_dropdown],
        )

        example_dropdown.change(
            fn=select_example_sequence,
            inputs=[example_dropdown],
            outputs=[source_state, preview_table, log_output],
        )

        upload_point_clouds.change(
            fn=select_uploaded_sequence,
            inputs=[upload_point_clouds],
            outputs=[source_state, preview_table, log_output],
        )

        submit_btn.click(
            fn=lambda: "Running registration...",
            inputs=[],
            outputs=[log_output],
        ).then(
            fn=reconstruct_sequence,
            inputs=[
                source_state,
                model_name,
                voxel_size,
                n_max_points,
                fuser_checkpoint,
                prior_checkpoint,
                surrogate_checkpoint,
                show_cam,
                max_points_per_scan,
                viz_rotation_preset,
            ],
            outputs=[
                reconstruction_output,
                log_output,
                scan_filter,
                pose_archive,
                summary_output,
                run_dir_state,
            ],
        )

        scan_filter.change(
            fn=update_visualization,
            inputs=[run_dir_state, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset],
            outputs=[reconstruction_output, log_output],
        )

        show_cam.change(
            fn=update_visualization,
            inputs=[run_dir_state, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset],
            outputs=[reconstruction_output, log_output],
        )

        max_points_per_scan.change(
            fn=update_visualization,
            inputs=[run_dir_state, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset],
            outputs=[reconstruction_output, log_output],
        )

        viz_rotation_preset.change(
            fn=update_visualization,
            inputs=[run_dir_state, scan_filter, show_cam, max_points_per_scan, viz_rotation_preset],
            outputs=[reconstruction_output, log_output],
        )

        clear_btn.click(
            fn=reset_demo,
            inputs=[],
            outputs=[
                reconstruction_output,
                log_output,
                preview_table,
                source_state,
                run_dir_state,
                scan_filter,
                pose_archive,
                summary_output,
                example_dropdown,
                viz_rotation_preset,
                upload_point_clouds,
            ],
        )

    demo.queue(max_size=20).launch(show_error=True)

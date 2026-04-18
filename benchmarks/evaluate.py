from pathlib import Path
import numpy as np
import torch

ERROR_THRES = {
    "ScanNet": {
        "re_thresh": 15.0,
        "te_thresh": 0.3,
        "re_thresh_ecdf": [3, 5, 10, 30, 45],
        "te_thresh_ecdf": [0.05, 0.1, 0.25, 0.5, 0.75],
    },
    "3DMatchFrame": {
        "re_thresh": 15.0,
        "te_thresh": 0.3,
        "re_thresh_ecdf": [3, 5, 10, 30, 45],
        "te_thresh_ecdf": [0.05, 0.1, 0.25, 0.5, 0.75],
    },
    "ArkitScenes": {
        "re_thresh": 15.0,
        "te_thresh": 0.3,
        "re_thresh_ecdf": [3, 5, 10, 30, 45],
        "te_thresh_ecdf": [0.05, 0.1, 0.25, 0.5, 0.75],
    },
}

def read_trajectory(filename, dim=4):
    with open(filename, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    keys = []
    trajectory_rows = []
    for line_idx, line in enumerate(lines):
        if line_idx % (dim + 1) == 0:
            keys.append(line.strip().split()[:3])
        else:
            trajectory_rows.append(np.fromstring(line.replace("\t", " "), sep=" ")[:dim])

    traj = np.asarray(trajectory_rows, dtype=np.float64).reshape(-1, dim, dim)
    return np.asarray(keys), traj

def rotation_error(R1, R2):
    relative = np.matmul(np.swapaxes(R1, -1, -2), R2)
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

def compute_pose_errors(gt_traj, est_traj):
    gt_traj = np.asarray(gt_traj, dtype=np.float64)
    est_traj = np.asarray(est_traj, dtype=np.float64)
    re = rotation_error(gt_traj[:, :3, :3], est_traj[:, :3, :3])
    te = np.sqrt(np.sum(np.square(gt_traj[:, :3, 3] - est_traj[:, :3, 3]), axis=1) + 1e-8)
    return re, te

def evaluate_scene(
    benchmark_name,
    camera_poses,
    scene_id,
    data_root
):
    scene_root = Path(data_root) / scene_id / "PointCloud"
    gt_log_path = scene_root / "gt.log"
    if not gt_log_path.is_file():
        raise FileNotFoundError(f"Missing gt.log for scene {scene_id}: {gt_log_path}")

    gt_pairs, gt_traj = read_trajectory(gt_log_path)
    camera_poses = camera_poses.detach().cpu()

    est_pairs = []
    est_traj = []
    for pair in gt_pairs:
        ref_id = int(pair[0])
        src_id = int(pair[1])
        pred_pose = torch.linalg.inv(camera_poses[ref_id]) @ camera_poses[src_id]
        pred_pose = pred_pose.clone()
        est_pairs.append([pair[0], pair[1], pair[2]])
        est_traj.append(pred_pose.numpy())

    est_pairs = np.asarray(est_pairs)
    est_traj = np.asarray(est_traj)
    re, te = compute_pose_errors(gt_traj, est_traj)
    re_thresh = ERROR_THRES[benchmark_name]["re_thresh"]
    te_thresh = ERROR_THRES[benchmark_name]["te_thresh"]
    rr_mask = (re < re_thresh) & (te < te_thresh)

    return {
        "benchmark_name": benchmark_name,
        "pair_count": int(re.shape[0]),
        "re": re.tolist(),
        "te": te.tolist(),
        "mean_re": float(np.mean(re)),
        "median_re": float(np.median(re)),
        "mean_te": float(np.mean(te)),
        "median_te": float(np.median(te)),
        "rr_percent": float(np.mean(rr_mask) * 100.0),
    }

def aggregate_results(scene_results):

    if not scene_results:
        raise ValueError("No pairwise results to aggregate.")

    all_re = np.asarray([value for result in scene_results for value in result["re"]], dtype=np.float64)
    all_te = np.asarray([value for result in scene_results for value in result["te"]], dtype=np.float64)
    pair_counts = np.asarray([result["pair_count"] for result in scene_results], dtype=np.int64)
    times = [result["time_sec"] for result in scene_results if "time_sec" in result]
    peak_mems = [result["peak_mem"] for result in scene_results if "peak_mem" in result]

    benchmark_name = scene_results[0]["benchmark_name"]
    re_thresh_ecdf = ERROR_THRES[benchmark_name]["re_thresh_ecdf"]
    te_thresh_ecdf = ERROR_THRES[benchmark_name]["te_thresh_ecdf"]
    re_thresh = ERROR_THRES[benchmark_name]["re_thresh"]
    te_thresh = ERROR_THRES[benchmark_name]["te_thresh"]
    rr_mask = (all_re < re_thresh) & (all_te < te_thresh)

    return {
        "scene_count": len(scene_results),
        "pair_count": int(np.sum(pair_counts)),
        "mean_re": float(np.mean(all_re)),
        "median_re": float(np.median(all_re)),
        "mean_te": float(np.mean(all_te)),
        "median_te": float(np.median(all_te)),
        "rr_percent": float(np.mean(rr_mask) * 100.0),
        "re_ecdf_percent": [float(np.mean(all_re < threshold) * 100.0) for threshold in re_thresh_ecdf],
        "te_ecdf_percent": [float(np.mean(all_te < threshold) * 100.0) for threshold in te_thresh_ecdf],
        "avg_time_sec": float(np.mean(times)) if times else float("nan"),
        "mean_peak_mem_gb": float(np.mean(peak_mems)) if peak_mems else float("nan"),
    }

def build_evluation_summary_text(summary, benchmark_name, model_name, checkpoint_info):
    re_thresh_ecdf = ERROR_THRES[benchmark_name]["re_thresh_ecdf"]
    te_thresh_ecdf = ERROR_THRES[benchmark_name]["te_thresh_ecdf"]
    re_thresh = ERROR_THRES[benchmark_name]["re_thresh"]
    te_thresh = ERROR_THRES[benchmark_name]["te_thresh"]

    header_lines = []
    if model_name == "FUSER":
        header_lines.append(f"Checkpoint: {checkpoint_info['checkpoint_path']}")
    elif model_name == "FUSER_DF":
        header_lines.extend(
            [
                f"Prior checkpoint    : {checkpoint_info['prior_path']}",
                f"Surrogate checkpoint: {checkpoint_info['surrogate_path']}",
            ]
        )
    else:
        raise NotImplementedError

    lines = [
        "=" * 66,
        f"Evaluation Results for {benchmark_name}",
        *header_lines,
        f"Scenes evaluated: {summary['scene_count']}",
        f"Pairs evaluated : {summary['pair_count']}",
        "-" * 66,
        f"Mean RE / Median RE : {summary['mean_re']:.3f} / {summary['median_re']:.3f} deg",
        f"Mean TE / Median TE : {summary['mean_te']:.3f} / {summary['median_te']:.3f} m",
        f"RR @ ({re_thresh} deg, {te_thresh} m) : {summary['rr_percent']:.2f}%",
        f"RE ECDF                : {summary['re_ecdf_percent']} @ {re_thresh_ecdf} deg",
        f"TE ECDF                : {summary['te_ecdf_percent']} @ {te_thresh_ecdf} m",
        "-" * 66,
        f"Avg time                        : {summary['avg_time_sec']:.2f}s",
        f"Peak mem (mean)                 : {summary['mean_peak_mem_gb']:.2f} GB",
        "=" * 66,
    ]
    return "\n".join(lines)

def save_summary_report(output_dir, summary_text):
    output_path = Path(output_dir) / f"results.txt"
    output_path.write_text(summary_text + "\n", encoding="utf-8")
    return output_path
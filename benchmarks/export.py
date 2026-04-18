from pathlib import Path
import numpy as np
import torch

def _to_numpy_pose_array(pose_seq):
    if pose_seq is None:
        return None

    if isinstance(pose_seq, list):
        pose_seq = [pose.detach().cpu().numpy() if torch.is_tensor(pose) else np.asarray(pose) for pose in pose_seq]
        pose_seq = np.asarray(pose_seq, dtype=np.float32)
    elif torch.is_tensor(pose_seq):
        pose_seq = pose_seq.detach().cpu().numpy().astype(np.float32)
    else:
        pose_seq = np.asarray(pose_seq, dtype=np.float32)

    if pose_seq.ndim != 3 or pose_seq.shape[1:] != (4, 4):
        raise ValueError(f"Expected pose array of shape [S, 4, 4], got {pose_seq.shape}")
    return pose_seq.astype(np.float32, copy=True)

def export_pred_poses(
    output_dir,
    scene_id,
    pred_pose_seq
):
    output_root = Path(output_dir).expanduser().resolve()
    pred_pose_seq = _to_numpy_pose_array(pred_pose_seq)
    num_views = int(pred_pose_seq.shape[0])
    frame_ids = [str(i) for i in range(num_views)]

    pose_dir = Path(output_root / "pred_poses" / scene_id)
    pose_dir.mkdir(parents=True, exist_ok=True)
    for frame_id, pred_pose in zip(frame_ids, pred_pose_seq):
        np.savetxt(pose_dir / f"cloud_bin_{frame_id}.pose.txt", pred_pose, fmt="%.8f")
    return pose_dir

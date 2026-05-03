import torch
import MinkowskiEngine as ME
from models.FUSER import FUSER
from utils.diffusion_scheduler import DiffusionScheduler
from utils.se_math import se3
from utils.seq_manipulation import pad_sequence

class FUSER_DF(FUSER):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.diff_num_steps = cfg.diffusion.num_steps
        self.diff_add_noise = cfg.diffusion.add_noise
        self.diff_sigma_r = cfg.diffusion.sigma_r
        self.diff_sigma_t = cfg.diffusion.sigma_t

        self.surrogate = FUSER(cfg)
        self.diff_scheduler = DiffusionScheduler(n_diff_steps=self.diff_num_steps)

    @staticmethod
    def _build_transformed_sparse_inputs(pcd_seqs, transforms, voxel_size, device):
        transformed_coords = []
        transformed_feats = []
        for pcd_seq, scene_transforms in zip(pcd_seqs, transforms):
            padded_points, padding_mask, _ = pad_sequence(pcd_seq, require_padding_mask=True)
            valid_mask = (~padding_mask).to(device)
            padded_points = padded_points.transpose(0, 1).to(device)
            scene_transforms = scene_transforms.to(device)

            transformed = (
                padded_points @ scene_transforms[:, :3, :3].transpose(1, 2)
                + scene_transforms[:, :3, [3]].transpose(1, 2)
            )
            transformed_seq = [points[mask].detach().cpu() for points, mask in zip(transformed, valid_mask)]
            transformed_coords.extend([torch.floor(points / voxel_size) for points in transformed_seq])
            transformed_feats.extend([torch.ones_like(points[:, [0]]) for points in transformed_seq])
        return ME.utils.sparse_collate(transformed_coords, transformed_feats)

    def forward(self, batch):

        # Prior Global Poses
        voxel_size = batch["voxel_size"]
        prior_raw_global_poses = self.forward_backbone(batch)
        prior_global_poses = self._to_metric_global_poses(prior_raw_global_poses, voxel_size)

        # SE(3)^N Diffusion Refinement
        current = prior_global_poses.clone()
        current = torch.linalg.inv(current[:, [0]]) @ current
        initial = current.clone()
        refined = current.clone()
        device = current.device
        dtype = current.dtype

        for step in range(self.diff_num_steps, 1, -1):
            current = torch.linalg.inv(current[:, [0]]) @ current
            coord_mink, feat_mink = self._build_transformed_sparse_inputs(
                batch["pcd_seqs"],
                current,
                voxel_size,
                device,
            )
            batch["coord_seqs_mink"] = coord_mink.to(device)
            batch["feat_seqs_mink"] = feat_mink.to(device)

            delta = self.surrogate(batch)["camera_poses"].clone()
            refined = delta @ current
            refined = torch.linalg.inv(refined[:, [0]]) @ refined

            gamma0 = self.diff_scheduler.gamma0[step].to(device=device, dtype=dtype)
            gamma1 = self.diff_scheduler.gamma1[step].to(device=device, dtype=dtype)
            gamma2 = self.diff_scheduler.gamma2[step].to(device=device, dtype=dtype)
            current = se3.exp(
                gamma0 * se3.log(refined)
                + gamma1 * se3.log(current)
                + gamma2 * se3.log(initial)
            )

            if self.diff_add_noise:
                alpha_bar = self.diff_scheduler.alpha_bars[step].to(device=device, dtype=dtype)
                alpha_bar_prev = self.diff_scheduler.alpha_bars[step - 1].to(device=device, dtype=dtype)
                beta = self.diff_scheduler.betas[step].to(device=device, dtype=dtype)
                coeff = ((1 - alpha_bar_prev) / (1 - alpha_bar)) * beta
                scale = torch.tensor(
                    [self.diff_sigma_r] * 3 + [self.diff_sigma_t] * 3,
                    device=device,
                    dtype=dtype,
                )[None]
                noise = torch.sqrt(coeff) * scale * torch.randn(
                    current.shape[0],
                    6,
                    device=device,
                    dtype=dtype,
                )
                current = se3.exp(noise) @ current

        global_poses = refined if self.diff_num_steps > 1 else current
        output = {
            "camera_poses": global_poses,
            "prior_global_poses": prior_global_poses,
        }
        return output
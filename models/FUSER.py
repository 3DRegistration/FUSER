from models.fuser_core import FUSERBackbone

class FUSER(FUSERBackbone):

    def _to_metric_global_poses(self, raw_global_poses, voxel_size):
        global_poses = raw_global_poses.clone()
        global_poses[:, :, :3, 3] *= voxel_size
        return global_poses

    def forward(self, batch):
        raw_global_poses = self.forward_backbone(batch)
        global_poses = self._to_metric_global_poses(raw_global_poses, batch["voxel_size"])
        output = {
            "camera_poses": global_poses,
        }
        return output
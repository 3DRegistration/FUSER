from .dataset_3dmatchframe import ThreeDMatchFrame

class ArkitScenes(ThreeDMatchFrame):
    def __init__(
        self,
        data_root,
        voxel_size=0.03,
        n_max_points=15000,
        scene_ids=None,
        seed=12345):
        super().__init__(data_root, voxel_size, n_max_points, scene_ids, seed)
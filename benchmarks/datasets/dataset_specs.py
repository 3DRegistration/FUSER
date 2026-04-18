DATASET_SPECS = {
    "ScanNet": {
        "data_root": "./datasets/data/scannet",
        "view_gap": 20,
        "image_num": 30,
        "voxel_size": 0.03,
        "n_max_points": 15000,
        "scene_ids": [
            "scene0197_01", "scene0030_02", "scene0406_02", "scene0694_00", "scene0701_01",
            "scene0457_01", "scene0208_00", "scene0578_01", "scene0286_02", "scene0569_00",
            "scene0309_00", "scene0265_02", "scene0588_02", "scene0474_01", "scene0477_01",
            "scene0334_02", "scene0353_00", "scene0043_00", "scene0224_00", "scene0661_00",
            "scene0335_02", "scene0231_01", "scene0025_01", "scene0642_02", "scene0493_01",
            "scene0057_01", "scene0575_02", "scene0146_02", "scene0223_00", "scene0262_01",
            "scene0229_01", "scene0676_01",
        ]
    },
    "3DMatchFrame": {
        "data_root": "./datasets/data/3dmatch_frame",
        "view_gap": 20,
        "image_num": 60,
        "voxel_size": 0.03,
        "n_max_points": 15000,
        "scene_ids": [
            "7-scenes-redkitchen_seq-01",
            "sun3d-home_at-home_at_scan1_2013_jan_1_seq-01",
            "sun3d-home_md-home_md_scan9_2012_sep_30_seq-01",
            "sun3d-hotel_uc-scan3_seq-01",
            "sun3d-hotel_umd-maryland_hotel1_seq-01",
            "sun3d-hotel_umd-maryland_hotel3_seq-01",
            "sun3d-mit_76_studyroom-76-1studyroom2_seq-01",
            "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika_seq-01"
        ]
    },
    "ArkitScenes": {
        "data_root": "./datasets/data/arkitscenes",
        "view_gap": 3,
        "image_num": 200,
        "voxel_size": 0.03,
        "n_max_points": 15000,
        "scene_ids": [
            "41069021", "41069050", "47333440", "47333457", "47333925",
            "47430468", "47895353", "47895371", "47895542", "47895549",
            "47895783", "48018361", "48018560", "48018970", "48458732",
        ]
    },
}

SUPPORTED_BENCHMARKS = tuple(DATASET_SPECS.keys())

def get_dataset_spec(benchmark_name):
    if benchmark_name not in DATASET_SPECS:
        raise ValueError(f"Unsupported db_nm: {benchmark_name}. Expected one of {SUPPORTED_BENCHMARKS}.")
    return DATASET_SPECS[benchmark_name]
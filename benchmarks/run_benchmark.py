import os
import time
import argparse
import logging
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

if __package__ and "." in __package__:
    from ..cfgs.cfg import cfg as base_cfg
    from ..models.model_factory import SUPPORTED_MODEL_NAMES, build_model, load_model_weights
    from .datasets.dataset_specs import SUPPORTED_BENCHMARKS, get_dataset_spec
    from .datasets.build_dataset import build_dataset
    from .export import export_pred_poses
    from .evaluate import evaluate_scene, aggregate_results, build_evluation_summary_text, save_summary_report
    from ..utils.checkpoints import log_checkpoint_info
else:
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from cfgs.cfg import cfg as base_cfg
    from models.model_factory import SUPPORTED_MODEL_NAMES, build_model, load_model_weights
    from benchmarks.datasets.dataset_specs import SUPPORTED_BENCHMARKS, get_dataset_spec
    from benchmarks.datasets.build_dataset import build_dataset
    from benchmarks.export import export_pred_poses
    from benchmarks.evaluate import (
        evaluate_scene,
        aggregate_results,
        build_evluation_summary_text,
        save_summary_report,
    )
    from utils.checkpoints import log_checkpoint_info

LOGGER = logging.getLogger(__name__)

torch.manual_seed(base_cfg.seed)
np.random.seed(base_cfg.seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for FUSER and FUSER-DF.")

    parser.add_argument("--model_name", type=str, choices=SUPPORTED_MODEL_NAMES, required=True)
    parser.add_argument("--benchmark_name", type=str, choices=SUPPORTED_BENCHMARKS, required=True)

    parser.add_argument("--fuser_checkpoint", "--checkpoint", dest="fuser_checkpoint", type=str, default=None)
    parser.add_argument("--prior_checkpoint", type=str, default=None)
    parser.add_argument("--surrogate_checkpoint", type=str, default=None)

    parser.add_argument("--save_poses", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()
    if args.model_name == "FUSER" and not args.fuser_checkpoint:
        parser.error("--fuser_checkpoint (alias: --checkpoint) is required for FUSER.")
    if args.model_name == "FUSER_DF" and (not args.prior_checkpoint or not args.surrogate_checkpoint):
        parser.error("--prior_checkpoint and --surrogate_checkpoint are required for FUSER_DF.")
    return args

def build_cfg(args):
    cfg = deepcopy(base_cfg)
    cfg.output_dir = args.output_dir or f"outputs/{args.model_name}/{args.benchmark_name}"
    cfg.save_poses = args.save_poses
    cfg.data.benchmark_name = args.benchmark_name

    cfg.model.model_name = args.model_name
    cfg.model.fuser_checkpoint = args.fuser_checkpoint
    cfg.model.prior_checkpoint = args.prior_checkpoint
    cfg.model.surrogate_checkpoint = args.surrogate_checkpoint
    return cfg

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

def run_benchmark(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert cfg.test_batch_size == 1
    dataset, collate_fn = build_dataset(cfg)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=cfg.test_batch_size,
        num_workers=cfg.dataloader_num_workers,
    )
    LOGGER.info(
        "Loaded %s with %d samples using %s (batch_size=%d, num_workers=%d).",
        cfg.data.benchmark_name,
        len(dataset),
        cfg.model.model_name,
        cfg.test_batch_size,
        cfg.dataloader_num_workers,
    )

    model = build_model(cfg).to(device)
    checkpoint_info = load_model_weights(model, cfg)
    log_checkpoint_info(LOGGER, cfg, checkpoint_info)
    model.eval()

    scene_results = []
    progress_bar = tqdm(dataloader, desc=f"Evaluating {cfg.model.model_name} on {cfg.data.benchmark_name}")
    for batch in progress_bar:
        scene_id = batch["s_ids"][0]
        start_time = time.time()
        pred_res, peak_mem = run_inference(model, batch, device)
        elapsed = time.time() - start_time

        camera_poses = pred_res["camera_poses"][0]
        data_spec = get_dataset_spec(cfg.data.benchmark_name)
        scene_result = evaluate_scene(
            cfg.data.benchmark_name,
            camera_poses,
            scene_id,
            data_spec["data_root"]
        )
        scene_result["time_sec"] = elapsed
        scene_result["peak_mem"] = peak_mem
        scene_results.append(scene_result)

        LOGGER.info(
            f"{scene_id}\tPairs={scene_result['pair_count']}  "
            f"RR={scene_result['rr_percent']:.2f}%  "
            f"MeanRE={scene_result['mean_re']:.3f}  "
            f"MeanTE={scene_result['mean_te']:.3f}  "
            f"Time={elapsed:.2f}s  "
            f"Mem={peak_mem:.2f}GB"
        )

        if cfg.save_poses:
            pose_dir = export_pred_poses(
                output_dir=cfg.output_dir,
                scene_id=scene_id,
                pred_pose_seq=camera_poses
            )
            LOGGER.info("Predicted poses were exported under %s", pose_dir)

    summary = aggregate_results(scene_results)
    summary_text = build_evluation_summary_text(summary, cfg.data.benchmark_name, cfg.model.model_name, checkpoint_info)
    print("\n" + summary_text + "\n")
    report_path = save_summary_report(cfg.output_dir, summary_text)
    LOGGER.info("Summary saved to %s", report_path)

def main():
    args = parse_args()
    cfg = build_cfg(args)
    run_benchmark(cfg)


if __name__ == "__main__":
    main()
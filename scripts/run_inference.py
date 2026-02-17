"""
Run ensemble inference and evaluate uncertainty indicators.

This script:
1. Loads K trained SECOND checkpoints
2. Runs inference on each member independently
3. Associates detections across members (Stage 2)
4. Computes uncertainty indicators
5. Matches to ground truth (Stage 3)
6. Runs full evaluation pipeline (Stages 4-5)
7. Saves results and generates figures

Prerequisites:
    - OpenPCDet installed
    - K trained model checkpoints (from train_ensemble.sh)
    - Test dataset prepared in KITTI format

Usage:
    python scripts/run_inference.py \\
        --cfg_file tools/cfgs/kitti_models/second.yaml \\
        --ckpt_dirs output/ensemble/seed_0 output/ensemble/seed_1 ... \\
        --data_path data/kitti/testing \\
        --output_dir results/
"""

import argparse
import os
import sys
import pickle
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ensemble inference for SOTIF uncertainty evaluation."
    )
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/kitti_models/second.yaml",
        help="OpenPCDet config file for SECOND.",
    )
    parser.add_argument(
        "--ckpt_dirs",
        nargs="+",
        required=True,
        help="Directories containing trained checkpoints (one per member).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/kitti/testing",
        help="Path to test dataset in KITTI format.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.1,
        help="Minimum score threshold per member (before aggregation).",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="BEV IoU threshold for matching to ground truth.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )
    return parser.parse_args()


def find_checkpoint(ckpt_dir):
    """Find the best checkpoint in a directory."""
    ckpt_dir = Path(ckpt_dir)

    # Look for checkpoint in common locations
    candidates = list(ckpt_dir.glob("ckpt/checkpoint_epoch_*.pth"))
    if not candidates:
        candidates = list(ckpt_dir.glob("*.pth"))
    if not candidates:
        candidates = list(ckpt_dir.rglob("*.pth"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # Prefer best model or latest
    for c in candidates:
        if "best" in c.name:
            return str(c)

    # Sort by epoch number and return latest
    return str(sorted(candidates)[-1])


def run_member_inference(cfg_file, ckpt_path, data_path, batch_size, score_thresh):
    """
    Run inference for a single ensemble member using OpenPCDet.

    Returns list of dicts, one per frame, with keys:
        'boxes_lidar': (N, 7) array
        'score': (N,) array
        'pred_labels': (N,) array
        'frame_id': str
    """
    try:
        # Attempt to import OpenPCDet
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
        import torch

        cfg_from_yaml_file(cfg_file, cfg)
        cfg.DATA_CONFIG.DATA_PATH = data_path

        # Build dataloader
        test_set, test_loader, _ = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=batch_size,
            dist=False,
            workers=4,
            training=False,
        )

        # Build and load model
        model = build_network(
            model_cfg=cfg.MODEL,
            num_class=len(cfg.CLASS_NAMES),
            dataset=test_set,
        )
        model.load_params_from_file(filename=ckpt_path, to_cpu=True)
        model.cuda()
        model.eval()

        predictions = []
        with torch.no_grad():
            for batch_dict in test_loader:
                load_data_to_gpu(batch_dict)
                pred_dicts, _ = model(batch_dict)

                for i, pred in enumerate(pred_dicts):
                    mask = pred["pred_scores"] >= score_thresh
                    predictions.append({
                        "boxes_lidar": pred["pred_boxes"][mask].cpu().numpy(),
                        "score": pred["pred_scores"][mask].cpu().numpy(),
                        "pred_labels": pred["pred_labels"][mask].cpu().numpy(),
                        "frame_id": batch_dict["frame_id"][i],
                    })

        return predictions

    except ImportError:
        print("ERROR: OpenPCDet is not installed.")
        print("Install it from: https://github.com/open-mmlab/OpenPCDet")
        print("")
        print("For a demo without OpenPCDet, use the Colab notebook with synthetic data:")
        print("  notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb")
        sys.exit(1)


def associate_detections(member_predictions, iou_thresh=0.1):
    """
    Associate detections across ensemble members by BEV IoU.

    Groups overlapping detections from different members into proposals.
    Uses greedy clustering: for each detection, merge with existing proposal
    if BEV IoU > threshold, else create new proposal.

    Returns:
        proposals: list of dicts with keys:
            'boxes': (K, 7) array (NaN for missing members)
            'scores': (K,) array (0 for missing members)
            'frame_id': str
    """
    from sotif_uncertainty.matching import compute_bev_iou

    K = len(member_predictions)

    # Group by frame
    frames = {}
    for k, preds in enumerate(member_predictions):
        for pred in preds:
            fid = pred["frame_id"]
            if fid not in frames:
                frames[fid] = {mk: [] for mk in range(K)}
            frames[fid][k].append(pred)

    all_proposals = []

    for fid, frame_preds in frames.items():
        proposals = []  # List of (boxes: K x 7, scores: K)

        for k in range(K):
            for pred in frame_preds[k]:
                for box_idx in range(len(pred["score"])):
                    box = pred["boxes_lidar"][box_idx]
                    score = pred["score"][box_idx]

                    # Try to match with existing proposal
                    matched = False
                    for p in proposals:
                        for pk in range(K):
                            if not np.any(np.isnan(p["boxes"][pk])):
                                iou = compute_bev_iou(box, p["boxes"][pk])
                                if iou > iou_thresh:
                                    p["boxes"][k] = box
                                    p["scores"][k] = score
                                    matched = True
                                    break
                        if matched:
                            break

                    if not matched:
                        new_p = {
                            "boxes": np.full((K, 7), np.nan),
                            "scores": np.zeros(K),
                            "frame_id": fid,
                        }
                        new_p["boxes"][k] = box
                        new_p["scores"][k] = score
                        proposals.append(new_p)

        all_proposals.extend(proposals)

    return all_proposals


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    K = len(args.ckpt_dirs)
    print(f"Ensemble size: K={K}")
    print(f"Config: {args.cfg_file}")

    # Stage 1: Ensemble inference
    print("\n=== Stage 1: Ensemble Inference ===")
    all_member_preds = []
    for k, ckpt_dir in enumerate(args.ckpt_dirs):
        ckpt_path = find_checkpoint(ckpt_dir)
        print(f"  Member {k}: {ckpt_path}")
        preds = run_member_inference(
            args.cfg_file, ckpt_path, args.data_path,
            args.batch_size, args.score_thresh,
        )
        all_member_preds.append(preds)
        print(f"    -> {sum(len(p['score']) for p in preds)} detections across {len(preds)} frames")

    # Stage 2: Association and uncertainty
    print("\n=== Stage 2: Association and Uncertainty Indicators ===")
    proposals = associate_detections(all_member_preds)

    scores_array = np.array([p["scores"] for p in proposals])
    boxes_array = np.array([p["boxes"] for p in proposals])
    frame_ids = np.array([p["frame_id"] for p in proposals])

    from sotif_uncertainty.uncertainty import compute_all_indicators
    indicators = compute_all_indicators(scores_array, boxes_array)

    print(f"  Total proposals: {len(proposals)}")
    print(f"  Mean confidence range: [{indicators['mean_confidence'].min():.3f}, "
          f"{indicators['mean_confidence'].max():.3f}]")

    # Save intermediate results
    results = {
        "scores": scores_array,
        "boxes": boxes_array,
        "frame_ids": frame_ids,
        "indicators": indicators,
        "K": K,
    }

    output_path = os.path.join(args.output_dir, "ensemble_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)

    print(f"\n  Results saved to: {output_path}")
    print(f"\n  Next: Run the evaluation notebook to compute metrics and SOTIF artefacts.")
    print(f"  Load data with:")
    print(f"    with open('{output_path}', 'rb') as f:")
    print(f"        results = pickle.load(f)")


if __name__ == "__main__":
    main()

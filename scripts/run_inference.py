"""
Run ensemble inference and evaluate uncertainty indicators.

This script:
1. Loads K trained SECOND checkpoints via OpenPCDet
2. Runs inference on each member independently
3. Associates detections via DBSCAN clustering (Stage 2)
4. Computes uncertainty indicators
5. Optionally matches to ground truth (Stage 3) and runs full evaluation

Prerequisites:
    - OpenPCDet installed (https://github.com/open-mmlab/OpenPCDet)
    - K trained model checkpoints (from train_ensemble.sh)
    - KITTI dataset prepared (see scripts/prepare_kitti.py)

Usage:
    # Standard ensemble inference (K=6 members)
    python scripts/run_inference.py \\
        --ckpt_dirs output/ensemble/seed_0 output/ensemble/seed_1 \\
                    output/ensemble/seed_2 output/ensemble/seed_3 \\
                    output/ensemble/seed_4 output/ensemble/seed_5 \\
        --data_path data/kitti

    # With ground truth evaluation
    python scripts/run_inference.py \\
        --ckpt_dirs output/ensemble/seed_* \\
        --data_path data/kitti \\
        --split val \\
        --gt_path data/kitti/training/label_2

    # Use existing prediction pickles (skip inference)
    python scripts/run_inference.py \\
        --pkl_files results/seed_0.pkl results/seed_1.pkl ... \\
        --gt_path data/kitti/training/label_2

    # Voting strategies (affirmative / consensus / unanimous)
    python scripts/run_inference.py \\
        --ckpt_dirs output/ensemble/seed_* \\
        --voting consensus

OpenPCDet output format per member per frame:
    {
        'name':        np.array(['Car', ...]),       # class names
        'score':       np.array([0.92, ...]),         # confidence scores
        'boxes_lidar': np.array([[x,y,z,dx,dy,dz,heading], ...]),  # 7-DOF
        'pred_labels': np.array([1, ...]),            # 1-indexed class labels
    }

After DBSCAN clustering, proposals get K-dimensional score vectors and
uncertainty indicators for the SOTIF evaluation pipeline.
"""

import argparse
import os
import sys
import pickle
import time
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ensemble inference for SOTIF uncertainty evaluation."
    )

    # Input options (checkpoint dirs OR pre-computed pickles)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--ckpt_dirs",
        nargs="+",
        help="Directories containing trained checkpoints (one per member).",
    )
    input_group.add_argument(
        "--pkl_files",
        nargs="+",
        help="Pre-computed prediction pickle files (one per member). "
             "Each pickle is a list[dict] in OpenPCDet format.",
    )

    # Model config
    parser.add_argument(
        "--cfg_file",
        type=str,
        default="tools/cfgs/kitti_models/second.yaml",
        help="OpenPCDet config file for SECOND.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/kitti",
        help="Path to dataset root in KITTI format.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on.",
    )

    # Ground truth
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to ground truth labels (KITTI format label_2 directory). "
             "If provided, runs full evaluation with TP/FP matching.",
    )

    # Clustering / aggregation
    parser.add_argument(
        "--voting",
        type=str,
        default="consensus",
        choices=["affirmative", "consensus", "unanimous"],
        help="Voting strategy for DBSCAN clustering. "
             "affirmative=keep all, consensus=majority, unanimous=all agree.",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="BEV IoU threshold for DBSCAN clustering and GT matching.",
    )

    # Inference parameters
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.1,
        help="Minimum score threshold per member (before aggregation).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference.",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--save_member_preds",
        action="store_true",
        help="Save individual member predictions as pickle files.",
    )

    return parser.parse_args()


def find_checkpoint(ckpt_dir):
    """Find the best or latest checkpoint in a directory."""
    ckpt_dir = Path(ckpt_dir)

    # Look for checkpoint in common locations
    candidates = list(ckpt_dir.glob("ckpt/checkpoint_epoch_*.pth"))
    if not candidates:
        candidates = list(ckpt_dir.glob("*.pth"))
    if not candidates:
        candidates = list(ckpt_dir.rglob("*.pth"))

    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    # Prefer best model if available
    for c in candidates:
        if "best" in c.name:
            return str(c)

    # Sort by epoch number and return latest
    return str(sorted(candidates)[-1])


def run_member_inference(cfg_file, ckpt_path, data_path, split, batch_size, score_thresh):
    """
    Run inference for a single ensemble member using OpenPCDet.

    Returns list of dicts in OpenPCDet format, one per frame.
    """
    try:
        from pcdet.config import cfg, cfg_from_yaml_file
        from pcdet.datasets import build_dataloader
        from pcdet.models import build_network, load_data_to_gpu
        from pcdet.utils import common_utils
        import torch

        cfg_from_yaml_file(cfg_file, cfg)
        cfg.DATA_CONFIG.DATA_PATH = data_path

        # Build dataloader for the specified split
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
                    boxes = pred["pred_boxes"][mask].cpu().numpy()

                    # KITTI Z-fix: OpenPCDet KITTI boxes have Z at bottom,
                    # add height/2 to center it (from LiDAR-MIMO)
                    if len(boxes) > 0:
                        boxes[:, 2] += boxes[:, 5] / 2

                    predictions.append({
                        "boxes_lidar": boxes,
                        "score": pred["pred_scores"][mask].cpu().numpy(),
                        "pred_labels": pred["pred_labels"][mask].cpu().numpy(),
                        "frame_id": batch_dict["frame_id"][i],
                    })

        return predictions

    except ImportError:
        print("=" * 60)
        print("ERROR: OpenPCDet is not installed.")
        print("=" * 60)
        print("")
        print("Option 1: Install OpenPCDet")
        print("  git clone https://github.com/open-mmlab/OpenPCDet.git")
        print("  cd OpenPCDet && pip install -r requirements.txt")
        print("  python setup.py develop")
        print("")
        print("Option 2: Use pre-computed pickles")
        print("  python scripts/run_inference.py \\")
        print("    --pkl_files results/seed_0.pkl results/seed_1.pkl ...")
        print("")
        print("Option 3: Demo mode (synthetic data, no GPU needed)")
        print("  python scripts/evaluate.py")
        print("  # or open notebooks/SOTIF_Uncertainty_Evaluation_Demo.ipynb")
        sys.exit(1)


def load_kitti_labels(gt_path, frame_id):
    """
    Load KITTI ground truth labels for a frame.

    Returns np.ndarray of shape (N, 7) with 3D boxes in LiDAR frame.
    """
    label_file = os.path.join(gt_path, f"{frame_id}.txt")
    if not os.path.exists(label_file):
        return np.zeros((0, 7))

    boxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in ("Car", "Pedestrian", "Cyclist"):
                continue
            # KITTI format: h, w, l, x, y, z, ry (camera frame)
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            # Note: full pipeline would need calib to convert camera -> LiDAR
            # This is a simplified version; real usage needs calibration matrices
            boxes.append([x, y, z, w, l, h, ry])

    return np.array(boxes) if boxes else np.zeros((0, 7))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # Stage 1: Ensemble Inference
    # =========================================================
    print("=" * 60)
    print("STAGE 1: Ensemble Inference")
    print("=" * 60)

    if args.pkl_files:
        # Load pre-computed predictions
        K = len(args.pkl_files)
        print(f"Loading {K} pre-computed prediction files...")
        all_member_preds = []
        for k, pkl_path in enumerate(args.pkl_files):
            print(f"  Member {k}: {pkl_path}")
            with open(pkl_path, "rb") as f:
                preds = pickle.load(f)
            # Normalize format: ensure each pred has required keys
            normalized = []
            for pred in preds:
                entry = {
                    "boxes_lidar": pred.get("boxes_lidar", pred.get("pred_boxes", np.zeros((0, 7)))),
                    "score": pred.get("score", pred.get("pred_scores", np.zeros(0))),
                    "pred_labels": pred.get("pred_labels", np.ones(len(pred.get("score", [])), dtype=int)),
                    "frame_id": pred.get("frame_id", str(len(normalized))),
                }
                normalized.append(entry)
            all_member_preds.append(normalized)
            n_det = sum(len(p["score"]) for p in normalized)
            print(f"    -> {n_det} detections across {len(normalized)} frames")
    else:
        # Run inference with OpenPCDet
        K = len(args.ckpt_dirs)
        print(f"Ensemble size: K={K}")
        print(f"Config: {args.cfg_file}")
        print(f"Data: {args.data_path} (split: {args.split})")
        print(f"Score threshold: {args.score_thresh}")
        print()

        all_member_preds = []
        for k, ckpt_dir in enumerate(args.ckpt_dirs):
            ckpt_path = find_checkpoint(ckpt_dir)
            print(f"  Member {k}: {ckpt_path}")
            t0 = time.time()
            preds = run_member_inference(
                args.cfg_file, ckpt_path, args.data_path,
                args.split, args.batch_size, args.score_thresh,
            )
            elapsed = time.time() - t0
            n_det = sum(len(p["score"]) for p in preds)
            print(f"    -> {n_det} detections across {len(preds)} frames ({elapsed:.1f}s)")

            all_member_preds.append(preds)

            # Optionally save individual predictions
            if args.save_member_preds:
                member_pkl = os.path.join(args.output_dir, f"member_{k}_preds.pkl")
                with open(member_pkl, "wb") as f:
                    pickle.dump(preds, f)

    # =========================================================
    # Stage 2: DBSCAN Clustering + Uncertainty Indicators
    # =========================================================
    print()
    print("=" * 60)
    print("STAGE 2: Detection Association (DBSCAN) + Uncertainty")
    print("=" * 60)
    print(f"  IoU threshold: {args.iou_thresh}")
    print(f"  Voting: {args.voting}")

    from sotif_uncertainty.ensemble import (
        cluster_detections,
        clustered_to_pipeline_format,
    )
    from sotif_uncertainty.uncertainty import compute_all_indicators

    clustered = cluster_detections(
        all_member_preds,
        iou_threshold=args.iou_thresh,
        voting=args.voting,
    )

    # Convert to pipeline format
    pipeline_data = clustered_to_pipeline_format(clustered, K)
    scores = pipeline_data["scores"]
    boxes = pipeline_data["boxes"]
    frame_ids = pipeline_data["frame_ids"]

    # Compute uncertainty indicators
    indicators = compute_all_indicators(scores, boxes)
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    n_proposals = len(mean_conf)
    print(f"\n  Total proposals after clustering: {n_proposals}")
    print(f"  Mean confidence range: [{mean_conf.min():.3f}, {mean_conf.max():.3f}]")
    print(f"  Confidence variance range: [{conf_var.min():.5f}, {conf_var.max():.5f}]")
    print(f"  Geometric disagreement range: [{geo_disagree.min():.3f}, {geo_disagree.max():.3f}]")

    # Per-frame summary
    unique_frames = np.unique(frame_ids)
    print(f"  Frames: {len(unique_frames)}")
    dets_per_frame = [np.sum(frame_ids == fid) for fid in unique_frames]
    print(f"  Detections/frame: mean={np.mean(dets_per_frame):.1f}, "
          f"min={np.min(dets_per_frame)}, max={np.max(dets_per_frame)}")

    # =========================================================
    # Stage 3: Ground Truth Matching (if labels provided)
    # =========================================================
    labels = None
    if args.gt_path:
        print()
        print("=" * 60)
        print("STAGE 3: Ground Truth Matching")
        print("=" * 60)

        from sotif_uncertainty.matching import greedy_match

        # Build aggregated boxes for matching
        agg_boxes = np.zeros((n_proposals, 7))
        for i, frame in enumerate(clustered):
            for p in range(len(frame["mean_score"])):
                idx = np.where(
                    (frame_ids == frame["frame_id"])
                )[0]
                if len(idx) > p:
                    agg_boxes[idx[p]] = frame["boxes_lidar"][p]

        labels = np.zeros(n_proposals, dtype=int)
        tp_total, fp_total, fn_total = 0, 0, 0

        for fid in unique_frames:
            mask = frame_ids == fid
            frame_boxes = agg_boxes[mask]
            frame_scores = mean_conf[mask]

            gt_boxes = load_kitti_labels(args.gt_path, fid)

            if len(frame_boxes) == 0 and len(gt_boxes) == 0:
                continue

            match_result = greedy_match(
                frame_boxes, frame_scores, gt_boxes,
                iou_threshold=args.iou_thresh,
            )

            labels[mask] = match_result["labels"]
            tp_total += match_result["tp_count"]
            fp_total += match_result["fp_count"]
            fn_total += match_result["fn_count"]

        print(f"  TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")
        print(f"  Precision: {tp_total/(tp_total+fp_total):.3f}" if tp_total+fp_total > 0 else "")
        print(f"  Recall: {tp_total/(tp_total+fn_total):.3f}" if tp_total+fn_total > 0 else "")

    # =========================================================
    # Save results
    # =========================================================
    results = {
        "scores": scores,
        "boxes": boxes,
        "frame_ids": frame_ids,
        "indicators": indicators,
        "K": K,
        "voting": args.voting,
        "iou_thresh": args.iou_thresh,
        "clustered_frames": clustered,
    }
    if labels is not None:
        results["labels"] = labels

    output_path = os.path.join(args.output_dir, "ensemble_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  Results saved to: {output_path}")

    # =========================================================
    # Stage 4-5: Full evaluation (if labels available)
    # =========================================================
    if labels is not None:
        print()
        print("=" * 60)
        print("STAGE 4: Metric Computation")
        print("=" * 60)

        from sotif_uncertainty.metrics import compute_all_metrics
        metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, labels)

        disc = metrics["discrimination"]
        cal = metrics["calibration"]
        rc = metrics["risk_coverage"]

        print(f"\n  AUROC (mean confidence):       {disc['auroc_mean_confidence']:.3f}")
        print(f"  AUROC (confidence variance):   {disc['auroc_confidence_variance']:.3f}")
        print(f"  AUROC (geometric disagreement):{disc['auroc_geometric_disagreement']:.3f}")
        print(f"  ECE: {cal['ece']:.3f}  NLL: {cal['nll']:.3f}  Brier: {cal['brier']:.3f}")
        print(f"  AURC: {rc['aurc']:.3f}")

        print()
        print("=" * 60)
        print("STAGE 5: SOTIF Analysis")
        print("=" * 60)

        from sotif_uncertainty.sotif_analysis import (
            compute_operating_points, flag_frames,
        )

        points = compute_operating_points(
            mean_conf, conf_var, geo_disagree, labels,
            tau_s_range=np.array([0.50, 0.60, 0.65, 0.70, 0.80, 0.85, 0.90]),
        )
        print(f"\n  {'Gate':<35} {'Cov':>5} {'Ret':>4} {'FP':>3} {'FAR':>6}")
        print("  " + "-" * 55)
        for p in points:
            print(f"  {p['gate']:<35} {p['coverage']:>5.3f} "
                  f"{p['retained']:>4d} {p['fp']:>3d} {p['far']:>6.3f}")

        flags = flag_frames(frame_ids, labels, conf_var)
        print(f"\n  Frame triage: {flags['flagged_count']}/{flags['total_frames']} flagged")

        # Generate figures
        figures_dir = os.path.join(args.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        from sotif_uncertainty.sotif_analysis import compute_frame_summary
        frame_summaries = compute_frame_summary(frame_ids, labels, mean_conf, conf_var)

        from sotif_uncertainty.visualization import generate_all_figures
        figures = generate_all_figures(
            metrics=metrics,
            mean_conf=mean_conf,
            conf_var=conf_var,
            labels=labels,
            frame_summaries=frame_summaries,
            tc_results=[],
            operating_points=points,
            output_dir=figures_dir,
            scores=scores,
            geo_disagree=geo_disagree,
        )
        print(f"\n  Saved {len(figures)} figures to {figures_dir}/")

    else:
        print()
        print("No ground truth provided. To run full evaluation:")
        print(f"  python scripts/evaluate.py --input {output_path} \\")
        print(f"    --gt_path data/kitti/training/label_2")
        print()
        print("Or load in Python:")
        print(f"  import pickle")
        print(f"  with open('{output_path}', 'rb') as f:")
        print(f"      results = pickle.load(f)")

    print()
    print("=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

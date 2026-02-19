"""
Standalone evaluation script.

Runs the full evaluation pipeline (Stages 2-5) on either:
- Synthetic demo data (default)
- Real ensemble results from run_inference.py (pickle format)
- KITTI/CARLA dataset with ground truth labels

Usage:
    # Demo mode (no dependencies beyond numpy/matplotlib)
    python scripts/evaluate.py

    # Real data mode (from ensemble inference pickle)
    python scripts/evaluate.py --input results/ensemble_results.pkl

    # Real data mode with separate GT path and calibration
    python scripts/evaluate.py \
        --input results/ensemble_results.pkl \
        --gt_path data/kitti/training/label_2 \
        --calib_path data/kitti/training/calib

    # CARLA-generated data with condition metadata
    python scripts/evaluate.py \
        --input results/ensemble_results.pkl \
        --gt_path data/carla/training/label_2 \
        --conditions_file data/carla/conditions.json

    # Custom output directory and thresholds
    python scripts/evaluate.py --output_dir my_results/ --iou_thresh 0.5
"""

import argparse
import json
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SOTIF uncertainty evaluation pipeline."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to ensemble_results.pkl from run_inference.py. "
             "If not provided, synthetic demo data is used.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default=None,
        help="Path to ground truth labels directory (KITTI format label_2/).",
    )
    parser.add_argument(
        "--calib_path",
        type=str,
        default=None,
        help="Path to calibration files directory (KITTI format calib/).",
    )
    parser.add_argument(
        "--conditions_file",
        type=str,
        default=None,
        help="Path to conditions.json (from CARLA data generation). "
             "Maps frame IDs to triggering condition categories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Analysis",
        help="Output directory for figures and results.",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="BEV IoU threshold for GT matching (default: 0.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "carla_study", "real"],
        default=None,
        help="Evaluation mode: demo (abstract stats), carla_study (Section 5), "
             "real (--input required). If not specified, uses demo when no --input.",
    )
    return parser.parse_args()


def load_gt_for_frame(frame_id, gt_path, calib_path=None):
    """
    Load ground truth boxes for a single frame.

    Supports both:
    - LiDAR-frame labels (if no calib_path, assumes labels are already in LiDAR)
    - Camera-frame labels + calibration (standard KITTI format)

    Returns
    -------
    np.ndarray, shape (M, 7)
        GT boxes in LiDAR frame [x, y, z, dx, dy, dz, heading].
    """
    label_file = os.path.join(gt_path, f"{frame_id}.txt")
    if not os.path.exists(label_file):
        return np.zeros((0, 7))

    if calib_path is not None:
        calib_file = os.path.join(calib_path, f"{frame_id}.txt")
        if os.path.exists(calib_file):
            from sotif_uncertainty.kitti_utils import load_kitti_labels_as_lidar
            return load_kitti_labels_as_lidar(label_file, calib_file)

    # Fallback: parse labels directly (assumes LiDAR frame or simplified format)
    boxes = []
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls not in ("Car", "Pedestrian", "Cyclist"):
                continue
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])
            boxes.append([x, y, z, l, w, h, ry])

    return np.array(boxes) if boxes else np.zeros((0, 7))


def load_conditions_metadata(conditions_file, frame_ids):
    """
    Load per-frame condition assignments from a JSON metadata file.

    Parameters
    ----------
    conditions_file : str
        Path to conditions.json.
    frame_ids : np.ndarray
        Frame IDs for each detection.

    Returns
    -------
    np.ndarray, shape (N,)
        Condition category per detection.
    """
    with open(conditions_file, "r") as f:
        metadata = json.load(f)

    conditions = np.empty(len(frame_ids), dtype="U20")
    for i, fid in enumerate(frame_ids):
        fid_str = str(fid)
        if fid_str in metadata:
            conditions[i] = metadata[fid_str].get("tc_category", "other")
        else:
            conditions[i] = "other"
    return conditions


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    conditions = None
    raw_scores = None

    # ==========================================================
    # Load or generate data
    # ==========================================================
    if args.input is not None:
        import pickle
        print(f"Loading ensemble results from: {args.input}")
        with open(args.input, "rb") as f:
            results = pickle.load(f)

        scores = results["scores"]
        boxes = results.get("boxes", None)
        frame_ids = results["frame_ids"]
        raw_scores = scores
        K = results.get("K", scores.shape[1] if scores.ndim == 2 else 6)

        # Check if labels are already in the pickle (from run_inference.py)
        if "labels" in results and results["labels"] is not None:
            labels = results["labels"]
            print(f"  Labels found in pickle: {np.sum(labels==1)} TP, {np.sum(labels==0)} FP")
        elif args.gt_path is not None:
            # Perform GT matching
            print(f"  Running ground truth matching (IoU >= {args.iou_thresh})...")

            from sotif_uncertainty.uncertainty import compute_all_indicators
            from sotif_uncertainty.matching import greedy_match

            # Compute indicators first (needed for aggregated boxes)
            indicators = compute_all_indicators(scores, boxes)
            mean_conf_temp = indicators["mean_confidence"]

            # Build aggregated boxes for matching
            clustered_frames = results.get("clustered_frames", None)
            if clustered_frames is not None:
                agg_boxes = np.zeros((len(mean_conf_temp), 7))
                idx_offset = 0
                for frame in clustered_frames:
                    n_props = len(frame["mean_score"])
                    for p in range(n_props):
                        agg_boxes[idx_offset + p] = frame["boxes_lidar"][p]
                    idx_offset += n_props
            else:
                # Approximate from mean of per-member boxes
                agg_boxes = np.nanmean(boxes, axis=1) if boxes is not None else np.zeros((len(mean_conf_temp), 7))

            # Match per frame
            unique_frames = np.unique(frame_ids)
            labels = np.zeros(len(mean_conf_temp), dtype=int)
            tp_total, fp_total, fn_total = 0, 0, 0

            for fid in unique_frames:
                mask = frame_ids == fid
                frame_boxes = agg_boxes[mask]
                frame_scores = mean_conf_temp[mask]

                gt_boxes = load_gt_for_frame(
                    str(fid), args.gt_path, args.calib_path
                )

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

            print(f"  Ground truth matching complete:")
            print(f"    TP: {tp_total}, FP: {fp_total}, FN: {fn_total}")
            if tp_total + fp_total > 0:
                print(f"    Precision: {tp_total/(tp_total+fp_total):.3f}")
            if tp_total + fn_total > 0:
                print(f"    Recall: {tp_total/(tp_total+fn_total):.3f}")
        else:
            print("  ERROR: No labels found and no --gt_path specified.")
            print("  Provide ground truth: --gt_path data/kitti/training/label_2")
            print("  Or use demo mode: python scripts/evaluate.py (no --input)")
            sys.exit(1)

        # Load conditions if available
        if args.conditions_file:
            conditions = load_conditions_metadata(args.conditions_file, frame_ids)
            print(f"  Loaded conditions from: {args.conditions_file}")

    elif args.mode == "carla_study" or (args.mode is None and args.input is None):
        # Determine which synthetic data generator to use
        if args.mode == "carla_study":
            print("Using CARLA case study data (Section 5 of paper).")
            from sotif_uncertainty.carla_case_study import generate_carla_case_study
            data = generate_carla_case_study(seed=args.seed)
        else:
            print("Using synthetic demo data (matching paper abstract statistics).")
            from sotif_uncertainty.demo_data import generate_demo_dataset
            data = generate_demo_dataset(seed=args.seed)
        scores = data["scores"]
        boxes = data["boxes"]
        labels = data["labels"]
        frame_ids = data["frame_ids"]
        conditions = data["conditions"]
        raw_scores = scores

    # ==========================================================
    # Stage 2: Uncertainty Indicators
    # ==========================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Uncertainty Indicators")
    print("=" * 60)

    from sotif_uncertainty.uncertainty import compute_all_indicators
    indicators = compute_all_indicators(scores, boxes)
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators.get("geometric_disagreement", np.zeros_like(mean_conf))

    tp_mask = labels == 1
    fp_mask = labels == 0

    print(f"  Proposals: {len(labels)} ({np.sum(tp_mask)} TP, {np.sum(fp_mask)} FP)")
    if np.sum(tp_mask) > 0:
        print(f"  Mean confidence: TP={np.mean(mean_conf[tp_mask]):.3f}, FP={np.mean(mean_conf[fp_mask]):.3f}")
        print(f"  Conf variance:   TP={np.mean(conf_var[tp_mask]):.5f}, FP={np.mean(conf_var[fp_mask]):.5f}")

    # ==========================================================
    # Stage 4: Metrics
    # ==========================================================
    print("\n" + "=" * 60)
    print("STAGE 4: Metric Computation")
    print("=" * 60)

    from sotif_uncertainty.metrics import compute_all_metrics
    metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, labels)

    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    print("\n  Discrimination (Table 3):")
    print(f"    AUROC (mean confidence):          {disc['auroc_mean_confidence']:.3f}")
    print(f"    AUROC (confidence variance):       {disc['auroc_confidence_variance']:.3f}")
    print(f"    AUROC (geometric disagreement):    {disc['auroc_geometric_disagreement']:.3f}")

    print("\n  Calibration (Table 4):")
    print(f"    ECE:         {cal['ece']:.3f}")
    print(f"    NLL:         {cal['nll']:.3f}")
    print(f"    Brier Score: {cal['brier']:.3f}")
    print(f"    AURC:        {rc['aurc']:.3f}")

    # ==========================================================
    # Stage 5: SOTIF Analysis
    # ==========================================================
    print("\n" + "=" * 60)
    print("STAGE 5: SOTIF Analysis")
    print("=" * 60)

    from sotif_uncertainty.sotif_analysis import (
        acceptance_gate,
        compute_coverage_far,
        compute_operating_points,
        rank_triggering_conditions,
        flag_frames,
        compute_frame_summary,
    )

    # Operating points (comprehensive set for paper tables)
    # Confidence-only gates
    conf_only_points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_s_range=np.array([0.35, 0.50, 0.60, 0.65, 0.70, 0.80]),
    )
    # Confidence + variance gates
    conf_var_points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_s_range=np.array([0.50, 0.60, 0.65]),
        tau_v_range=np.array([0.002, 0.005, 0.010]),
    )
    # Confidence + geometric disagreement gates
    conf_geo_points = []
    for ts in [0.35, 0.50]:
        for td in [0.30, 0.49, 0.60]:
            accepted = acceptance_gate(mean_conf, conf_var, geo_disagree,
                                       tau_s=ts, tau_d=td)
            cov, far, ret, fp_count = compute_coverage_far(accepted, labels)
            conf_geo_points.append({
                "gate": f"s>={ts:.2f} & d<={td:.2f}",
                "tau_s": ts, "tau_d": td, "tau_v": float("inf"),
                "coverage": cov, "retained": ret, "fp": fp_count, "far": far,
            })
    # Geometric disagreement-only gates
    geo_only_points = []
    for td in [0.20, 0.30, 0.40, 0.50]:
        accepted = acceptance_gate(mean_conf, conf_var, geo_disagree,
                                   tau_s=0.0, tau_d=td)
        cov, far, ret, fp_count = compute_coverage_far(accepted, labels)
        geo_only_points.append({
            "gate": f"d<={td:.2f}",
            "tau_s": 0.0, "tau_d": td, "tau_v": float("inf"),
            "coverage": cov, "retained": ret, "fp": fp_count, "far": far,
        })
    all_points = conf_only_points + conf_geo_points + geo_only_points + conf_var_points

    print("\n  Operating Points (Table 6):")
    print(f"  {'Gate':<35} {'Cov.':>6} {'Ret.':>5} {'FP':>4} {'FAR':>8}")
    print("  " + "-" * 60)
    for p in all_points:
        print(f"  {p['gate']:<35} {p['coverage']:>6.3f} {p['retained']:>5d} "
              f"{p['fp']:>4d} {p['far']:>8.3f}")

    # TC ranking (only if conditions available)
    tc_results = None
    if conditions is not None:
        tc_results = rank_triggering_conditions(conditions, labels, mean_conf, conf_var)

        print("\n  Triggering Condition Ranking (Table 7):")
        print(f"  {'Condition':<18} {'FP Share':>10} {'Mean s (FP)':>12} {'Mean var (FP)':>14}")
        print("  " + "-" * 56)
        for tc in tc_results:
            conf_str = f"{tc['mean_conf_fp']:.2f}" if not np.isnan(tc['mean_conf_fp']) else "N/A"
            var_str = f"{tc['mean_var_fp']:.4f}" if not np.isnan(tc['mean_var_fp']) else "N/A"
            print(f"  {tc['condition']:<18} {tc['fp_share']:>10.1%} {conf_str:>12} {var_str:>14}")
    else:
        print("\n  Triggering condition analysis skipped (no conditions data).")
        print("  Provide --conditions_file for CARLA data, or use demo mode.")
        # Create a simple breakdown without condition metadata
        tc_results = [
            {"condition": "all", "fp_count": int(np.sum(fp_mask)),
             "fp_share": 1.0,
             "mean_conf_fp": float(np.mean(mean_conf[fp_mask])) if np.sum(fp_mask) > 0 else 0.0,
             "mean_var_fp": float(np.mean(conf_var[fp_mask])) if np.sum(fp_mask) > 0 else 0.0,
             "tp_count": int(np.sum(tp_mask)), "total": len(labels)},
        ]

    # Frame flags
    flag_results = flag_frames(frame_ids, labels, conf_var)
    print(f"\n  Frame-Level Triage:")
    print(f"    Flagged: {flag_results['flagged_count']}/{flag_results['total_frames']} frames")
    print(f"    Variance threshold: {flag_results['threshold']:.5f}")

    # ==========================================================
    # Generate figures
    # ==========================================================
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    frame_summaries = compute_frame_summary(
        frame_ids, labels, mean_conf, conf_var,
        conditions if conditions is not None else None,
    )

    from sotif_uncertainty.visualization import generate_all_figures
    figures = generate_all_figures(
        metrics=metrics,
        mean_conf=mean_conf,
        conf_var=conf_var,
        labels=labels,
        frame_summaries=frame_summaries,
        tc_results=tc_results,
        operating_points=all_points,
        output_dir=args.output_dir,
        scores=raw_scores,
        geo_disagree=geo_disagree,
        conditions=conditions,
    )

    print(f"  Saved {len(figures)} figures to {args.output_dir}/")
    for name in figures:
        print(f"    - {name}")

    # ==========================================================
    # Save comprehensive numerical results
    # ==========================================================
    results_summary = {
        "dataset": {
            "n_proposals": len(labels),
            "n_tp": int(np.sum(tp_mask)),
            "n_fp": int(np.sum(fp_mask)),
            "fp_ratio": float(np.sum(fp_mask) / len(labels)),
            "n_frames": int(len(np.unique(frame_ids))),
            "K": int(scores.shape[1]) if scores.ndim == 2 else 6,
        },
        "indicator_statistics": {
            "tp": {
                "mean_confidence": {"mean": float(np.mean(mean_conf[tp_mask])),
                                    "std": float(np.std(mean_conf[tp_mask]))},
                "confidence_variance": {"mean": float(np.mean(conf_var[tp_mask])),
                                        "std": float(np.std(conf_var[tp_mask]))},
                "geometric_disagreement": {"mean": float(np.mean(geo_disagree[tp_mask])),
                                           "std": float(np.std(geo_disagree[tp_mask]))},
            },
            "fp": {
                "mean_confidence": {"mean": float(np.mean(mean_conf[fp_mask])),
                                    "std": float(np.std(mean_conf[fp_mask]))},
                "confidence_variance": {"mean": float(np.mean(conf_var[fp_mask])),
                                        "std": float(np.std(conf_var[fp_mask]))},
                "geometric_disagreement": {"mean": float(np.mean(geo_disagree[fp_mask])),
                                           "std": float(np.std(geo_disagree[fp_mask]))},
            },
        },
        "discrimination": {
            "auroc_mean_confidence": float(disc["auroc_mean_confidence"]),
            "auroc_confidence_variance": float(disc["auroc_confidence_variance"]),
            "auroc_geometric_disagreement": float(disc["auroc_geometric_disagreement"]),
        },
        "calibration": {
            "ece": float(cal["ece"]),
            "nll": float(cal["nll"]),
            "brier": float(cal["brier"]),
            "aurc": float(rc["aurc"]),
        },
        "operating_points": [
            {"gate": p["gate"], "coverage": float(p["coverage"]),
             "retained": int(p["retained"]), "fp": int(p["fp"]),
             "far": float(p["far"])}
            for p in all_points
        ],
        "triggering_conditions": [
            {"condition": tc["condition"],
             "fp_count": int(tc["fp_count"]),
             "fp_share": float(tc["fp_share"]),
             "mean_conf_fp": float(tc["mean_conf_fp"]) if not np.isnan(tc["mean_conf_fp"]) else None,
             "mean_var_fp": float(tc["mean_var_fp"]) if not np.isnan(tc["mean_var_fp"]) else None}
            for tc in tc_results
        ],
        "frame_triage": {
            "flagged_count": int(flag_results["flagged_count"]),
            "total_frames": int(flag_results["total_frames"]),
            "flagged_fraction": float(flag_results["flagged_count"] / flag_results["total_frames"])
                if flag_results["total_frames"] > 0 else 0.0,
            "variance_threshold": float(flag_results["threshold"]),
        },
    }
    results_path = os.path.join(args.output_dir, "results_summary.json")
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n  Results summary saved to: {results_path}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

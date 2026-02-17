"""
Standalone evaluation script.

Runs the full evaluation pipeline (Stages 2-5) on either:
- Synthetic demo data (default)
- Real ensemble results from run_inference.py

Usage:
    # Demo mode (no dependencies beyond numpy/matplotlib)
    python scripts/evaluate.py

    # Real data mode
    python scripts/evaluate.py --input results/ensemble_results.pkl --gt_path data/kitti/label_2/

    # Custom output directory
    python scripts/evaluate.py --output_dir my_results/
"""

import argparse
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
        help="Path to ground truth labels (KITTI format).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Analysis",
        help="Output directory for figures and results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic data generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================================
    # Load or generate data
    # ==========================================================
    if args.input is not None:
        import pickle
        print(f"Loading ensemble results from: {args.input}")
        with open(args.input, "rb") as f:
            results = pickle.load(f)
        scores = results["scores"]
        boxes = results["boxes"]
        frame_ids = results["frame_ids"]

        # Check if labels are already in the results (from run_inference.py)
        if "labels" in results:
            labels = results["labels"]
            print(f"  Labels found in results: {np.sum(labels==1)} TP, {np.sum(labels==0)} FP")
        elif args.gt_path is not None:
            # Perform GT matching using provided labels
            print(f"  Matching to ground truth at: {args.gt_path}")
            from sotif_uncertainty.uncertainty import compute_all_indicators, aggregate_box
            from sotif_uncertainty.matching import greedy_match

            indicators_tmp = compute_all_indicators(scores, boxes)
            mean_conf_tmp = indicators_tmp["mean_confidence"]
            agg_boxes = aggregate_box(boxes)

            labels = np.zeros(len(scores), dtype=int)
            unique_frames = np.unique(frame_ids)
            tp_total, fp_total, fn_total = 0, 0, 0

            for fid in unique_frames:
                mask = frame_ids == fid
                frame_boxes = agg_boxes[mask]
                frame_scores = mean_conf_tmp[mask]

                # Load GT for this frame
                gt_file = os.path.join(args.gt_path, f"{fid}.txt")
                gt_boxes_list = []
                if os.path.exists(gt_file):
                    with open(gt_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 15 and parts[0] in ("Car", "Pedestrian", "Cyclist"):
                                h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
                                x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
                                ry = float(parts[14])
                                gt_boxes_list.append([x, y, z, w, l, h, ry])

                gt_boxes = np.array(gt_boxes_list) if gt_boxes_list else np.zeros((0, 7))

                if len(frame_boxes) == 0 and len(gt_boxes) == 0:
                    continue

                match_result = greedy_match(frame_boxes, frame_scores, gt_boxes, iou_threshold=0.5)
                labels[mask] = match_result["labels"]
                tp_total += match_result["tp_count"]
                fp_total += match_result["fp_count"]
                fn_total += match_result["fn_count"]

            print(f"  Matched: {tp_total} TP, {fp_total} FP, {fn_total} FN")
        else:
            print("ERROR: --input requires either labels in the pickle or --gt_path.")
            print("  Option 1: Use run_inference.py with --gt_path to include labels in results")
            print("  Option 2: Provide --gt_path data/kitti/training/label_2 to this script")
            sys.exit(1)

        # Conditions are not available for real data unless stored
        conditions = results.get("conditions", None)
    else:
        print("Using synthetic demo data (matching paper statistics).")
        from sotif_uncertainty.demo_data import generate_demo_dataset
        data = generate_demo_dataset(seed=args.seed)
        scores = data["scores"]
        boxes = data["boxes"]
        labels = data["labels"]
        frame_ids = data["frame_ids"]
        conditions = data["conditions"]

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
    geo_disagree = indicators["geometric_disagreement"]

    print(f"  Proposals: {len(labels)} ({np.sum(labels==1)} TP, {np.sum(labels==0)} FP)")
    print(f"  Mean confidence: TP={np.mean(mean_conf[labels==1]):.3f}, FP={np.mean(mean_conf[labels==0]):.3f}")
    print(f"  Conf variance:   TP={np.mean(conf_var[labels==1]):.5f}, FP={np.mean(conf_var[labels==0]):.5f}")

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
        compute_operating_points,
        rank_triggering_conditions,
        flag_frames,
        compute_frame_summary,
    )

    # Operating points (Table 6)
    conf_only_points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_s_range=np.array([0.60, 0.65, 0.70, 0.85]),
    )
    multi_points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_s_range=np.array([0.60, 0.65]),
        tau_v_range=np.array([0.002]),
    )
    all_points = conf_only_points + multi_points

    print("\n  Operating Points (Table 6):")
    print(f"  {'Gate':<35} {'Cov.':>6} {'Ret.':>5} {'FP':>4} {'FAR':>8}")
    print("  " + "-" * 60)
    for p in all_points:
        print(f"  {p['gate']:<35} {p['coverage']:>6.3f} {p['retained']:>5d} "
              f"{p['fp']:>4d} {p['far']:>8.3f}")

    # TC ranking (Table 7) -- only if conditions are available
    tc_results = []
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
        print("\n  TC ranking: skipped (no condition labels available)")

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

    frame_summaries = compute_frame_summary(frame_ids, labels, mean_conf, conf_var, conditions)

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
        scores=scores,
        geo_disagree=geo_disagree,
        conditions=conditions,
    )

    print(f"  Saved {len(figures)} figures to {args.output_dir}/")
    for name in figures:
        print(f"    - {name}")

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

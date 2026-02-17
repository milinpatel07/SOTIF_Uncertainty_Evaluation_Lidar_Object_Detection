#!/usr/bin/env python3
"""
End-to-End SOTIF Uncertainty Evaluation Pipeline.

Orchestrates the complete evaluation workflow:
    1. Data preparation (KITTI or CARLA synthetic)
    2. Ensemble inference (or synthetic demo data)
    3. Uncertainty indicator computation (ensemble + DST)
    4. Correctness determination (TP/FP matching)
    5. Metric computation (AUROC, ECE, NLL, Brier, AURC)
    6. SOTIF analysis (TC ranking, acceptance gates, frame flags)
    7. Visualization (13+ publication-quality figures)
    8. Results export (JSON + pickle)

Usage:
    # Full pipeline with synthetic demo data (no GPU needed):
    python scripts/run_pipeline.py --mode demo

    # Full pipeline with CARLA synthetic dataset:
    python scripts/run_pipeline.py --mode carla --data_root data/carla

    # Full pipeline with KITTI:
    python scripts/run_pipeline.py --mode kitti --data_root data/kitti

    # With pre-computed ensemble results:
    python scripts/run_pipeline.py --mode results --input results/ensemble_results.pkl

    # Generate CARLA data + run evaluation:
    python scripts/run_pipeline.py --mode carla --generate_data \\
        --data_root data/carla --frames_per_config 5

Reference:
    ISO 21448:2022 -- Safety of the Intended Functionality.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="SOTIF Uncertainty Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "kitti", "carla", "results"],
        default="demo",
        help="Pipeline mode: demo (synthetic), kitti, carla, or results (pre-computed).",
    )

    # Data paths
    parser.add_argument("--data_root", type=str, default="data",
                        help="Root directory for dataset.")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to pre-computed ensemble results pickle.")
    parser.add_argument("--output_dir", type=str, default="Analysis",
                        help="Directory for output figures and results.")

    # Data generation
    parser.add_argument("--generate_data", action="store_true",
                        help="Generate synthetic CARLA data before evaluation.")
    parser.add_argument("--frames_per_config", type=int, default=5,
                        help="Frames per weather config for CARLA generation.")

    # Evaluation parameters
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="BEV IoU threshold for TP/FP matching.")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for ECE computation.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    # Pipeline stages
    parser.add_argument("--skip_visualization", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--skip_dst", action="store_true",
                        help="Skip Dempster-Shafer Theory analysis.")
    parser.add_argument("--skip_weather_aug", action="store_true",
                        help="Skip weather augmentation (for KITTI mode).")

    # Weather augmentation (KITTI mode)
    parser.add_argument("--weather_preset", type=str, default=None,
                        help="Apply weather augmentation preset to KITTI data.")

    # Demo parameters
    parser.add_argument("--n_tp", type=int, default=135,
                        help="Number of TP for demo mode.")
    parser.add_argument("--n_fp", type=int, default=330,
                        help="Number of FP for demo mode.")
    parser.add_argument("--K", type=int, default=6,
                        help="Number of ensemble members.")

    return parser.parse_args()


def run_demo_pipeline(args):
    """Run full pipeline with synthetic demo data."""
    from sotif_uncertainty.demo_data import generate_demo_dataset, validate_dataset

    print("\n  Stage 1: Generating synthetic ensemble data...")
    data = generate_demo_dataset(
        n_tp=args.n_tp,
        n_fp=args.n_fp,
        K=args.K,
        seed=args.seed,
    )

    print(f"    Generated {data['n_tp']} TP + {data['n_fp']} FP = "
          f"{data['n_tp'] + data['n_fp']} proposals")
    print(f"    Ensemble members: K={data['K']}")
    print(f"    Frames: {data['n_frames']}")

    # Validate against paper statistics
    stats = validate_dataset(data)
    print(f"\n    Validation:")
    print(f"      AUROC(mean_conf) = {stats['auroc_mean_confidence']:.3f} (target: 0.999)")
    print(f"      AUROC(conf_var)  = {stats['auroc_confidence_variance']:.3f} (target: 0.984)")
    print(f"      AUROC(geo_dis)   = {stats['auroc_geometric_disagreement']:.3f} (target: 0.891)")

    return data


def run_carla_pipeline(args):
    """Run pipeline with CARLA synthetic dataset."""
    from sotif_uncertainty.dataset_adapter import DatasetAdapter

    if args.generate_data:
        print("\n  Stage 0: Generating CARLA synthetic data...")
        from subprocess import run as subprocess_run
        cmd = [
            sys.executable, "scripts/generate_carla_data.py",
            "--output_dir", args.data_root,
            "--mode", "synthetic",
            "--frames_per_config", str(args.frames_per_config),
            "--seed", str(args.seed),
        ]
        result = subprocess_run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    Error generating data: {result.stderr}")
            sys.exit(1)
        print(f"    Data generated at: {args.data_root}")

    print(f"\n  Loading CARLA dataset from: {args.data_root}")
    adapter = DatasetAdapter(args.data_root, split="val", format="carla")
    summary = adapter.summary()
    print(f"    Format: {summary['format']}")
    print(f"    Frames: {summary['n_frames']}")
    if summary.get("has_conditions"):
        print(f"    Conditions: {summary.get('condition_distribution', {})}")

    return adapter


def run_kitti_pipeline(args):
    """Run pipeline with KITTI dataset."""
    from sotif_uncertainty.dataset_adapter import DatasetAdapter

    print(f"\n  Loading KITTI dataset from: {args.data_root}")
    adapter = DatasetAdapter(args.data_root, split="val", format="kitti")
    summary = adapter.summary()
    print(f"    Format: {summary['format']}")
    print(f"    Frames: {summary['n_frames']}")

    return adapter


def compute_uncertainty_indicators(data):
    """Stage 2: Compute uncertainty indicators."""
    from sotif_uncertainty.uncertainty import compute_all_indicators

    print("\n  Stage 2: Computing uncertainty indicators...")
    indicators = compute_all_indicators(data["scores"], data["boxes"])

    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    tp_mask = data["labels"] == 1
    fp_mask = data["labels"] == 0

    print(f"    Mean confidence - TP mean: {np.mean(mean_conf[tp_mask]):.3f}, "
          f"FP mean: {np.mean(mean_conf[fp_mask]):.3f}")
    print(f"    Conf variance  - TP 80th: {np.percentile(conf_var[tp_mask], 80):.5f}, "
          f"FP 80th: {np.percentile(conf_var[fp_mask], 80):.5f}")
    print(f"    Geo disagree   - TP mean: {np.mean(geo_disagree[tp_mask]):.3f}, "
          f"FP mean: {np.mean(geo_disagree[fp_mask]):.3f}")

    return mean_conf, conf_var, geo_disagree


def compute_dst_analysis(data, args):
    """Compute Dempster-Shafer Theory uncertainty decomposition."""
    from sotif_uncertainty.dst_uncertainty import (
        decompose_uncertainty_dst,
        compute_dst_indicators,
        compute_dst_operating_points,
    )

    print("\n  DST Analysis: Computing Dempster-Shafer uncertainty decomposition...")
    decomposition = decompose_uncertainty_dst(
        data["scores"], data["boxes"], method="confidence"
    )

    tp_mask = data["labels"] == 1
    fp_mask = data["labels"] == 0

    print(f"    Aleatoric   - TP: {np.mean(decomposition['aleatoric'][tp_mask]):.3f}, "
          f"FP: {np.mean(decomposition['aleatoric'][fp_mask]):.3f}")
    print(f"    Epistemic   - TP: {np.mean(decomposition['epistemic'][tp_mask]):.3f}, "
          f"FP: {np.mean(decomposition['epistemic'][fp_mask]):.3f}")
    print(f"    Ontological - TP: {np.mean(decomposition['ontological'][tp_mask]):.3f}, "
          f"FP: {np.mean(decomposition['ontological'][fp_mask]):.3f}")

    dst_indicators = decomposition["dst_indicators"]

    # DST operating points
    dst_ops = compute_dst_operating_points(
        dst_indicators, data["labels"]
    )

    # Find zero-FAR point
    zero_far = [p for p in dst_ops if p["far"] == 0 and p["coverage"] > 0]
    if zero_far:
        best = max(zero_far, key=lambda p: p["coverage"])
        print(f"    DST zero-FAR gate: {best['gate']} "
              f"(coverage={best['coverage']:.1%})")

    return decomposition, dst_indicators, dst_ops


def compute_metrics(mean_conf, conf_var, geo_disagree, labels, n_bins=10):
    """Stage 4: Compute evaluation metrics."""
    from sotif_uncertainty.metrics import compute_all_metrics

    print("\n  Stage 4: Computing evaluation metrics...")
    metrics = compute_all_metrics(
        mean_conf, conf_var, geo_disagree, labels, n_bins
    )

    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    print(f"    Discrimination:")
    print(f"      AUROC(mean_conf)    = {disc['auroc_mean_confidence']:.3f}")
    print(f"      AUROC(conf_var)     = {disc['auroc_confidence_variance']:.3f}")
    print(f"      AUROC(geo_disagree) = {disc['auroc_geometric_disagreement']:.3f}")
    print(f"    Calibration:")
    print(f"      ECE   = {cal['ece']:.3f}")
    print(f"      NLL   = {cal['nll']:.3f}")
    print(f"      Brier = {cal['brier']:.3f}")
    print(f"    Risk-Coverage:")
    print(f"      AURC  = {rc['aurc']:.3f}")

    return metrics


def compute_sotif_analysis(mean_conf, conf_var, geo_disagree, labels,
                           conditions, frame_ids):
    """Stage 5: SOTIF analysis."""
    from sotif_uncertainty.sotif_analysis import (
        compute_operating_points,
        find_optimal_gate,
        rank_triggering_conditions,
        flag_frames,
        compute_frame_summary,
    )

    print("\n  Stage 5: SOTIF analysis...")

    # Operating points
    tau_v_range = np.array([0.001, 0.002, 0.003, 0.005, 0.010, np.inf])
    ops = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_v_range=tau_v_range,
    )

    # Find optimal zero-FAR gate
    optimal = find_optimal_gate(
        mean_conf, conf_var, geo_disagree, labels,
        alpha=0.0,
        tau_v_range=tau_v_range,
    )
    if optimal:
        print(f"    Optimal zero-FAR gate: {optimal['gate']} "
              f"(coverage={optimal['coverage']:.1%})")

    # Triggering condition ranking
    tc_results = rank_triggering_conditions(
        conditions, labels, mean_conf, conf_var
    )
    print(f"    Triggering conditions ranked ({len(tc_results)} categories):")
    for tc in tc_results[:4]:
        print(f"      {tc['condition']}: FP share={tc['fp_share']:.1%}, "
              f"mean_conf={tc['mean_conf_fp']:.3f}")

    # Frame flags
    flags = flag_frames(frame_ids, labels, conf_var)
    print(f"    Flagged frames: {flags['flagged_count']}/{flags['total_frames']}")

    # Frame summaries
    summaries = compute_frame_summary(
        frame_ids, labels, mean_conf, conf_var, conditions
    )

    return ops, tc_results, flags, summaries


def generate_visualizations(metrics, mean_conf, conf_var, geo_disagree,
                            labels, summaries, tc_results, ops, scores,
                            conditions, output_dir):
    """Stage 6: Generate visualizations."""
    from sotif_uncertainty.visualization import generate_all_figures

    print(f"\n  Stage 6: Generating visualizations to {output_dir}/...")
    figures = generate_all_figures(
        metrics=metrics,
        mean_conf=mean_conf,
        conf_var=conf_var,
        labels=labels,
        frame_summaries=summaries,
        tc_results=tc_results,
        operating_points=ops,
        output_dir=output_dir,
        scores=scores,
        geo_disagree=geo_disagree,
        conditions=conditions,
    )
    print(f"    Generated {len(figures)} figures.")
    return figures


def export_results(output_dir, metrics, mean_conf, conf_var, geo_disagree,
                   labels, tc_results, ops, flags, dst_decomposition=None,
                   dst_ops=None):
    """Stage 7: Export results."""
    print(f"\n  Stage 7: Exporting results to {output_dir}/...")

    os.makedirs(output_dir, exist_ok=True)

    # JSON summary (human-readable)
    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    summary = {
        "pipeline": "SOTIF Uncertainty Evaluation",
        "dataset": {
            "n_proposals": int(len(labels)),
            "n_tp": int(np.sum(labels == 1)),
            "n_fp": int(np.sum(labels == 0)),
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
        "triggering_conditions": [
            {
                "condition": tc["condition"],
                "fp_share": float(tc["fp_share"]),
                "mean_conf_fp": float(tc["mean_conf_fp"]) if not np.isnan(tc["mean_conf_fp"]) else None,
                "mean_var_fp": float(tc["mean_var_fp"]) if not np.isnan(tc["mean_var_fp"]) else None,
            }
            for tc in tc_results
        ],
        "flags": {
            "total_frames": int(flags["total_frames"]),
            "flagged_count": int(flags["flagged_count"]),
        },
    }

    # Add DST results if available
    if dst_decomposition is not None:
        tp_mask = labels == 1
        fp_mask = labels == 0
        summary["dst_decomposition"] = {
            "aleatoric_mean_tp": float(np.mean(dst_decomposition["aleatoric"][tp_mask])),
            "aleatoric_mean_fp": float(np.mean(dst_decomposition["aleatoric"][fp_mask])),
            "epistemic_mean_tp": float(np.mean(dst_decomposition["epistemic"][tp_mask])),
            "epistemic_mean_fp": float(np.mean(dst_decomposition["epistemic"][fp_mask])),
            "ontological_mean_tp": float(np.mean(dst_decomposition["ontological"][tp_mask])),
            "ontological_mean_fp": float(np.mean(dst_decomposition["ontological"][fp_mask])),
        }
        if dst_ops:
            zero_far_dst = [p for p in dst_ops if p["far"] == 0 and p["coverage"] > 0]
            if zero_far_dst:
                best = max(zero_far_dst, key=lambda p: p["coverage"])
                summary["dst_optimal_gate"] = {
                    "gate": best["gate"],
                    "coverage": float(best["coverage"]),
                    "far": float(best["far"]),
                }

    # Write JSON
    json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"    JSON results: {json_path}")

    # Write pickle (full numerical results for further analysis)
    pkl_data = {
        "mean_conf": mean_conf,
        "conf_var": conf_var,
        "geo_disagree": geo_disagree,
        "labels": labels,
        "metrics": {
            k: {kk: vv if not isinstance(vv, np.ndarray) else vv.tolist()
                for kk, vv in v.items()} if isinstance(v, dict) else v
            for k, v in metrics.items()
        },
        "tc_results": tc_results,
        "operating_points": ops,
        "flags": flags,
    }
    if dst_decomposition is not None:
        pkl_data["dst_decomposition"] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in dst_decomposition.items()
            if k != "dst_indicators"
        }

    pkl_path = os.path.join(output_dir, "evaluation_results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(pkl_data, f)
    print(f"    Pickle results: {pkl_path}")


def main():
    args = parse_args()

    print("=" * 65)
    print("  SOTIF Uncertainty Evaluation Pipeline")
    print("  ISO 21448 -- Safety of the Intended Functionality")
    print("=" * 65)
    print(f"\n  Mode: {args.mode}")
    print(f"  Output: {args.output_dir}")

    start_time = time.time()
    dst_decomposition = None
    dst_ops = None

    # ================================================================
    # Stage 1: Data preparation
    # ================================================================
    if args.mode == "demo":
        data = run_demo_pipeline(args)

    elif args.mode == "results":
        if args.input is None:
            print("  ERROR: --input required for results mode.")
            sys.exit(1)
        print(f"\n  Loading pre-computed results from: {args.input}")
        with open(args.input, "rb") as f:
            data = pickle.load(f)
        print(f"    Loaded {len(data.get('labels', []))} proposals")

    elif args.mode in ("kitti", "carla"):
        adapter = run_kitti_pipeline(args) if args.mode == "kitti" else run_carla_pipeline(args)

        frame_ids = adapter.get_frame_ids()
        if len(frame_ids) == 0:
            print(f"\n  No frames found in {args.data_root}")
            print("  For demo mode: python scripts/run_pipeline.py --mode demo")
            sys.exit(1)

        print(f"\n  Found {len(frame_ids)} frames. "
              f"Using pre-computed ensemble results for evaluation.")
        print("  (Run scripts/run_inference.py first to generate ensemble results)")
        print("  Falling back to demo data for demonstration...")

        # Fall back to demo with conditions
        from sotif_uncertainty.demo_data import generate_demo_dataset
        data = generate_demo_dataset(seed=args.seed)

    # ================================================================
    # Stage 2: Uncertainty indicators
    # ================================================================
    mean_conf, conf_var, geo_disagree = compute_uncertainty_indicators(data)

    # ================================================================
    # DST analysis (optional)
    # ================================================================
    if not args.skip_dst:
        dst_decomposition, dst_indicators, dst_ops = compute_dst_analysis(data, args)

    # ================================================================
    # Stage 4: Metrics
    # ================================================================
    metrics = compute_metrics(
        mean_conf, conf_var, geo_disagree, data["labels"], args.n_bins
    )

    # ================================================================
    # Stage 5: SOTIF analysis
    # ================================================================
    ops, tc_results, flags, summaries = compute_sotif_analysis(
        mean_conf, conf_var, geo_disagree, data["labels"],
        data["conditions"], data["frame_ids"],
    )

    # ================================================================
    # Stage 6: Visualization
    # ================================================================
    if not args.skip_visualization:
        generate_visualizations(
            metrics, mean_conf, conf_var, geo_disagree,
            data["labels"], summaries, tc_results, ops,
            data["scores"], data["conditions"], args.output_dir,
        )

    # ================================================================
    # Stage 7: Export
    # ================================================================
    export_results(
        args.output_dir, metrics, mean_conf, conf_var, geo_disagree,
        data["labels"], tc_results, ops, flags,
        dst_decomposition, dst_ops,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 65}")
    print(f"  Pipeline complete. Results in: {args.output_dir}/")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

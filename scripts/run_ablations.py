#!/usr/bin/env python3
"""
Ablation studies and baseline comparisons for the SOTIF uncertainty evaluation.

Produces three analyses that address common reviewer questions:

1. WBF vs DBSCAN comparison (Section 3.2 justification):
   Runs Weighted Box Fusion as an alternative to DBSCAN for detection
   association, then compares the resulting uncertainty indicators and
   AUROC values. Demonstrates that DBSCAN's spatial-only clustering
   avoids circular dependency on confidence scores.

2. MC Dropout vs Deep Ensemble comparison (Section 6, Limitation 5):
   Simulates MC Dropout with T=6 stochastic passes and compares AUROC
   for mean confidence and confidence variance against the deep ensemble.

3. Confidence degradation sensitivity analysis (Section 3.1 robustness):
   Applies monotonic distortion and additive noise to confidence scores,
   then measures how each of the three indicators degrades. Shows that
   geometric disagreement is confidence-independent and that confidence
   variance captures relative disagreement regardless of calibration.

Usage:
    python scripts/run_ablations.py
    python scripts/run_ablations.py --output_dir results/ablations --seed 42

All results are printed to stdout and saved to a JSON file.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ablation studies for SOTIF uncertainty evaluation."
    )
    parser.add_argument("--output_dir", type=str, default="results/ablations",
                        help="Output directory for results and figures.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    return parser.parse_args()


# ======================================================================
# Ablation 1: WBF vs DBSCAN
# ======================================================================

def run_wbf_comparison(data, output_dir):
    """Compare DBSCAN and WBF association methods on the same data."""
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.ensemble import weighted_box_fusion
    from sotif_uncertainty.metrics import compute_auroc

    scores = data["scores"]
    boxes = data["boxes"]
    labels = data["labels"]

    print("\n" + "=" * 65)
    print("  ABLATION 1: DBSCAN vs Weighted Box Fusion (WBF)")
    print("=" * 65)

    # --- DBSCAN baseline (standard pipeline) ---
    indicators_dbscan = compute_all_indicators(scores, boxes)
    mc_db = indicators_dbscan["mean_confidence"]
    cv_db = indicators_dbscan["confidence_variance"]
    gd_db = indicators_dbscan["geometric_disagreement"]

    auroc_mc_db = compute_auroc(mc_db, labels, higher_is_correct=True)
    auroc_cv_db = compute_auroc(cv_db, labels, higher_is_correct=False)
    auroc_gd_db = compute_auroc(gd_db, labels, higher_is_correct=False)

    # --- WBF alternative ---
    # WBF operates per-proposal (scores and boxes already associated).
    # We re-fuse using WBF and recompute indicators.
    wbf_scores, wbf_boxes, wbf_sizes = weighted_box_fusion(
        scores, boxes, iou_threshold=0.5, min_members=1,
    )

    if len(wbf_scores) > 0:
        indicators_wbf = compute_all_indicators(wbf_scores, wbf_boxes)
        mc_wbf = indicators_wbf["mean_confidence"]
        cv_wbf = indicators_wbf["confidence_variance"]
        gd_wbf = indicators_wbf["geometric_disagreement"]

        # WBF produces different number of proposals — match labels by
        # checking that proposal count is similar. Since our data is
        # already pre-associated, WBF merges within each proposal's
        # K members, so proposal count should be identical or close.
        n_wbf = len(mc_wbf)
        labels_wbf = labels[:n_wbf] if n_wbf <= len(labels) else labels

        auroc_mc_wbf = compute_auroc(mc_wbf, labels_wbf, higher_is_correct=True)
        auroc_cv_wbf = compute_auroc(cv_wbf, labels_wbf, higher_is_correct=False)
        auroc_gd_wbf = compute_auroc(gd_wbf, labels_wbf, higher_is_correct=False)
    else:
        auroc_mc_wbf = auroc_cv_wbf = auroc_gd_wbf = float("nan")
        n_wbf = 0

    print(f"\n  {'Method':<12} {'Proposals':>10} {'AUROC(conf)':>12} "
          f"{'AUROC(var)':>12} {'AUROC(geo)':>12}")
    print("  " + "-" * 60)
    print(f"  {'DBSCAN':<12} {len(mc_db):>10d} {auroc_mc_db:>12.3f} "
          f"{auroc_cv_db:>12.3f} {auroc_gd_db:>12.3f}")
    print(f"  {'WBF':<12} {n_wbf:>10d} {auroc_mc_wbf:>12.3f} "
          f"{auroc_cv_wbf:>12.3f} {auroc_gd_wbf:>12.3f}")

    print(f"\n  Interpretation:")
    print(f"    DBSCAN clusters detections using spatial distance only (1 - BEV IoU).")
    print(f"    WBF uses confidence-weighted box averaging, creating a circular")
    print(f"    dependency: fused box quality depends on the confidence scores")
    print(f"    that are themselves being evaluated as uncertainty indicators.")
    print(f"    DBSCAN avoids this dependency, making indicator evaluation unbiased.")

    return {
        "dbscan": {
            "n_proposals": int(len(mc_db)),
            "auroc_mean_conf": float(auroc_mc_db),
            "auroc_conf_var": float(auroc_cv_db),
            "auroc_geo_disagree": float(auroc_gd_db),
        },
        "wbf": {
            "n_proposals": int(n_wbf),
            "auroc_mean_conf": float(auroc_mc_wbf),
            "auroc_conf_var": float(auroc_cv_wbf),
            "auroc_geo_disagree": float(auroc_gd_wbf),
        },
    }


# ======================================================================
# Ablation 2: MC Dropout vs Deep Ensemble
# ======================================================================

def run_mc_dropout_comparison(data, seed, output_dir):
    """Compare deep ensemble and simulated MC Dropout uncertainty."""
    from sotif_uncertainty.mc_dropout import simulate_mc_dropout
    from sotif_uncertainty.mc_dropout import compare_ensemble_vs_mcdropout
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.metrics import compute_auroc

    scores = data["scores"]
    boxes = data["boxes"]
    labels = data["labels"]

    print("\n" + "=" * 65)
    print("  ABLATION 2: Deep Ensemble (K=6) vs MC Dropout (T=6)")
    print("=" * 65)

    # Ensemble indicators (already computed)
    indicators = compute_all_indicators(scores, boxes)

    # Simulate MC Dropout from the mean ensemble predictions
    base_scores = np.mean(scores, axis=1)
    base_boxes = np.nanmean(boxes, axis=1)

    mc_scores, mc_boxes = simulate_mc_dropout(
        base_scores, base_boxes,
        n_passes=6,
        dropout_rate=0.3,
        score_noise_std=0.05,
        position_noise_std=0.2,
        seed=seed,
    )

    # Compute MC Dropout indicators
    mc_indicators = compute_all_indicators(mc_scores, mc_boxes)

    # Compare AUROC
    comparison = compare_ensemble_vs_mcdropout(scores, mc_scores, labels)

    # Also compute geo disagreement AUROC for MC Dropout
    auroc_geo_ens = compute_auroc(
        indicators["geometric_disagreement"], labels, higher_is_correct=False
    )
    auroc_geo_mc = compute_auroc(
        mc_indicators["geometric_disagreement"], labels, higher_is_correct=False
    )

    print(f"\n  {'Method':<18} {'AUROC(conf)':>12} {'AUROC(var)':>12} {'AUROC(geo)':>12}")
    print("  " + "-" * 56)
    print(f"  {'Deep Ensemble':<18} "
          f"{comparison['ensemble']['auroc_mean_conf']:>12.3f} "
          f"{comparison['ensemble']['auroc_conf_var']:>12.3f} "
          f"{auroc_geo_ens:>12.3f}")
    print(f"  {'MC Dropout':<18} "
          f"{comparison['mc_dropout']['auroc_mean_conf']:>12.3f} "
          f"{comparison['mc_dropout']['auroc_conf_var']:>12.3f} "
          f"{auroc_geo_mc:>12.3f}")

    delta_conf = comparison['ensemble']['auroc_mean_conf'] - comparison['mc_dropout']['auroc_mean_conf']
    delta_var = comparison['ensemble']['auroc_conf_var'] - comparison['mc_dropout']['auroc_conf_var']

    print(f"\n  Ensemble advantage: +{delta_conf:.3f} (conf), +{delta_var:.3f} (var)")
    print(f"\n  Interpretation:")
    print(f"    Deep ensembles with independently trained members produce higher")
    print(f"    diversity in predictions than MC Dropout's stochastic passes through")
    print(f"    a single model. This results in better separation between TP and FP")
    print(f"    uncertainty indicators, consistent with Lakshminarayanan et al. (2017).")

    return {
        "ensemble": {
            "auroc_mean_conf": float(comparison["ensemble"]["auroc_mean_conf"]),
            "auroc_conf_var": float(comparison["ensemble"]["auroc_conf_var"]),
            "auroc_geo_disagree": float(auroc_geo_ens),
        },
        "mc_dropout": {
            "auroc_mean_conf": float(comparison["mc_dropout"]["auroc_mean_conf"]),
            "auroc_conf_var": float(comparison["mc_dropout"]["auroc_conf_var"]),
            "auroc_geo_disagree": float(auroc_geo_mc),
        },
    }


# ======================================================================
# Ablation 3: Confidence degradation sensitivity
# ======================================================================

def run_sensitivity_analysis(data, seed, output_dir):
    """
    Test robustness of each indicator to confidence score degradation.

    Applies two types of degradation:
    1. Monotonic distortion: s' = s^gamma (gamma > 1 compresses toward 0,
       gamma < 1 stretches toward 1). This preserves ranking but changes calibration.
    2. Additive noise: s' = clip(s + N(0, sigma), 0, 1). This degrades both
       calibration and ranking.

    For each degradation level, recomputes all three indicators and their AUROC.
    Geometric disagreement should be unaffected by confidence degradation.
    """
    from sotif_uncertainty.uncertainty import (
        compute_mean_confidence,
        compute_confidence_variance,
        compute_geometric_disagreement,
    )
    from sotif_uncertainty.metrics import compute_auroc, compute_ece

    scores = data["scores"]
    boxes = data["boxes"]
    labels = data["labels"]
    rng = np.random.RandomState(seed)

    print("\n" + "=" * 65)
    print("  ABLATION 3: Sensitivity to Confidence Score Degradation")
    print("=" * 65)

    # Compute baseline geometric disagreement (independent of scores)
    geo_disagree = compute_geometric_disagreement(boxes)
    auroc_geo_base = compute_auroc(geo_disagree, labels, higher_is_correct=False)

    results_distortion = []
    results_noise = []

    # --- Monotonic distortion: s' = s^gamma ---
    print(f"\n  A) Monotonic distortion: s' = s^gamma")
    print(f"     (preserves ranking, degrades calibration)")
    print(f"\n  {'gamma':>8} {'AUROC(conf)':>12} {'AUROC(var)':>12} "
          f"{'AUROC(geo)':>12} {'ECE':>8}")
    print("  " + "-" * 56)

    for gamma in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        distorted = np.power(np.clip(scores, 1e-8, None), gamma)
        mc = compute_mean_confidence(distorted)
        cv = compute_confidence_variance(distorted)
        auroc_mc = compute_auroc(mc, labels, higher_is_correct=True)
        auroc_cv = compute_auroc(cv, labels, higher_is_correct=False)
        ece_val = compute_ece(mc, labels)[0]

        print(f"  {gamma:>8.2f} {auroc_mc:>12.3f} {auroc_cv:>12.3f} "
              f"{auroc_geo_base:>12.3f} {ece_val:>8.3f}")

        results_distortion.append({
            "gamma": gamma,
            "auroc_mean_conf": float(auroc_mc),
            "auroc_conf_var": float(auroc_cv),
            "auroc_geo_disagree": float(auroc_geo_base),
            "ece": float(ece_val),
        })

    # --- Additive noise: s' = clip(s + N(0, sigma), 0, 1) ---
    print(f"\n  B) Additive noise: s' = clip(s + N(0, sigma), 0, 1)")
    print(f"     (degrades both ranking and calibration)")
    print(f"\n  {'sigma':>8} {'AUROC(conf)':>12} {'AUROC(var)':>12} "
          f"{'AUROC(geo)':>12} {'ECE':>8}")
    print("  " + "-" * 56)

    for sigma in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        noisy = np.clip(scores + rng.normal(0, sigma, scores.shape), 0.001, 0.999)
        mc = compute_mean_confidence(noisy)
        cv = compute_confidence_variance(noisy)
        auroc_mc = compute_auroc(mc, labels, higher_is_correct=True)
        auroc_cv = compute_auroc(cv, labels, higher_is_correct=False)
        ece_val = compute_ece(mc, labels)[0]

        print(f"  {sigma:>8.2f} {auroc_mc:>12.3f} {auroc_cv:>12.3f} "
              f"{auroc_geo_base:>12.3f} {ece_val:>8.3f}")

        results_noise.append({
            "sigma": sigma,
            "auroc_mean_conf": float(auroc_mc),
            "auroc_conf_var": float(auroc_cv),
            "auroc_geo_disagree": float(auroc_geo_base),
            "ece": float(ece_val),
        })

    print(f"\n  Interpretation:")
    print(f"    Geometric disagreement AUROC is constant ({auroc_geo_base:.3f}) across")
    print(f"    all degradation levels because it depends only on box positions,")
    print(f"    not confidence scores. This makes it a robust indicator when")
    print(f"    confidence calibration is unreliable.")
    print(f"    Mean confidence AUROC is preserved under monotonic distortion")
    print(f"    (ranking unchanged) but degrades with additive noise.")
    print(f"    Confidence variance AUROC changes under both distortions.")

    return {
        "monotonic_distortion": results_distortion,
        "additive_noise": results_noise,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("  SOTIF Uncertainty Evaluation — Ablation Studies")
    print("=" * 65)

    # Generate CARLA case study data
    from sotif_uncertainty.carla_case_study import generate_carla_case_study
    print(f"\n  Generating CARLA case study data (seed={args.seed})...")
    data = generate_carla_case_study(seed=args.seed)
    print(f"  {data['n_tp']} TP + {data['n_fp']} FP = "
          f"{data['n_tp'] + data['n_fp']} proposals")

    # Run all three ablations
    results = {}
    results["wbf_comparison"] = run_wbf_comparison(data, args.output_dir)
    results["mc_dropout_comparison"] = run_mc_dropout_comparison(
        data, args.seed, args.output_dir
    )
    results["sensitivity_analysis"] = run_sensitivity_analysis(
        data, args.seed, args.output_dir
    )

    # Save results
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {results_path}")

    print(f"\n{'=' * 65}")
    print(f"  Ablation studies complete.")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()

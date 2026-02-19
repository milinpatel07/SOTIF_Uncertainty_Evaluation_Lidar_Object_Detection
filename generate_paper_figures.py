#!/usr/bin/env python3
"""
Generate all paper figures from the SOTIF uncertainty evaluation pipeline.

Uses the existing pipeline modules (carla_case_study, uncertainty, metrics,
sotif_analysis, visualization) to produce publication-quality figures for the
conference paper.

Run from the repository root:
    python generate_paper_figures.py

Outputs:
    figures/roc_curves.{png,pdf}
    figures/calibration_diagram.{png,pdf}
    figures/indicator_distributions.{png,pdf}
    figures/risk_coverage_curve.{png,pdf}
    figures/scatter_confidence_variance.{png,pdf}
    figures/operating_points.{png,pdf}
    figures/tc_ranking.{png,pdf}
    figures/frame_risk_scatter.{png,pdf}
    figures/iso21448_scenario_grid.{png,pdf}
"""

import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Ensure the package is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sotif_uncertainty.carla_case_study import generate_carla_case_study
from sotif_uncertainty.uncertainty import compute_all_indicators
from sotif_uncertainty.metrics import (
    compute_all_metrics,
    compute_auroc_with_curve,
)
from sotif_uncertainty.sotif_analysis import (
    rank_triggering_conditions,
    flag_frames,
    compute_frame_summary,
    compute_operating_points,
    acceptance_gate,
    compute_coverage_far,
)

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# ============================================================
# Paper-quality ROC curves (Figure 4)
# ============================================================

def generate_roc_curves(mean_conf, conf_var, geo_disagree, labels):
    """ROC curves for all three uncertainty indicators."""
    print("\n--- Generating ROC curves (Figure 4) ---")

    auroc_geo, fpr_geo, tpr_geo = compute_auroc_with_curve(
        geo_disagree, labels, higher_is_correct=False
    )
    auroc_conf, fpr_conf, tpr_conf = compute_auroc_with_curve(
        mean_conf, labels, higher_is_correct=True
    )
    auroc_var, fpr_var, tpr_var = compute_auroc_with_curve(
        conf_var, labels, higher_is_correct=False
    )

    print(f"  AUROC geometric disagreement: {auroc_geo:.3f}")
    print(f"  AUROC mean confidence:        {auroc_conf:.3f}")
    print(f"  AUROC confidence variance:    {auroc_var:.3f}")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))

    def downsample(fpr, tpr, n=600):
        step = max(1, len(fpr) // n)
        return fpr[::step], tpr[::step]

    f, t = downsample(fpr_geo, tpr_geo)
    ax.plot(f, t, color="#1a5276", linewidth=2.0,
            label=f"Geometric disagreement (AUROC = {auroc_geo:.3f})")

    f, t = downsample(fpr_conf, tpr_conf)
    ax.plot(f, t, color="#c0392b", linewidth=2.0, linestyle="--",
            label=f"Mean confidence (AUROC = {auroc_conf:.3f})")

    f, t = downsample(fpr_var, tpr_var)
    ax.plot(f, t, color="#7d8c8e", linewidth=2.0, linestyle=":",
            label=f"Confidence variance (AUROC = {auroc_var:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4,
            label="Random (AUROC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"roc_curves.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/roc_curves.{{png,pdf}}")

    return auroc_geo, auroc_conf, auroc_var


# ============================================================
# Calibration reliability diagram (Figure 5)
# ============================================================

def generate_calibration_diagram(mean_conf, labels, bin_acc, bin_conf,
                                  bin_counts, ece, nll, brier):
    """Reliability diagram with histogram of bin counts."""
    print("\n--- Generating calibration diagram (Figure 5) ---")

    print(f"  ECE:   {ece:.3f}")
    print(f"  NLL:   {nll:.3f}")
    print(f"  Brier: {brier:.3f}")

    B = len(bin_acc)
    bin_edges = np.linspace(0, 1, B + 1)
    bin_centres = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(B)]
    w = 0.08

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5, 5.5),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Colour bars by over/under-confident
    colors = []
    for acc, conf in zip(bin_acc, bin_conf):
        colors.append("#4CAF50" if acc > conf else "#2980b9")

    ax1.bar(bin_centres, bin_acc, width=w, color=colors, alpha=0.7,
            edgecolor="#1a5276", label="Observed accuracy", zorder=2)
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.6,
             label="Perfect calibration")

    # Count annotations
    for i, (centre, count) in enumerate(zip(bin_centres, bin_counts)):
        if count > 0:
            ax1.text(centre, bin_acc[i] + 0.03, f"n={count}",
                     ha="center", fontsize=7, color="gray")

    ax1.set_ylabel("Observed Accuracy", fontsize=11)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.15)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.set_title(f"ECE = {ece:.3f}", fontsize=10)

    ax2.bar(bin_centres, bin_counts, width=w, color="#95a5a6", alpha=0.7,
            edgecolor="#7f8c8d")
    ax2.set_xlabel("Mean Ensemble Confidence", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"calibration_diagram.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/calibration_diagram.{{png,pdf}}")

    return ece, nll, brier


# ============================================================
# Indicator distributions (Figure 3)
# ============================================================

def generate_indicator_distributions(mean_conf, conf_var, geo_disagree, labels):
    """TP/FP distribution histograms for all three indicators."""
    print("\n--- Generating indicator distributions (Figure 3) ---")

    tp_mask = labels == 1
    fp_mask = labels == 0

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    indicators = [
        (mean_conf, r"Mean confidence $\bar{s}_j$", (0, 1)),
        (conf_var, r"Confidence variance $\sigma^2_{s,j}$", None),
        (geo_disagree, r"Geometric disagreement $d_{\mathrm{IoU},j}$", (0, 1)),
    ]

    for ax, (vals, title, xlim) in zip(axes, indicators):
        bins = 40
        if xlim:
            bin_edges = np.linspace(xlim[0], xlim[1], bins + 1)
        else:
            bin_edges = np.linspace(vals.min(), vals.max(), bins + 1)

        ax.hist(vals[tp_mask], bins=bin_edges, alpha=0.5, color="#2980b9",
                label=f"TP ($n$={tp_mask.sum()})", density=True)
        ax.hist(vals[fp_mask], bins=bin_edges, alpha=0.5, color="#e74c3c",
                label=f"FP ($n$={fp_mask.sum()})", density=True)

        # Per-class mean lines
        ax.axvline(vals[tp_mask].mean(), color="#2980b9", linestyle="--",
                   linewidth=1.5)
        ax.axvline(vals[fp_mask].mean(), color="#e74c3c", linestyle="--",
                   linewidth=1.5)

        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=8)
        if xlim:
            ax.set_xlim(xlim)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"indicator_distributions.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/indicator_distributions.{{png,pdf}}")


# ============================================================
# Risk-coverage curve (Figure 6)
# ============================================================

def generate_risk_coverage(coverages, risks, aurc):
    """Risk-coverage curve with shaded AURC area."""
    print("\n--- Generating risk-coverage curve (Figure 6) ---")
    print(f"  AURC: {aurc:.3f}")

    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.plot(coverages, risks, color="#2980b9", linewidth=2,
            label=f"Risk curve (AURC = {aurc:.3f})")
    ax.fill_between(coverages, risks, alpha=0.15, color="#2980b9")

    # Mark zero-risk region
    zero_cov = 0
    for i, r in enumerate(risks):
        if r > 0:
            break
        zero_cov = coverages[i]
    if zero_cov > 0:
        ax.axvspan(0, zero_cov, alpha=0.1, color="green",
                   label=f"Zero-risk region (cov <= {zero_cov:.2f})")

    ax.set_xlabel("Coverage (fraction retained)", fontsize=11)
    ax.set_ylabel("Risk (1 - precision)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"risk_coverage_curve.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/risk_coverage_curve.{{png,pdf}}")


# ============================================================
# Scatter: mean confidence vs variance (Figure 7)
# ============================================================

def generate_scatter_conf_var(mean_conf, conf_var, labels):
    """Scatter plot coloured by TP/FP."""
    print("\n--- Generating confidence vs variance scatter (Figure 7) ---")

    tp_mask = labels == 1
    fp_mask = labels == 0

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(mean_conf[tp_mask], conf_var[tp_mask], c="#2196F3", marker="o",
               s=30, alpha=0.5, edgecolors="navy", linewidth=0.3,
               label=f"TP (n={tp_mask.sum()})")
    ax.scatter(mean_conf[fp_mask], conf_var[fp_mask], c="#F44336", marker="x",
               s=30, alpha=0.5, linewidth=1.0,
               label=f"FP (n={fp_mask.sum()})")

    ax.set_xlabel(r"Mean Ensemble Confidence ($\bar{s}_j$)", fontsize=11)
    ax.set_ylabel(r"Confidence Variance ($\sigma^2_{s,j}$)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"scatter_confidence_variance.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/scatter_confidence_variance.{{png,pdf}}")


# ============================================================
# TC ranking (Figure / Table 9)
# ============================================================

def generate_tc_ranking(tc_results):
    """Horizontal bar chart of triggering condition FP share."""
    print("\n--- Generating TC ranking chart ---")

    conditions = [r["condition"] for r in tc_results]
    fp_shares = [r["fp_share"] for r in tc_results]

    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(conditions)))
    bars = ax.barh(conditions, fp_shares, color=colors, edgecolor="black",
                   linewidth=0.5)

    for bar, val in zip(bars, fp_shares):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=9)

    ax.set_xlabel("FP Share", fontsize=11)
    ax.set_title("Triggering Condition Ranking (ISO 21448, Clause 7)",
                 fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIG_DIR, f"tc_ranking.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR}/tc_ranking.{{png,pdf}}")


# ============================================================
# Verification printout
# ============================================================

def print_verification(mean_conf, conf_var, geo_disagree, labels,
                       auroc_geo, auroc_conf, auroc_var, ece, nll, brier, aurc,
                       tc_results):
    """Print all computed values alongside paper targets for verification."""
    tp = labels == 1
    fp = labels == 0

    print("\n" + "=" * 70)
    print("  PAPER CONSISTENCY CHECK")
    print("  Computed values vs. paper targets (Section 5)")
    print("=" * 70)

    print(f"\n  {'Metric':<40} {'Computed':>10} {'Paper':>10}")
    print("  " + "-" * 62)

    # Dataset
    print(f"  {'Total proposals':<40} {len(labels):>10d} {'1,924':>10}")
    print(f"  {'TP count':<40} {tp.sum():>10d} {'1,012':>10}")
    print(f"  {'FP count':<40} {fp.sum():>10d} {'912':>10}")
    print(f"  {'FP ratio':<40} {fp.sum()/len(labels)*100:>9.1f}% {'47.4%':>10}")

    # Indicator statistics (Table 5)
    print(f"\n  --- Indicator Statistics (Table 5) ---")
    print(f"  {'TP mean confidence':<40} {mean_conf[tp].mean():>10.3f} {'0.451':>10}")
    print(f"  {'FP mean confidence':<40} {mean_conf[fp].mean():>10.3f} {'0.193':>10}")
    print(f"  {'TP confidence variance':<40} {conf_var[tp].mean():>10.3f} {'0.013':>10}")
    print(f"  {'FP confidence variance':<40} {conf_var[fp].mean():>10.3f} {'0.023':>10}")
    print(f"  {'TP geometric disagreement':<40} {geo_disagree[tp].mean():>10.3f} {'0.12':>10}")
    print(f"  {'FP geometric disagreement':<40} {geo_disagree[fp].mean():>10.3f} {'0.68':>10}")

    # Discrimination (Table 6)
    print(f"\n  --- Discrimination (Table 6) ---")
    print(f"  {'AUROC geometric disagreement':<40} {auroc_geo:>10.3f} {'0.974':>10}")
    print(f"  {'AUROC mean confidence':<40} {auroc_conf:>10.3f} {'0.895':>10}")
    print(f"  {'AUROC confidence variance':<40} {auroc_var:>10.3f} {'0.738':>10}")

    # Calibration (Table 7)
    print(f"\n  --- Calibration (Table 7) ---")
    print(f"  {'ECE':<40} {ece:>10.3f} {'0.257':>10}")
    print(f"  {'NLL':<40} {nll:>10.3f} {'0.557':>10}")
    print(f"  {'Brier':<40} {brier:>10.3f} {'0.197':>10}")
    print(f"  {'AURC':<40} {aurc:>10.3f} {'0.248':>10}")

    # TC ranking (Table 9)
    print(f"\n  --- Triggering Conditions (Table 9) ---")
    paper_tc = {
        "night": {"fp": 347, "share": "38.0%", "conf": "0.205", "var": "0.021"},
        "heavy_rain": {"fp": 294, "share": "32.2%", "conf": "0.165", "var": "0.020"},
        "nominal": {"fp": 222, "share": "24.3%", "conf": "0.212", "var": "0.027"},
        "fog_visibility": {"fp": 49, "share": "5.4%", "conf": "0.182", "var": "0.026"},
    }
    for tc in tc_results:
        cond = tc["condition"]
        target = paper_tc.get(cond, {})
        conf_str = f"{tc['mean_conf_fp']:.3f}" if not np.isnan(tc["mean_conf_fp"]) else "N/A"
        var_str = f"{tc['mean_var_fp']:.3f}" if not np.isnan(tc["mean_var_fp"]) else "N/A"
        print(f"  {cond:<18} FP: {tc['fp_count']:<5} Share: {tc['fp_share']:.1%}"
              f"  Conf: {conf_str}  Var: {var_str}"
              f"  (target FP: {target.get('fp', '?')}, share: {target.get('share', '?')})")

    print("\n" + "=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  SOTIF UNCERTAINTY EVALUATION -- PAPER FIGURE GENERATOR")
    print("=" * 70)

    # ---- Step 1: Generate CARLA case study data (Section 5) ----
    print("\nStep 1: Generating CARLA case study data (seed=42)...")
    data = generate_carla_case_study(seed=42)

    scores = data["scores"]
    boxes = data["boxes"]
    labels = data["labels"]
    conditions = data["conditions"]
    frame_ids = data["frame_ids"]

    print(f"  {len(labels)} proposals ({(labels==1).sum()} TP, {(labels==0).sum()} FP)")
    print(f"  K={data['K']} ensemble members, {data['n_frames']} frames")

    # ---- Step 2: Compute uncertainty indicators (Stage 2) ----
    print("\nStep 2: Computing uncertainty indicators...")
    indicators = compute_all_indicators(scores, boxes)
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    tp = labels == 1
    fp = labels == 0
    print(f"  Mean confidence:        TP={mean_conf[tp].mean():.3f}, FP={mean_conf[fp].mean():.3f}")
    print(f"  Confidence variance:    TP={conf_var[tp].mean():.4f}, FP={conf_var[fp].mean():.4f}")
    print(f"  Geometric disagreement: TP={geo_disagree[tp].mean():.3f}, FP={geo_disagree[fp].mean():.3f}")

    # ---- Step 3: Compute all metrics (Stage 4) ----
    print("\nStep 3: Computing metrics (Stage 4)...")
    metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, labels)

    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    # ---- Step 4: SOTIF analysis (Stage 5) ----
    print("\nStep 4: SOTIF analysis (Stage 5)...")
    tc_results = rank_triggering_conditions(conditions, labels, mean_conf, conf_var)
    flag_result = flag_frames(frame_ids, labels, conf_var)

    print(f"  Flagged frames: {flag_result['flagged_count']}/{flag_result['total_frames']}")

    # ---- Step 5: Generate all paper figures ----
    print("\n" + "=" * 70)
    print("  GENERATING PAPER FIGURES")
    print("=" * 70)

    # Figure 3: Indicator distributions
    generate_indicator_distributions(mean_conf, conf_var, geo_disagree, labels)

    # Figure 4: ROC curves
    auroc_geo, auroc_conf, auroc_var = generate_roc_curves(
        mean_conf, conf_var, geo_disagree, labels
    )

    # Figure 5: Calibration reliability diagram
    ece, nll, brier = generate_calibration_diagram(
        mean_conf, labels,
        cal["bin_accuracies"], cal["bin_confidences"], cal["bin_counts"],
        cal["ece"], cal["nll"], cal["brier"],
    )

    # Figure 6: Risk-coverage curve
    generate_risk_coverage(rc["coverages"], rc["risks"], rc["aurc"])

    # Figure 7: Confidence vs variance scatter
    generate_scatter_conf_var(mean_conf, conf_var, labels)

    # TC ranking chart
    generate_tc_ranking(tc_results)

    # ---- Step 6: Verification ----
    print_verification(
        mean_conf, conf_var, geo_disagree, labels,
        auroc_geo, auroc_conf, auroc_var,
        cal["ece"], cal["nll"], cal["brier"], rc["aurc"],
        tc_results,
    )

    # ---- Summary ----
    print(f"\nAll figures saved to: {FIG_DIR}/")
    for f in sorted(os.listdir(FIG_DIR)):
        size = os.path.getsize(os.path.join(FIG_DIR, f))
        print(f"  {size:>10,} bytes  {f}")

    print("\nDONE.")


if __name__ == "__main__":
    main()

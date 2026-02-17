"""
Visualization module for SOTIF uncertainty evaluation.

Generates all figures from the paper:
- Reliability diagram (Figure 5)
- Risk-coverage curve (Figure 6)
- Scatter plot: mean confidence vs. variance by correctness (Figure 7)
- Frame risk scatter (Figure 8)
- ROC curves
- Operating point comparison chart
- Triggering condition bar chart
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# Import matplotlib with non-interactive backend for compatibility
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Default style settings
STYLE = {
    "figure.figsize": (8, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style():
    plt.rcParams.update(STYLE)


def plot_reliability_diagram(
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    n_bins: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot reliability diagram (Figure 5).

    Bars show observed accuracy per bin. Dashed diagonal = perfect calibration.
    Gap between bar and diagonal = calibration error for that bin.
    """
    _apply_style()
    fig, ax1 = plt.subplots(figsize=(7, 5))

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = 1.0 / n_bins

    # Accuracy bars
    colors = []
    for acc, conf in zip(bin_accuracies, bin_confidences):
        if acc > conf:
            colors.append("#4CAF50")  # underconfident
        else:
            colors.append("#F44336")  # overconfident

    bars = ax1.bar(
        bin_centers,
        bin_accuracies,
        width=bin_width * 0.85,
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        label="Observed accuracy",
    )

    # Perfect calibration line
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")

    # Count annotations
    for i, (center, count) in enumerate(zip(bin_centers, bin_counts)):
        if count > 0:
            ax1.text(
                center,
                bin_accuracies[i] + 0.03,
                f"n={count}",
                ha="center",
                fontsize=7,
                color="gray",
            )

    ax1.set_xlabel("Mean Ensemble Confidence")
    ax1.set_ylabel("Observed Accuracy (fraction TP)")
    ax1.set_title(f"Reliability Diagram (ECE = {ece:.3f})")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc="upper left")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_risk_coverage(
    coverages: np.ndarray,
    risks: np.ndarray,
    aurc: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot risk-coverage curve (Figure 6).

    Risk = 1 - precision on the retained set, as a function of coverage.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(coverages, risks, "b-", linewidth=2, label=f"Risk curve (AURC={aurc:.3f})")
    ax.fill_between(coverages, risks, alpha=0.15, color="blue")

    # Mark zero-risk region
    zero_risk_end = 0
    for i, r in enumerate(risks):
        if r > 0:
            break
        zero_risk_end = coverages[i]

    if zero_risk_end > 0:
        ax.axvspan(0, zero_risk_end, alpha=0.1, color="green", label=f"Zero-risk region (cov<={zero_risk_end:.2f})")

    ax.set_xlabel("Coverage (fraction of detections retained)")
    ax.set_ylabel("Risk (1 - precision)")
    ax.set_title("Risk-Coverage Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_scatter_confidence_variance(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot mean confidence vs. confidence variance by TP/FP (Figure 7).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    tp_mask = labels == 1
    fp_mask = labels == 0

    ax.scatter(
        mean_conf[tp_mask],
        conf_var[tp_mask],
        c="#2196F3",
        marker="o",
        s=40,
        alpha=0.6,
        edgecolors="navy",
        linewidth=0.5,
        label=f"TP (n={np.sum(tp_mask)})",
    )
    ax.scatter(
        mean_conf[fp_mask],
        conf_var[fp_mask],
        c="#F44336",
        marker="x",
        s=40,
        alpha=0.6,
        linewidth=1.5,
        label=f"FP (n={np.sum(fp_mask)})",
    )

    ax.set_xlabel("Mean Ensemble Confidence ($\\bar{s}_j$)")
    ax.set_ylabel("Confidence Variance ($\\sigma^2_{s,j}$)")
    ax.set_title("Uncertainty Landscape: Mean Confidence vs. Variance")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_frame_risk_scatter(
    frame_summaries: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-frame prediction count vs. high-uncertainty FP count (Figure 8).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    totals = [f["total_detections"] for f in frame_summaries]
    high_var_fps = [f["high_var_fp_count"] for f in frame_summaries]

    colors = ["#F44336" if h > 0 else "#4CAF50" for h in high_var_fps]

    ax.scatter(totals, high_var_fps, c=colors, s=60, alpha=0.7, edgecolors="black", linewidth=0.5)

    ax.set_xlabel("Total Predictions per Frame")
    ax.set_ylabel("High-Uncertainty FP Count per Frame")
    ax.set_title("Frame-Level Risk Analysis")

    # Legend
    safe_patch = mpatches.Patch(color="#4CAF50", label="No high-uncertainty FP")
    risk_patch = mpatches.Patch(color="#F44336", label="Contains high-uncertainty FP (flagged)")
    ax.legend(handles=[safe_patch, risk_patch])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple indicators.

    Parameters
    ----------
    roc_data : dict
        Keys are indicator names, values are (fpr, tpr, auroc) tuples.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    for i, (name, (fpr, tpr, auroc)) in enumerate(roc_data.items()):
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUROC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: TP vs FP Discrimination")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tc_ranking(
    tc_results: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot triggering condition ranking as horizontal bar chart.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    conditions = [r["condition"] for r in tc_results]
    fp_shares = [r["fp_share"] for r in tc_results]
    mean_vars = [r["mean_var_fp"] for r in tc_results]

    # Replace NaN with 0 for plotting
    mean_vars = [v if not np.isnan(v) else 0 for v in mean_vars]

    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(conditions)))

    # FP share
    bars1 = ax1.barh(conditions, fp_shares, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xlabel("FP Share")
    ax1.set_title("Ranking by FP Share")
    for bar, val in zip(bars1, fp_shares):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=9)

    # Mean FP variance
    bars2 = ax2.barh(conditions, mean_vars, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Mean FP Variance ($\\sigma^2_s$)")
    ax2.set_title("Ranking by Mean FP Uncertainty")
    for bar, val in zip(bars2, mean_vars):
        ax2.text(val + 0.0001, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", fontsize=9)

    plt.suptitle("Triggering Condition Identification (ISO 21448, Clause 7)", fontsize=13, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_operating_points_comparison(
    operating_points: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot operating points as coverage vs FAR scatter.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    coverages = [p["coverage"] for p in operating_points if p["coverage"] > 0]
    fars = [p["far"] for p in operating_points if p["coverage"] > 0]
    gate_labels = [p["gate"] for p in operating_points if p["coverage"] > 0]

    scatter = ax.scatter(coverages, fars, c=fars, cmap="RdYlGn_r", s=80, edgecolors="black", linewidth=0.5, zorder=5)

    # Annotate key points
    for cov, far, label in zip(coverages, fars, gate_labels):
        if far == 0 or cov > 0.25:
            ax.annotate(
                label,
                (cov, far),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=7,
                alpha=0.8,
            )

    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5, label="Zero FAR")
    ax.set_xlabel("Coverage (fraction retained)")
    ax.set_ylabel("False Acceptance Rate (FAR)")
    ax.set_title("Operating Points: Coverage vs. False Acceptance Rate")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_iso21448_scenario_grid(save_path: Optional[str] = None) -> plt.Figure:
    """
    Reproduce the ISO 21448 known/unknown scenario categorisation (Figure 1).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))

    # Four quadrants
    ax.fill([0, 4, 4, 0], [0, 0, 3, 3], color="#C8E6C9", alpha=0.7)  # Area 1
    ax.fill([4, 8, 8, 4], [0, 0, 3, 3], color="#FFCDD2", alpha=0.7)  # Area 2
    ax.fill([4, 8, 8, 4], [3, 3, 6, 6], color="#EF9A9A", alpha=0.7)  # Area 3
    ax.fill([0, 4, 4, 0], [3, 3, 6, 6], color="#E8F5E9", alpha=0.7)  # Area 4

    ax.plot([0, 8], [3, 3], "k-", linewidth=2)
    ax.plot([4, 4], [0, 6], "k-", linewidth=2)
    ax.plot([0, 8, 8, 0, 0], [0, 0, 6, 6, 0], "k-", linewidth=2)

    # Labels
    ax.text(2, 2.4, "Area 1", ha="center", fontsize=12, fontweight="bold")
    ax.text(2, 1.5, "Known TCs\nSafe behaviour", ha="center", fontsize=8)
    ax.text(6, 2.4, "Area 2", ha="center", fontsize=12, fontweight="bold")
    ax.text(6, 1.5, "Known TCs\nPotentially hazardous", ha="center", fontsize=8)
    ax.text(6, 5.4, "Area 3", ha="center", fontsize=12, fontweight="bold")
    ax.text(6, 4.5, "Unknown TCs\nPotentially hazardous", ha="center", fontsize=8)
    ax.text(2, 5.4, "Area 4", ha="center", fontsize=12, fontweight="bold")
    ax.text(2, 4.5, "Unknown TCs\nSafe behaviour", ha="center", fontsize=8)

    # SOTIF goal arrow
    ax.annotate(
        "",
        xy=(6, 2.8),
        xytext=(6, 4.0),
        arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
    )
    ax.text(6.3, 3.4, "SOTIF goal", fontsize=8, color="red", fontweight="bold")

    # Axis labels
    ax.set_ylabel("Triggering Condition", fontsize=11, fontweight="bold")
    ax.set_xlabel("System Behaviour", fontsize=11, fontweight="bold")
    ax.set_xticks([2, 6])
    ax.set_xticklabels(["Safe", "Potentially Hazardous"], fontsize=9)
    ax.set_yticks([1.5, 4.5])
    ax.set_yticklabels(["Known", "Unknown"], fontsize=9)
    ax.set_title("ISO 21448 Scenario Categorisation (Clause 4.2.2)")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_indicator_distributions(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot distribution histograms for all three uncertainty indicators,
    split by TP/FP. Shows the separation between correct and incorrect
    detections for each indicator.
    """
    _apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tp_mask = labels == 1
    fp_mask = labels == 0

    indicators = [
        ("Mean Confidence ($\\bar{s}_j$)", mean_conf, True),
        ("Confidence Variance ($\\sigma^2_{s,j}$)", conf_var, False),
        ("Geometric Disagreement ($d_{iou,j}$)", geo_disagree, False),
    ]

    for ax, (name, vals, higher_is_correct) in zip(axes, indicators):
        bins = 30
        ax.hist(vals[tp_mask], bins=bins, alpha=0.6, color="#2196F3",
                label=f"TP (n={np.sum(tp_mask)})", density=True, edgecolor="white")
        ax.hist(vals[fp_mask], bins=bins, alpha=0.6, color="#F44336",
                label=f"FP (n={np.sum(fp_mask)})", density=True, edgecolor="white")

        # Mark medians
        ax.axvline(np.median(vals[tp_mask]), color="#1565C0", linestyle="--", linewidth=1.5, alpha=0.8)
        ax.axvline(np.median(vals[fp_mask]), color="#C62828", linestyle="--", linewidth=1.5, alpha=0.8)

        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Uncertainty Indicator Distributions: TP vs FP", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_condition_boxplots(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    labels: np.ndarray,
    conditions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Box plots of uncertainty indicators per triggering condition category.
    Shows how uncertainty varies across environmental conditions.
    """
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    unique_conds = sorted(np.unique(conditions))
    fp_mask = labels == 0

    # 1. Mean confidence per condition (all detections)
    ax = axes[0, 0]
    data = [mean_conf[conditions == c] for c in unique_conds]
    bp = ax.boxplot(data, labels=unique_conds, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#E3F2FD')
    ax.set_ylabel("Mean Confidence")
    ax.set_title("Mean Confidence by Condition (All Detections)")

    # 2. Confidence variance per condition (all detections)
    ax = axes[0, 1]
    data = [conf_var[conditions == c] for c in unique_conds]
    bp = ax.boxplot(data, labels=unique_conds, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#FFF3E0')
    ax.set_ylabel("Confidence Variance")
    ax.set_title("Confidence Variance by Condition (All Detections)")

    # 3. FP-only mean confidence
    ax = axes[1, 0]
    data = [mean_conf[fp_mask & (conditions == c)] for c in unique_conds]
    data = [d for d in data if len(d) > 0]
    cond_labels = [c for c, d in zip(unique_conds, [mean_conf[fp_mask & (conditions == c)] for c in unique_conds]) if len(d) > 0]
    if data:
        bp = ax.boxplot(data, labels=cond_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#FFEBEE')
    ax.set_ylabel("Mean Confidence (FP only)")
    ax.set_title("FP Mean Confidence by Condition")

    # 4. FP-only variance
    ax = axes[1, 1]
    data = [conf_var[fp_mask & (conditions == c)] for c in unique_conds]
    data = [d for d in data if len(d) > 0]
    if data:
        bp = ax.boxplot(data, labels=cond_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('#FFEBEE')
    ax.set_ylabel("Confidence Variance (FP only)")
    ax.set_title("FP Confidence Variance by Condition")

    plt.suptitle("Per-Condition Uncertainty Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_member_agreement(
    scores: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot ensemble member agreement: how many members detected each proposal.
    Shows that TP are detected by most/all members while FP have fewer.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    K = scores.shape[1]
    n_detecting = np.sum(scores > 0, axis=1)

    tp_mask = labels == 1
    fp_mask = labels == 0

    bins = np.arange(0.5, K + 1.5, 1)

    # TP
    ax1.hist(n_detecting[tp_mask], bins=bins, color="#2196F3", alpha=0.7,
             edgecolor="white", label=f"TP (n={np.sum(tp_mask)})")
    ax1.set_xlabel("Number of Members Detecting")
    ax1.set_ylabel("Count")
    ax1.set_title("TP: Member Agreement")
    ax1.set_xticks(range(1, K + 1))
    ax1.legend()

    # FP
    ax2.hist(n_detecting[fp_mask], bins=bins, color="#F44336", alpha=0.7,
             edgecolor="white", label=f"FP (n={np.sum(fp_mask)})")
    ax2.set_xlabel("Number of Members Detecting")
    ax2.set_ylabel("Count")
    ax2.set_title("FP: Member Agreement")
    ax2.set_xticks(range(1, K + 1))
    ax2.legend()

    fig.suptitle(f"Ensemble Member Agreement (K={K})", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_condition_breakdown(
    labels: np.ndarray,
    conditions: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Stacked bar chart showing TP/FP breakdown per condition.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    unique_conds = sorted(np.unique(conditions))
    tp_counts = [np.sum((labels == 1) & (conditions == c)) for c in unique_conds]
    fp_counts = [np.sum((labels == 0) & (conditions == c)) for c in unique_conds]

    x = np.arange(len(unique_conds))
    width = 0.6

    bars_tp = ax.bar(x, tp_counts, width, label="TP", color="#2196F3", edgecolor="white")
    bars_fp = ax.bar(x, fp_counts, width, bottom=tp_counts, label="FP", color="#F44336", edgecolor="white")

    # Annotate
    for i, (tp, fp) in enumerate(zip(tp_counts, fp_counts)):
        total = tp + fp
        ax.text(i, total + 1, f"{total}", ha="center", fontsize=9, fontweight="bold")
        if tp > 0:
            ax.text(i, tp / 2, f"{tp}", ha="center", fontsize=8, color="white")
        if fp > 0:
            ax.text(i, tp + fp / 2, f"{fp}", ha="center", fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(unique_conds, rotation=15, ha="right")
    ax.set_ylabel("Detection Count")
    ax.set_title("TP / FP Breakdown by Triggering Condition")
    ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_operating_point_heatmap(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap showing FAR as a function of confidence threshold and variance threshold.
    """
    _apply_style()
    from sotif_uncertainty.sotif_analysis import acceptance_gate, compute_coverage_far

    tau_s_vals = np.arange(0.30, 0.91, 0.05)
    tau_v_vals = np.array([0.001, 0.002, 0.003, 0.005, 0.008, 0.010, 0.020, np.inf])
    tau_v_labels = ["0.001", "0.002", "0.003", "0.005", "0.008", "0.010", "0.020", "inf"]

    far_matrix = np.zeros((len(tau_v_vals), len(tau_s_vals)))
    cov_matrix = np.zeros((len(tau_v_vals), len(tau_s_vals)))

    for i, tv in enumerate(tau_v_vals):
        for j, ts in enumerate(tau_s_vals):
            accepted = acceptance_gate(mean_conf, conf_var, geo_disagree, ts, tv)
            cov, far, _, _ = compute_coverage_far(accepted, labels)
            far_matrix[i, j] = far
            cov_matrix[i, j] = cov

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # FAR heatmap
    im1 = ax1.imshow(far_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(len(tau_s_vals)))
    ax1.set_xticklabels([f"{t:.2f}" for t in tau_s_vals], rotation=45, fontsize=8)
    ax1.set_yticks(range(len(tau_v_vals)))
    ax1.set_yticklabels(tau_v_labels, fontsize=8)
    ax1.set_xlabel("Confidence Threshold ($\\tau_s$)")
    ax1.set_ylabel("Variance Threshold ($\\tau_v$)")
    ax1.set_title("False Acceptance Rate (FAR)")
    plt.colorbar(im1, ax=ax1)

    # Coverage heatmap
    im2 = ax2.imshow(cov_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=0.5)
    ax2.set_xticks(range(len(tau_s_vals)))
    ax2.set_xticklabels([f"{t:.2f}" for t in tau_s_vals], rotation=45, fontsize=8)
    ax2.set_yticks(range(len(tau_v_vals)))
    ax2.set_yticklabels(tau_v_labels, fontsize=8)
    ax2.set_xlabel("Confidence Threshold ($\\tau_s$)")
    ax2.set_ylabel("Variance Threshold ($\\tau_v$)")
    ax2.set_title("Coverage")
    plt.colorbar(im2, ax=ax2)

    fig.suptitle("Acceptance Gate Operating Space", fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_summary_dashboard(
    metrics: Dict,
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    labels: np.ndarray,
    tc_results: List[Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-panel summary dashboard with key results.
    """
    _apply_style()
    fig = plt.figure(figsize=(18, 12))

    tp_mask = labels == 1
    fp_mask = labels == 0
    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    # 1. Scatter (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.scatter(mean_conf[tp_mask], conf_var[tp_mask], c="#2196F3", s=20, alpha=0.5, label="TP")
    ax1.scatter(mean_conf[fp_mask], conf_var[fp_mask], c="#F44336", marker="x", s=20, alpha=0.5, label="FP")
    ax1.set_xlabel("Mean Confidence")
    ax1.set_ylabel("Confidence Variance")
    ax1.set_title("Uncertainty Landscape")
    ax1.legend(fontsize=8)

    # 2. ROC (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    fpr = disc["roc_fpr"]
    tpr = disc["roc_tpr"]
    ax2.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC={disc['auroc_mean_confidence']:.3f}")
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("ROC Curve (Mean Confidence)")
    ax2.legend(fontsize=8)

    # 3. Risk-Coverage (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(rc["coverages"], rc["risks"], "b-", linewidth=2)
    ax3.fill_between(rc["coverages"], rc["risks"], alpha=0.1)
    ax3.set_xlabel("Coverage")
    ax3.set_ylabel("Risk")
    ax3.set_title(f"Risk-Coverage (AURC={rc['aurc']:.3f})")

    # 4. Reliability diagram (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    n_bins = len(cal["bin_accuracies"])
    bin_centers = np.linspace(1 / (2 * n_bins), 1 - 1 / (2 * n_bins), n_bins)
    ax4.bar(bin_centers, cal["bin_accuracies"], width=0.08, alpha=0.7, color="#2196F3", edgecolor="white")
    ax4.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax4.set_xlabel("Predicted Confidence")
    ax4.set_ylabel("Observed Accuracy")
    ax4.set_title(f"Calibration (ECE={cal['ece']:.3f})")
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1.1)

    # 5. TC ranking (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    conds = [r["condition"] for r in tc_results]
    shares = [r["fp_share"] for r in tc_results]
    colors_tc = plt.cm.Reds(np.linspace(0.3, 0.9, len(conds)))
    ax5.barh(conds, shares, color=colors_tc, edgecolor="black", linewidth=0.5)
    ax5.set_xlabel("FP Share")
    ax5.set_title("TC Ranking (Clause 7)")

    # 6. Key metrics text box (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    metrics_text = (
        f"SOTIF Evaluation Summary\n"
        f"{'='*35}\n\n"
        f"Dataset:\n"
        f"  Proposals: {len(labels)} ({np.sum(tp_mask)} TP, {np.sum(fp_mask)} FP)\n\n"
        f"Discrimination (Table 3):\n"
        f"  AUROC (conf):  {disc['auroc_mean_confidence']:.3f}\n"
        f"  AUROC (var):   {disc['auroc_confidence_variance']:.3f}\n"
        f"  AUROC (geo):   {disc['auroc_geometric_disagreement']:.3f}\n\n"
        f"Calibration (Table 4):\n"
        f"  ECE:   {cal['ece']:.3f}\n"
        f"  NLL:   {cal['nll']:.3f}\n"
        f"  Brier: {cal['brier']:.3f}\n"
        f"  AURC:  {rc['aurc']:.3f}\n\n"
        f"Top Triggering Conditions:\n"
        f"  1. {tc_results[0]['condition']} ({tc_results[0]['fp_share']:.0%})\n"
        f"  2. {tc_results[1]['condition']} ({tc_results[1]['fp_share']:.0%})"
    )
    ax6.text(0.05, 0.95, metrics_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))

    fig.suptitle("SOTIF Uncertainty Evaluation -- Summary Dashboard", fontsize=15, y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_figures(
    metrics: Dict,
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    labels: np.ndarray,
    frame_summaries: List[Dict],
    tc_results: List[Dict],
    operating_points: List[Dict],
    output_dir: str = "results",
    scores: Optional[np.ndarray] = None,
    geo_disagree: Optional[np.ndarray] = None,
    conditions: Optional[np.ndarray] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate ALL figures and save to output directory.

    Produces 13 figures covering all aspects of the evaluation:
    - Paper figures (reliability, risk-coverage, scatter, frame risk)
    - ROC curves, TC ranking, operating points, ISO grid
    - NEW: distributions, box plots, member agreement, condition breakdown
    - NEW: operating point heatmap, summary dashboard
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # ============================================================
    # Paper figures (Figures 5-8)
    # ============================================================
    cal = metrics["calibration"]
    figures["reliability_diagram"] = plot_reliability_diagram(
        cal["bin_accuracies"], cal["bin_confidences"], cal["bin_counts"],
        cal["ece"], save_path=os.path.join(output_dir, "reliability_diagram_rich.png"),
    )

    rc = metrics["risk_coverage"]
    figures["risk_coverage"] = plot_risk_coverage(
        rc["coverages"], rc["risks"], rc["aurc"],
        save_path=os.path.join(output_dir, "risk_coverage_curve.png"),
    )

    figures["scatter_conf_var"] = plot_scatter_confidence_variance(
        mean_conf, conf_var, labels,
        save_path=os.path.join(output_dir, "scatter_score_var_tp_fp.png"),
    )

    figures["frame_risk"] = plot_frame_risk_scatter(
        frame_summaries,
        save_path=os.path.join(output_dir, "frame_risk_scatter.png"),
    )

    # ============================================================
    # Analysis figures
    # ============================================================
    figures["tc_ranking"] = plot_tc_ranking(
        tc_results,
        save_path=os.path.join(output_dir, "tc_ranking.png"),
    )

    figures["operating_points"] = plot_operating_points_comparison(
        operating_points,
        save_path=os.path.join(output_dir, "operating_points.png"),
    )

    figures["iso21448_grid"] = plot_iso21448_scenario_grid(
        save_path=os.path.join(output_dir, "iso21448_scenario_grid.png"),
    )

    # ============================================================
    # Extended visualizations
    # ============================================================
    if geo_disagree is not None:
        figures["indicator_distributions"] = plot_indicator_distributions(
            mean_conf, conf_var, geo_disagree, labels,
            save_path=os.path.join(output_dir, "indicator_distributions.png"),
        )

        figures["operating_heatmap"] = plot_operating_point_heatmap(
            mean_conf, conf_var, geo_disagree, labels,
            save_path=os.path.join(output_dir, "operating_point_heatmap.png"),
        )

    if conditions is not None:
        figures["condition_boxplots"] = plot_condition_boxplots(
            mean_conf, conf_var, labels, conditions,
            save_path=os.path.join(output_dir, "condition_boxplots.png"),
        )

        figures["condition_breakdown"] = plot_condition_breakdown(
            labels, conditions,
            save_path=os.path.join(output_dir, "condition_breakdown.png"),
        )

    if scores is not None:
        figures["member_agreement"] = plot_member_agreement(
            scores, labels,
            save_path=os.path.join(output_dir, "member_agreement.png"),
        )

    # Summary dashboard
    figures["summary_dashboard"] = plot_summary_dashboard(
        metrics, mean_conf, conf_var, labels, tc_results,
        save_path=os.path.join(output_dir, "summary_dashboard.png"),
    )

    plt.close("all")
    return figures

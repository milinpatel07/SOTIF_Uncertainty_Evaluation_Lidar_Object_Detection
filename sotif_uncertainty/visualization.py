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


def generate_all_figures(
    metrics: Dict,
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    labels: np.ndarray,
    frame_summaries: List[Dict],
    tc_results: List[Dict],
    operating_points: List[Dict],
    output_dir: str = "Analysis",
) -> Dict[str, plt.Figure]:
    """
    Generate all paper figures and save to output directory.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    figures = {}

    # Reliability diagram
    cal = metrics["calibration"]
    figures["reliability"] = plot_reliability_diagram(
        cal["bin_accuracies"], cal["bin_confidences"], cal["bin_counts"],
        cal["ece"], save_path=os.path.join(output_dir, "reliability_diagram_rich.png"),
    )

    # Risk-coverage
    rc = metrics["risk_coverage"]
    figures["risk_coverage"] = plot_risk_coverage(
        rc["coverages"], rc["risks"], rc["aurc"],
        save_path=os.path.join(output_dir, "risk_coverage_curve.png"),
    )

    # Scatter
    figures["scatter"] = plot_scatter_confidence_variance(
        mean_conf, conf_var, labels,
        save_path=os.path.join(output_dir, "scatter_score_var_tp_fp.png"),
    )

    # Frame risk
    figures["frame_risk"] = plot_frame_risk_scatter(
        frame_summaries,
        save_path=os.path.join(output_dir, "frame_risk_scatter.png"),
    )

    # TC ranking
    figures["tc_ranking"] = plot_tc_ranking(
        tc_results,
        save_path=os.path.join(output_dir, "tc_ranking.png"),
    )

    # Operating points
    figures["operating_points"] = plot_operating_points_comparison(
        operating_points,
        save_path=os.path.join(output_dir, "operating_points.png"),
    )

    # ISO 21448 grid
    figures["iso21448_grid"] = plot_iso21448_scenario_grid(
        save_path=os.path.join(output_dir, "iso21448_scenario_grid.png"),
    )

    plt.close("all")
    return figures

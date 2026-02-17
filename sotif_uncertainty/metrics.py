"""
Stage 4: Metric Computation.

Three metric categories:
- Discrimination: AUROC, AURC (does uncertainty identify incorrect detections?)
- Calibration: ECE, NLL, Brier (do confidence values match observed accuracy?)
- Operating characteristics: coverage and FAR at acceptance thresholds.

Reference: Section 3.5 of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

# Compatibility: np.trapz was removed in NumPy 2.0
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def compute_auroc(
    scores: np.ndarray,
    labels: np.ndarray,
    higher_is_correct: bool = True,
) -> float:
    """
    Compute Area Under the ROC Curve for TP vs FP separation.

    AUROC = 1.0 means perfect separation; 0.5 means no discriminative ability.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Uncertainty indicator values.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    higher_is_correct : bool
        If True, higher scores indicate correct detections (e.g., mean confidence).
        If False, higher scores indicate incorrect detections (e.g., variance).

    Returns
    -------
    float
        AUROC value in [0, 1].
    """
    if len(np.unique(labels)) < 2:
        return float("nan")

    if not higher_is_correct:
        scores = -scores

    # Manual AUROC computation (no sklearn dependency)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Sort by score descending
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    # Compute TPR and FPR at each threshold
    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_cum = 0
    fp_cum = 0

    for label in sorted_labels:
        if label == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_list.append(tp_cum / n_pos)
        fpr_list.append(fp_cum / n_neg)

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return float(auc)


def compute_auroc_with_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    higher_is_correct: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute AUROC and return the ROC curve points.

    Returns
    -------
    tuple of (auroc, fpr_array, tpr_array)
    """
    if not higher_is_correct:
        scores = -scores

    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0 or n_neg == 0:
        return float("nan"), np.array([]), np.array([])

    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp_cum = 0
    fp_cum = 0

    for label in sorted_labels:
        if label == 1:
            tp_cum += 1
        else:
            fp_cum += 1
        tpr_list.append(tp_cum / n_pos)
        fpr_list.append(fp_cum / n_neg)

    fpr_arr = np.array(fpr_list)
    tpr_arr = np.array(tpr_list)
    auc = float(_trapz(tpr_arr, fpr_arr))

    return auc, fpr_arr, tpr_arr


def compute_aurc(
    scores: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute Area Under the Risk-Coverage Curve.

    Detections are sorted by confidence (descending). At each coverage level,
    risk = 1 - precision on the retained set. Lower AURC is better.

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Mean confidence scores.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).

    Returns
    -------
    tuple of (aurc_value, coverage_array, risk_array)
    """
    N = len(scores)
    sorted_idx = np.argsort(-scores)
    sorted_labels = labels[sorted_idx]

    coverages = []
    risks = []

    for i in range(1, N + 1):
        retained = sorted_labels[:i]
        coverage = i / N
        precision = np.sum(retained == 1) / len(retained)
        risk = 1.0 - precision

        coverages.append(coverage)
        risks.append(risk)

    coverages = np.array(coverages)
    risks = np.array(risks)
    aurc = float(_trapz(risks, coverages))

    return aurc, coverages, risks


def compute_ece(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (Eq. 4).

    ECE = sum_{m=1}^{M} (|B_m|/n) * |acc(B_m) - conf(B_m)|

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Predicted confidence scores.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    tuple of (ece_value, bin_accuracies, bin_confidences, bin_counts)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for m in range(n_bins):
        lower = bin_boundaries[m]
        upper = bin_boundaries[m + 1]

        if m == n_bins - 1:
            mask = (scores >= lower) & (scores <= upper)
        else:
            mask = (scores >= lower) & (scores < upper)

        bin_counts[m] = np.sum(mask)

        if bin_counts[m] > 0:
            bin_accuracies[m] = np.mean(labels[mask])
            bin_confidences[m] = np.mean(scores[mask])

    n = len(scores)
    ece = 0.0
    for m in range(n_bins):
        if bin_counts[m] > 0:
            ece += (bin_counts[m] / n) * abs(bin_accuracies[m] - bin_confidences[m])

    return float(ece), bin_accuracies, bin_confidences, bin_counts


def compute_nll(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Negative Log-Likelihood (Eq. 5).

    NLL = -(1/n) * sum [y_i * log(s_i) + (1-y_i) * log(1-s_i)]

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Predicted confidence scores in (0, 1).
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).

    Returns
    -------
    float
        NLL value (lower is better).
    """
    eps = 1e-7
    s = np.clip(scores, eps, 1 - eps)
    nll = -np.mean(labels * np.log(s) + (1 - labels) * np.log(1 - s))
    return float(nll)


def compute_brier(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier Score (Eq. 6).

    Brier = (1/n) * sum (s_i - y_i)^2

    Parameters
    ----------
    scores : np.ndarray, shape (N,)
        Predicted confidence scores.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).

    Returns
    -------
    float
        Brier score (lower is better, 0 = perfect).
    """
    return float(np.mean((scores - labels) ** 2))


def compute_all_metrics(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute all Stage 4 metrics.

    Parameters
    ----------
    mean_conf : np.ndarray, shape (N,)
    conf_var : np.ndarray, shape (N,)
    geo_disagree : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    n_bins : int

    Returns
    -------
    dict with all metric values and curve data.
    """
    # Discrimination
    auroc_conf, fpr_conf, tpr_conf = compute_auroc_with_curve(
        mean_conf, labels, higher_is_correct=True
    )
    auroc_var = compute_auroc(conf_var, labels, higher_is_correct=False)
    auroc_geo = compute_auroc(geo_disagree, labels, higher_is_correct=False)

    # Risk-coverage
    aurc_val, coverages, risks = compute_aurc(mean_conf, labels)

    # Calibration
    ece_val, bin_acc, bin_conf, bin_counts = compute_ece(mean_conf, labels, n_bins)
    nll_val = compute_nll(mean_conf, labels)
    brier_val = compute_brier(mean_conf, labels)

    return {
        "discrimination": {
            "auroc_mean_confidence": auroc_conf,
            "auroc_confidence_variance": auroc_var,
            "auroc_geometric_disagreement": auroc_geo,
            "roc_fpr": fpr_conf,
            "roc_tpr": tpr_conf,
        },
        "risk_coverage": {
            "aurc": aurc_val,
            "coverages": coverages,
            "risks": risks,
        },
        "calibration": {
            "ece": ece_val,
            "nll": nll_val,
            "brier": brier_val,
            "bin_accuracies": bin_acc,
            "bin_confidences": bin_conf,
            "bin_counts": bin_counts,
        },
    }

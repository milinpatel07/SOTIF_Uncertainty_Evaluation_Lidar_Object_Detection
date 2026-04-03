"""
Stage 5: SOTIF Analysis.

Produces three categories of SOTIF artefacts:
- Triggering condition identification and ranking (Clause 7)
- Frame-level triage flags (Clause 7)
- Acceptance criteria input via operating point tables (Clause 11)

Reference: Section 3.6 of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings


def acceptance_gate(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    tau_s: float = 0.70,
    tau_v: float = np.inf,
    tau_d: float = np.inf,
) -> np.ndarray:
    """
    Apply acceptance gate (Eq. 7).

    G(s, sigma^2, d) = [s >= tau_s] AND [sigma^2 <= tau_v] AND [d <= tau_d]

    Parameters
    ----------
    mean_conf : np.ndarray, shape (N,)
    conf_var : np.ndarray, shape (N,)
    geo_disagree : np.ndarray, shape (N,)
    tau_s : float
        Minimum mean confidence threshold.
    tau_v : float
        Maximum confidence variance threshold.
    tau_d : float
        Maximum geometric disagreement threshold.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True if detection is accepted.
    """
    accepted = (mean_conf >= tau_s) & (conf_var <= tau_v) & (geo_disagree <= tau_d)
    return accepted


def compute_coverage_far(
    accepted: np.ndarray,
    labels: np.ndarray,
) -> Tuple[float, float, int, int]:
    """
    Compute coverage and false acceptance rate (Eq. 8).

    Coverage = |A| / n
    FAR = |{j in A : y_j = 0}| / |A|

    Parameters
    ----------
    accepted : np.ndarray, shape (N,), dtype bool
    labels : np.ndarray, shape (N,)

    Returns
    -------
    tuple of (coverage, far, retained_count, fp_in_accepted)
    """
    n = len(labels)
    n_accepted = np.sum(accepted)

    if n_accepted == 0:
        return 0.0, 0.0, 0, 0

    coverage = n_accepted / n
    fp_accepted = np.sum((accepted) & (labels == 0))
    far = fp_accepted / n_accepted

    return float(coverage), float(far), int(n_accepted), int(fp_accepted)


def compute_operating_points(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    labels: np.ndarray,
    tau_s_range: Optional[np.ndarray] = None,
    tau_v_range: Optional[np.ndarray] = None,
    tau_d_range: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Compute operating point table for acceptance gates (Table 6 in paper).

    Sweeps threshold combinations and reports coverage, retained count,
    FP count, and FAR for each.

    Parameters
    ----------
    mean_conf, conf_var, geo_disagree : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    tau_s_range : np.ndarray, optional
        Confidence thresholds to evaluate.
    tau_v_range : np.ndarray, optional
        Variance thresholds to evaluate.
    tau_d_range : np.ndarray, optional
        Geometric disagreement thresholds to evaluate.

    Returns
    -------
    list of dict
        Each dict contains: gate_description, tau_s, tau_v, tau_d,
        coverage, retained, fp, far.
    """
    if tau_s_range is None:
        tau_s_range = np.arange(0.50, 0.91, 0.05)
    if tau_v_range is None:
        tau_v_range = np.array([np.inf])
    if tau_d_range is None:
        tau_d_range = np.array([np.inf])

    results = []
    for ts in tau_s_range:
        for tv in tau_v_range:
            for td in tau_d_range:
                accepted = acceptance_gate(
                    mean_conf, conf_var, geo_disagree, ts, tv, td
                )
                cov, far, ret, fp = compute_coverage_far(accepted, labels)

                # Build gate description
                parts = [f"s>={ts:.2f}"]
                if not np.isinf(tv):
                    parts.append(f"var<={tv:.4f}")
                if not np.isinf(td):
                    parts.append(f"d<={td:.2f}")
                desc = " & ".join(parts)

                results.append({
                    "gate": desc,
                    "tau_s": float(ts),
                    "tau_v": float(tv),
                    "tau_d": float(td),
                    "coverage": cov,
                    "retained": ret,
                    "fp": fp,
                    "far": far,
                })

    return results


def find_optimal_gate(
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    geo_disagree: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.0,
    tau_s_range: Optional[np.ndarray] = None,
    tau_v_range: Optional[np.ndarray] = None,
    tau_d_range: Optional[np.ndarray] = None,
) -> Dict:
    """
    Find optimal gate maximising coverage subject to FAR <= alpha (Eq. 9).

    Parameters
    ----------
    alpha : float
        Maximum acceptable false acceptance rate.

    Returns
    -------
    dict
        Best operating point, or None if no point satisfies the constraint.
    """
    points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, labels,
        tau_s_range, tau_v_range, tau_d_range,
    )

    valid = [p for p in points if p["far"] <= alpha]
    if not valid:
        warnings.warn(f"No operating point satisfies FAR <= {alpha}")
        return None

    # Maximise coverage among valid points
    return max(valid, key=lambda p: p["coverage"])


def rank_triggering_conditions(
    conditions: np.ndarray,
    labels: np.ndarray,
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
) -> List[Dict]:
    """
    Rank triggering conditions by FP share and mean uncertainty (Table 7).

    For each condition: compute FP share, mean confidence among FP,
    and mean variance among FP.

    Parameters
    ----------
    conditions : np.ndarray, shape (N,)
        Condition label per detection (e.g., 'heavy_rain', 'night').
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    mean_conf : np.ndarray, shape (N,)
    conf_var : np.ndarray, shape (N,)

    Returns
    -------
    list of dict, sorted by FP share descending.
        Each dict: condition, fp_count, fp_share, mean_conf_fp, mean_var_fp,
        tp_count, total_count.
    """
    unique_conditions = np.unique(conditions)
    total_fp = np.sum(labels == 0)

    results = []
    for cond in unique_conditions:
        mask = conditions == cond
        cond_labels = labels[mask]
        cond_conf = mean_conf[mask]
        cond_var = conf_var[mask]

        fp_mask = cond_labels == 0
        tp_mask = cond_labels == 1

        fp_count = int(np.sum(fp_mask))
        tp_count = int(np.sum(tp_mask))
        total = int(np.sum(mask))
        fp_share = fp_count / total_fp if total_fp > 0 else 0.0

        mean_conf_fp = float(np.mean(cond_conf[fp_mask])) if fp_count > 0 else float("nan")
        mean_var_fp = float(np.mean(cond_var[fp_mask])) if fp_count > 0 else float("nan")

        results.append({
            "condition": str(cond),
            "fp_count": fp_count,
            "fp_share": fp_share,
            "mean_conf_fp": mean_conf_fp,
            "mean_var_fp": mean_var_fp,
            "tp_count": tp_count,
            "total": total,
        })

    # Sort by FP share descending
    results.sort(key=lambda x: x["fp_share"], reverse=True)

    return results


def flag_frames(
    frame_ids: np.ndarray,
    labels: np.ndarray,
    conf_var: np.ndarray,
    percentile: float = 80.0,
    var_threshold: Optional[float] = None,
) -> Dict:
    """
    Flag frames containing high-uncertainty false positives (Section 3.6).

    A frame is flagged if it contains at least one FP whose variance
    exceeds the given percentile of the FP variance distribution.

    Parameters
    ----------
    frame_ids : np.ndarray, shape (N,)
        Frame identifier per detection.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    conf_var : np.ndarray, shape (N,)
        Confidence variance per detection.
    percentile : float
        Percentile of FP variance distribution for threshold.
    var_threshold : float, optional
        If provided, overrides percentile computation.

    Returns
    -------
    dict with keys:
        'flagged_frames': list of frame IDs
        'threshold': float (the variance threshold used)
        'total_frames': int
        'flagged_count': int
    """
    fp_mask = labels == 0
    fp_vars = conf_var[fp_mask]

    if len(fp_vars) == 0:
        return {
            "flagged_frames": [],
            "threshold": 0.0,
            "total_frames": len(np.unique(frame_ids)),
            "flagged_count": 0,
        }

    if var_threshold is None:
        var_threshold = float(np.percentile(fp_vars, percentile))

    # Find FP detections above threshold
    high_var_fp = fp_mask & (conf_var > var_threshold)

    # Find frames containing such detections
    flagged = set()
    unique_frames = np.unique(frame_ids)

    for fid in unique_frames:
        frame_mask = frame_ids == fid
        if np.any(high_var_fp[frame_mask]):
            flagged.add(fid)

    return {
        "flagged_frames": sorted(flagged),
        "threshold": var_threshold,
        "total_frames": len(unique_frames),
        "flagged_count": len(flagged),
    }


def compute_frame_summary(
    frame_ids: np.ndarray,
    labels: np.ndarray,
    mean_conf: np.ndarray,
    conf_var: np.ndarray,
    conditions: Optional[np.ndarray] = None,
) -> List[Dict]:
    """
    Compute per-frame summary statistics for frame-level analysis.

    Parameters
    ----------
    frame_ids : np.ndarray, shape (N,)
    labels : np.ndarray, shape (N,)
    mean_conf : np.ndarray, shape (N,)
    conf_var : np.ndarray, shape (N,)
    conditions : np.ndarray, shape (N,), optional

    Returns
    -------
    list of dict per frame.
    """
    unique_frames = np.unique(frame_ids)
    summaries = []

    for fid in unique_frames:
        mask = frame_ids == fid
        f_labels = labels[mask]
        f_conf = mean_conf[mask]
        f_var = conf_var[mask]

        total = int(np.sum(mask))
        tp = int(np.sum(f_labels == 1))
        fp = int(np.sum(f_labels == 0))

        fp_mask = f_labels == 0
        high_var_fp = int(np.sum(fp_mask & (f_var > np.percentile(conf_var[labels == 0], 80)))) if np.sum(labels == 0) > 0 else 0

        summary = {
            "frame_id": fid,
            "total_detections": total,
            "tp": tp,
            "fp": fp,
            "mean_confidence": float(np.mean(f_conf)),
            "max_variance": float(np.max(f_var)) if total > 0 else 0.0,
            "high_var_fp_count": high_var_fp,
        }

        if conditions is not None:
            cond_vals = conditions[mask]
            summary["condition"] = str(cond_vals[0]) if len(cond_vals) > 0 else "unknown"

        summaries.append(summary)

    return summaries

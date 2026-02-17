"""
Ensemble Aggregation via DBSCAN Clustering.

Associates detections from K ensemble members into unified proposals
using DBSCAN clustering on a BEV IoU distance matrix, following the
LiDAR-MIMO / uncertainty_eval methodology.

Three voting strategies control the minimum cluster size:
- Affirmative: keep any detection (min_samples=1)
- Consensus:   majority must agree (min_samples=K//2+1)
- Unanimous:   all members must agree (min_samples=K)

Reference:
    Pitropov et al. (2022). LiDAR-MIMO.
    Section 3.3 of the SOTIF uncertainty evaluation paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BEV IoU computation (vectorised for large distance matrices)
# ---------------------------------------------------------------------------

def _bev_iou_pair(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Axis-aligned BEV IoU between two 7-DOF boxes."""
    xa, ya, wa, la = box_a[0], box_a[1], box_a[3], box_a[4]
    xb, yb, wb, lb = box_b[0], box_b[1], box_b[3], box_b[4]

    x1 = max(xa - wa / 2, xb - wb / 2)
    x2 = min(xa + wa / 2, xb + wb / 2)
    y1 = max(ya - la / 2, yb - lb / 2)
    y2 = min(ya + la / 2, yb + lb / 2)

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = wa * la + wb * lb - inter
    return float(inter / union) if union > 1e-8 else 0.0


def compute_iou_distance_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise (1 - BEV IoU) distance matrix.

    Parameters
    ----------
    boxes : np.ndarray, shape (N, 7)
        All boxes from all ensemble members for one frame.

    Returns
    -------
    np.ndarray, shape (N, N)
        Symmetric distance matrix in [0, 1].
    """
    N = len(boxes)
    dist = np.ones((N, N))
    for i in range(N):
        dist[i, i] = 0.0
        for j in range(i + 1, N):
            iou = _bev_iou_pair(boxes[i], boxes[j])
            d = 1.0 - min(iou, 1.0)
            dist[i, j] = d
            dist[j, i] = d
    return dist


# ---------------------------------------------------------------------------
# DBSCAN clustering
# ---------------------------------------------------------------------------

def cluster_detections(
    member_predictions: List[List[Dict]],
    iou_threshold: float = 0.5,
    voting: str = "consensus",
) -> List[Dict]:
    """
    Cluster detections from K ensemble members using DBSCAN.

    Parameters
    ----------
    member_predictions : list of list of dict
        Outer list: K ensemble members.
        Inner list: per-frame dicts with keys
            'boxes_lidar' (N, 7), 'score' (N,), 'pred_labels' (N,),
            'frame_id' str.
    iou_threshold : float
        Minimum BEV IoU to consider two detections as the same object.
    voting : str
        'affirmative' (min_samples=1), 'consensus' (K//2+1),
        or 'unanimous' (K).

    Returns
    -------
    list of dict
        One dict per frame with keys:
            'frame_id', 'boxes_lidar' (P, 7), 'scores' (P, K),
            'mean_score' (P,), 'pred_labels' (P,),
            'cluster_sizes' (P,), 'member_ids' list of lists.
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError(
            "scikit-learn is required for DBSCAN clustering. "
            "Install it with: pip install scikit-learn"
        )

    K = len(member_predictions)

    # Determine min_samples from voting strategy
    if voting == "affirmative":
        min_samples = 1
    elif voting == "consensus":
        min_samples = K // 2 + 1
    elif voting == "unanimous":
        min_samples = K
    else:
        raise ValueError(f"Unknown voting strategy: {voting}")

    eps = 1.0 - iou_threshold

    # Group predictions by frame_id
    frames = {}  # frame_id -> list of (member_idx, box_idx_in_member, box, score, label)
    for k in range(K):
        for pred in member_predictions[k]:
            fid = pred["frame_id"]
            if fid not in frames:
                frames[fid] = []
            n_det = len(pred["score"])
            for i in range(n_det):
                frames[fid].append({
                    "member": k,
                    "box": pred["boxes_lidar"][i],
                    "score": pred["score"][i],
                    "label": pred["pred_labels"][i] if "pred_labels" in pred else 1,
                })

    results = []
    for fid in sorted(frames.keys()):
        dets = frames[fid]
        if len(dets) == 0:
            results.append({
                "frame_id": fid,
                "boxes_lidar": np.zeros((0, 7)),
                "scores": np.zeros((0, K)),
                "mean_score": np.zeros(0),
                "pred_labels": np.zeros(0, dtype=int),
                "cluster_sizes": np.zeros(0, dtype=int),
                "member_ids": [],
            })
            continue

        all_boxes = np.array([d["box"] for d in dets])
        all_scores = np.array([d["score"] for d in dets])
        all_members = np.array([d["member"] for d in dets])
        all_labels = np.array([d["label"] for d in dets])

        # Compute distance matrix and run DBSCAN
        dist_matrix = compute_iou_distance_matrix(all_boxes)
        cluster_ids = DBSCAN(
            eps=eps, min_samples=min_samples, metric="precomputed"
        ).fit_predict(dist_matrix)

        # Aggregate clusters
        valid_ids = set(cluster_ids[cluster_ids >= 0])
        n_clusters = len(valid_ids)

        agg_boxes = np.zeros((n_clusters, 7))
        agg_scores = np.zeros((n_clusters, K))
        agg_labels = np.zeros(n_clusters, dtype=int)
        agg_sizes = np.zeros(n_clusters, dtype=int)
        agg_member_ids = []

        for idx, cid in enumerate(sorted(valid_ids)):
            mask = cluster_ids == cid
            cluster_boxes = all_boxes[mask]
            cluster_scores = all_scores[mask]
            cluster_members = all_members[mask]
            cluster_labels = all_labels[mask]

            # Mean box (yaw from highest-confidence member)
            best_idx = np.argmax(cluster_scores)
            mean_box = np.mean(cluster_boxes, axis=0)
            mean_box[6] = cluster_boxes[best_idx, 6]  # yaw from best member
            agg_boxes[idx] = mean_box

            # Per-member scores (K-dimensional vector)
            for det_i in range(len(cluster_scores)):
                m = cluster_members[det_i]
                # If multiple detections from same member, keep highest score
                agg_scores[idx, m] = max(agg_scores[idx, m], cluster_scores[det_i])

            # Label from highest-confidence detection
            agg_labels[idx] = cluster_labels[best_idx]
            agg_sizes[idx] = len(set(cluster_members))
            agg_member_ids.append(sorted(set(cluster_members.tolist())))

        results.append({
            "frame_id": fid,
            "boxes_lidar": agg_boxes,
            "scores": agg_scores,
            "mean_score": np.mean(agg_scores, axis=1),
            "pred_labels": agg_labels,
            "cluster_sizes": agg_sizes,
            "member_ids": agg_member_ids,
        })

    return results


# ---------------------------------------------------------------------------
# Uncertainty decomposition (classification)
# ---------------------------------------------------------------------------

def compute_classification_uncertainty(
    score_all_members: np.ndarray,
) -> Dict[str, float]:
    """
    Decompose classification uncertainty into total, aleatoric, and epistemic.

    Uses Shannon entropy of the mean softmax distribution (total),
    mean of individual entropies (aleatoric), and their difference
    (mutual information = epistemic).

    Parameters
    ----------
    score_all_members : np.ndarray, shape (K, C)
        Softmax probabilities from K members over C classes.

    Returns
    -------
    dict with 'total_entropy', 'aleatoric_entropy', 'mutual_information'.
    """
    from scipy import stats
    from scipy.special import softmax

    # Apply softmax if logits (values outside [0,1] or not summing to ~1)
    row_sums = score_all_members.sum(axis=1)
    if np.any(row_sums < 0) or np.any(np.abs(row_sums - 1.0) > 0.1):
        probs = softmax(score_all_members, axis=1)
    else:
        probs = score_all_members

    # Total uncertainty: H[E[p]]
    mean_probs = np.mean(probs, axis=0)
    # Ensure valid distribution
    mean_probs = np.clip(mean_probs, 1e-10, None)
    mean_probs /= mean_probs.sum()
    total_entropy = float(stats.entropy(mean_probs, base=2))

    # Aleatoric uncertainty: E[H[p]]
    aleatoric_entropy = float(np.mean([
        stats.entropy(np.clip(probs[k], 1e-10, None), base=2)
        for k in range(len(probs))
    ]))

    # Epistemic uncertainty: MI = H - AE
    mutual_info = max(0.0, total_entropy - aleatoric_entropy)

    return {
        "total_entropy": total_entropy,
        "aleatoric_entropy": aleatoric_entropy,
        "mutual_information": mutual_info,
    }


# ---------------------------------------------------------------------------
# Uncertainty decomposition (regression)
# ---------------------------------------------------------------------------

def compute_regression_uncertainty(
    member_boxes: np.ndarray,
    member_variances: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Decompose regression uncertainty into epistemic and aleatoric.

    Parameters
    ----------
    member_boxes : np.ndarray, shape (K, 7)
        Predicted boxes from K members [x, y, z, dx, dy, dz, heading].
    member_variances : np.ndarray, shape (K, 7), optional
        Predicted aleatoric variances from K members.
        If None, only epistemic variance is computed.

    Returns
    -------
    dict with 'epistemic_var' (7,), 'aleatoric_var' (7,), 'total_var' (7,).
    """
    mean_box = np.mean(member_boxes, axis=0)

    # Epistemic: variance of predicted means
    epistemic_var = np.mean((member_boxes - mean_box) ** 2, axis=0)

    if member_variances is not None:
        aleatoric_var = np.mean(member_variances, axis=0)
        total_var = epistemic_var + aleatoric_var
    else:
        aleatoric_var = np.zeros(7)
        total_var = epistemic_var

    return {
        "epistemic_var": epistemic_var,
        "aleatoric_var": aleatoric_var,
        "total_var": total_var,
    }


# ---------------------------------------------------------------------------
# Convert clustered results to the format expected by our pipeline
# ---------------------------------------------------------------------------

def clustered_to_pipeline_format(
    clustered_frames: List[Dict],
    K: int,
) -> Dict[str, np.ndarray]:
    """
    Convert DBSCAN-clustered frame results to the flat arrays expected
    by sotif_uncertainty.uncertainty.compute_all_indicators().

    Parameters
    ----------
    clustered_frames : list of dict
        Output of cluster_detections().
    K : int
        Number of ensemble members.

    Returns
    -------
    dict with keys:
        'scores' : (N_total, K) array
        'boxes' : (N_total, K, 7) array (NaN for missing members)
        'frame_ids' : (N_total,) array of str
    """
    all_scores = []
    all_boxes = []
    all_frame_ids = []

    for frame in clustered_frames:
        n_proposals = len(frame["mean_score"])
        for p in range(n_proposals):
            all_scores.append(frame["scores"][p])

            # Build per-member box array
            box_row = np.full((K, 7), np.nan)
            for m_idx in frame["member_ids"][p]:
                box_row[m_idx] = frame["boxes_lidar"][p]  # approx: use mean box
            all_boxes.append(box_row)

            all_frame_ids.append(frame["frame_id"])

    if len(all_scores) == 0:
        return {
            "scores": np.zeros((0, K)),
            "boxes": np.zeros((0, K, 7)),
            "frame_ids": np.array([]),
        }

    return {
        "scores": np.array(all_scores),
        "boxes": np.array(all_boxes),
        "frame_ids": np.array(all_frame_ids),
    }

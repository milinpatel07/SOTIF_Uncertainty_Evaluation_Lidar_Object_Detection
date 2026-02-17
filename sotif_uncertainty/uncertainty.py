"""
Stage 2: Cross-Member Association and Uncertainty Indicators.

Computes three uncertainty indicators per proposal from ensemble outputs:
- Mean confidence (Eq. 1): existence uncertainty
- Confidence variance (Eq. 2): epistemic uncertainty
- Geometric disagreement (Eq. 3): localisation uncertainty

Reference: Section 3.3 of the paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


def compute_mean_confidence(scores: np.ndarray) -> np.ndarray:
    """
    Compute mean confidence across ensemble members (Eq. 1).

    s_bar_j = (1/K) * sum_{k=1}^{K} s_j^{(k)}

    Parameters
    ----------
    scores : np.ndarray, shape (M, K)
        Confidence scores for M proposals from K ensemble members.
        Score is 0 if a member did not detect the proposal.

    Returns
    -------
    np.ndarray, shape (M,)
        Mean confidence per proposal.
    """
    return np.mean(scores, axis=1)


def compute_confidence_variance(scores: np.ndarray) -> np.ndarray:
    """
    Compute confidence variance across ensemble members (Eq. 2).

    sigma^2_{s,j} = (1/(K-1)) * sum_{k=1}^{K} (s_j^{(k)} - s_bar_j)^2

    High variance indicates epistemic uncertainty: members disagree,
    suggesting the input lies in a feature-space region with insufficient
    training data coverage.

    Parameters
    ----------
    scores : np.ndarray, shape (M, K)
        Confidence scores for M proposals from K ensemble members.

    Returns
    -------
    np.ndarray, shape (M,)
        Confidence variance per proposal (unbiased, ddof=1).
    """
    return np.var(scores, axis=1, ddof=1)


def compute_geometric_disagreement(boxes: np.ndarray) -> np.ndarray:
    """
    Compute geometric disagreement across ensemble members (Eq. 3).

    d_{iou,j} = 1 - (2 / (K*(K-1))) * sum_{u<v} IoU_BEV(b_j^(u), b_j^(v))

    High geometric disagreement indicates members agree the object exists
    but disagree on its position (localisation uncertainty).

    Parameters
    ----------
    boxes : np.ndarray, shape (M, K, 7)
        Bounding boxes [x, y, z, w, l, h, yaw] for M proposals from K members.
        Use NaN for members that did not detect the proposal.

    Returns
    -------
    np.ndarray, shape (M,)
        Geometric disagreement per proposal in [0, 1].
    """
    M, K, _ = boxes.shape
    disagreements = np.zeros(M)

    for j in range(M):
        valid_members = []
        for k in range(K):
            if not np.any(np.isnan(boxes[j, k])):
                valid_members.append(k)

        if len(valid_members) < 2:
            disagreements[j] = 1.0
            continue

        pair_count = 0
        iou_sum = 0.0
        for i, u in enumerate(valid_members):
            for v in valid_members[i + 1 :]:
                iou_val = _bev_iou_single(boxes[j, u], boxes[j, v])
                iou_sum += iou_val
                pair_count += 1

        mean_iou = iou_sum / pair_count if pair_count > 0 else 0.0
        disagreements[j] = 1.0 - mean_iou

    return disagreements


def _bev_iou_single(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute BEV IoU between two oriented 3D boxes.

    Uses axis-aligned approximation for efficiency. For exact oriented IoU,
    use Shapely or the OpenPCDet iou3d_nms_utils.

    Parameters
    ----------
    box_a, box_b : np.ndarray, shape (7,)
        Bounding box [x, y, z, w, l, h, yaw].

    Returns
    -------
    float
        BEV IoU in [0, 1].
    """
    # Axis-aligned BEV IoU approximation
    xa, ya, _, wa, la = box_a[0], box_a[1], box_a[2], box_a[3], box_a[4]
    xb, yb, _, wb, lb = box_b[0], box_b[1], box_b[2], box_b[3], box_b[4]

    x1_a, x2_a = xa - wa / 2, xa + wa / 2
    y1_a, y2_a = ya - la / 2, ya + la / 2
    x1_b, x2_b = xb - wb / 2, xb + wb / 2
    y1_b, y2_b = yb - lb / 2, yb + lb / 2

    inter_x = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
    inter_y = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
    inter_area = inter_x * inter_y

    area_a = wa * la
    area_b = wb * lb
    union_area = area_a + area_b - inter_area

    if union_area < 1e-6:
        return 0.0

    return inter_area / union_area


def compute_all_indicators(
    scores: np.ndarray, boxes: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    Compute all three uncertainty indicators for ensemble proposals.

    Parameters
    ----------
    scores : np.ndarray, shape (M, K)
        Confidence scores for M proposals from K ensemble members.
    boxes : np.ndarray, shape (M, K, 7), optional
        Bounding boxes. If None, geometric disagreement is not computed.

    Returns
    -------
    dict
        Keys: 'mean_confidence', 'confidence_variance', 'geometric_disagreement'.
    """
    result = {
        "mean_confidence": compute_mean_confidence(scores),
        "confidence_variance": compute_confidence_variance(scores),
    }
    if boxes is not None:
        result["geometric_disagreement"] = compute_geometric_disagreement(boxes)
    return result


def aggregate_box(boxes: np.ndarray) -> np.ndarray:
    """
    Compute aggregated bounding box by averaging parameters.

    Heading (yaw) is averaged via sine/cosine projection to handle
    angle wraparound.

    Parameters
    ----------
    boxes : np.ndarray, shape (K, 7) or (M, K, 7)
        Bounding boxes [x, y, z, w, l, h, yaw].

    Returns
    -------
    np.ndarray, shape (7,) or (M, 7)
        Aggregated bounding box(es).
    """
    if boxes.ndim == 2:
        # Single proposal, K members
        valid = ~np.any(np.isnan(boxes), axis=1)
        if not np.any(valid):
            return np.full(7, np.nan)

        valid_boxes = boxes[valid]
        agg = np.mean(valid_boxes[:, :6], axis=0)
        # Average yaw via sin/cos
        mean_sin = np.mean(np.sin(valid_boxes[:, 6]))
        mean_cos = np.mean(np.cos(valid_boxes[:, 6]))
        agg_yaw = np.arctan2(mean_sin, mean_cos)
        return np.append(agg, agg_yaw)

    elif boxes.ndim == 3:
        M = boxes.shape[0]
        result = np.zeros((M, 7))
        for j in range(M):
            result[j] = aggregate_box(boxes[j])
        return result

    else:
        raise ValueError(f"Expected 2D or 3D array, got {boxes.ndim}D")

"""
Stage 3: Correctness Determination.

Matches aggregated ensemble predictions to ground truth using greedy
assignment at BEV IoU >= 0.5. Produces TP/FP/FN labels per proposal.

Reference: Section 3.4 of the paper.
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Bird's-Eye-View IoU between two boxes.

    Uses axis-aligned approximation. Both boxes are projected onto the
    ground plane (x, y) with dimensions (w, l).

    Parameters
    ----------
    box_a : np.ndarray, shape (7,)
        [x, y, z, w, l, h, yaw]
    box_b : np.ndarray, shape (7,)
        [x, y, z, w, l, h, yaw]

    Returns
    -------
    float
        BEV IoU in [0, 1].
    """
    xa, ya, wa, la = box_a[0], box_a[1], box_a[3], box_a[4]
    xb, yb, wb, lb = box_b[0], box_b[1], box_b[3], box_b[4]

    x1_a, x2_a = xa - wa / 2, xa + wa / 2
    y1_a, y2_a = ya - la / 2, ya + la / 2
    x1_b, x2_b = xb - wb / 2, xb + wb / 2
    y1_b, y2_b = yb - lb / 2, yb + lb / 2

    inter_x = max(0.0, min(x2_a, x2_b) - max(x1_a, x1_b))
    inter_y = max(0.0, min(y2_a, y2_b) - max(y1_a, y1_b))
    inter_area = inter_x * inter_y

    area_a = wa * la
    area_b = wb * lb
    union_area = area_a + area_b - inter_area

    if union_area < 1e-6:
        return 0.0

    return float(inter_area / union_area)


def compute_bev_iou_matrix(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise BEV IoU matrix between predictions and ground truth.

    Parameters
    ----------
    pred_boxes : np.ndarray, shape (N, 7)
    gt_boxes : np.ndarray, shape (M, 7)

    Returns
    -------
    np.ndarray, shape (N, M)
        IoU matrix.
    """
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]
    iou_matrix = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_bev_iou(pred_boxes[i], gt_boxes[j])

    return iou_matrix


def greedy_match(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    Greedy matching of predictions to ground truth (Algorithm 1, lines 8-10).

    Predictions are sorted by confidence (descending) and matched to
    ground truth with one-to-one assignment at BEV IoU >= threshold.

    Parameters
    ----------
    pred_boxes : np.ndarray, shape (N, 7)
        Aggregated predicted bounding boxes [x, y, z, w, l, h, yaw].
    pred_scores : np.ndarray, shape (N,)
        Mean ensemble confidence scores.
    gt_boxes : np.ndarray, shape (M, 7)
        Ground truth bounding boxes.
    iou_threshold : float
        Minimum BEV IoU for a match (default: 0.5).

    Returns
    -------
    dict with keys:
        'labels' : np.ndarray, shape (N,)
            1 for TP, 0 for FP.
        'matched_gt' : np.ndarray, shape (N,)
            Index of matched GT box (-1 if FP).
        'unmatched_gt' : np.ndarray
            Indices of unmatched GT boxes (FN).
        'tp_count' : int
        'fp_count' : int
        'fn_count' : int
    """
    N = pred_boxes.shape[0]
    M = gt_boxes.shape[0]

    if N == 0:
        return {
            "labels": np.array([], dtype=int),
            "matched_gt": np.array([], dtype=int),
            "unmatched_gt": np.arange(M),
            "tp_count": 0,
            "fp_count": 0,
            "fn_count": M,
        }

    if M == 0:
        return {
            "labels": np.zeros(N, dtype=int),
            "matched_gt": np.full(N, -1, dtype=int),
            "unmatched_gt": np.array([], dtype=int),
            "tp_count": 0,
            "fp_count": N,
            "fn_count": 0,
        }

    # Sort predictions by confidence descending
    sort_idx = np.argsort(-pred_scores)

    # Compute IoU matrix
    iou_matrix = compute_bev_iou_matrix(pred_boxes, gt_boxes)

    labels = np.zeros(N, dtype=int)
    matched_gt_idx = np.full(N, -1, dtype=int)
    gt_matched = np.zeros(M, dtype=bool)

    for pred_idx in sort_idx:
        if M == 0:
            break

        # Find best matching GT for this prediction
        ious = iou_matrix[pred_idx]
        # Mask already matched GT
        ious_masked = ious.copy()
        ious_masked[gt_matched] = 0.0

        best_gt = np.argmax(ious_masked)
        best_iou = ious_masked[best_gt]

        if best_iou >= iou_threshold:
            labels[pred_idx] = 1  # TP
            matched_gt_idx[pred_idx] = best_gt
            gt_matched[best_gt] = True

    unmatched_gt = np.where(~gt_matched)[0]

    return {
        "labels": labels,
        "matched_gt": matched_gt_idx,
        "unmatched_gt": unmatched_gt,
        "tp_count": int(np.sum(labels)),
        "fp_count": int(np.sum(labels == 0)),
        "fn_count": int(len(unmatched_gt)),
    }

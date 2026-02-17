"""
SOTIF Uncertainty Evaluation for LiDAR Object Detection.

Evaluation methodology for determining whether prediction uncertainty from
ML-based LiDAR object detection supports SOTIF analysis per ISO 21448.

Reference:
    Patel, M. and Jung, R. (2026). "Uncertainty Evaluation to Support Safety
    of the Intended Functionality Analysis for Identifying Performance
    Insufficiencies in ML-Based LiDAR Object Detection."
"""

__version__ = "1.0.0"

from sotif_uncertainty.uncertainty import (
    compute_mean_confidence,
    compute_confidence_variance,
    compute_geometric_disagreement,
    compute_all_indicators,
)
from sotif_uncertainty.matching import greedy_match, compute_bev_iou
from sotif_uncertainty.metrics import (
    compute_auroc,
    compute_aurc,
    compute_ece,
    compute_nll,
    compute_brier,
)
from sotif_uncertainty.sotif_analysis import (
    acceptance_gate,
    compute_operating_points,
    rank_triggering_conditions,
    flag_frames,
)
from sotif_uncertainty.ensemble import (
    cluster_detections,
    clustered_to_pipeline_format,
    compute_classification_uncertainty,
    compute_regression_uncertainty,
)

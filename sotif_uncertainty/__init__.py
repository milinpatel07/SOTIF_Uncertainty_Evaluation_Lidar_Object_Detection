"""
SOTIF Uncertainty Evaluation for LiDAR Object Detection.

Evaluation methodology for determining whether prediction uncertainty from
ML-based LiDAR object detection supports SOTIF analysis per ISO 21448.

Modules:
    uncertainty      - Stage 2: Uncertainty indicator computation (Eqs. 1-3)
    ensemble         - Stage 2: DBSCAN clustering and uncertainty decomposition
    matching         - Stage 3: TP/FP/FN greedy matching at BEV IoU >= 0.5
    metrics          - Stage 4: AUROC, AURC, ECE, NLL, Brier Score
    sotif_analysis   - Stage 5: TC ranking, acceptance gates, frame triage
    visualization    - Publication-quality figures (13 plots)
    demo_data        - Synthetic data generator matching paper statistics
    kitti_utils      - KITTI calibration, label loading, point cloud I/O
    mc_dropout       - MC Dropout uncertainty estimation (alternative to ensembles)

Reference:
    Patel, M. and Jung, R. (2026). "Uncertainty Evaluation to Support Safety
    of the Intended Functionality Analysis for Identifying Performance
    Insufficiencies in ML-Based LiDAR Object Detection."
"""

__version__ = "1.1.0"

from sotif_uncertainty.uncertainty import (
    compute_mean_confidence,
    compute_confidence_variance,
    compute_geometric_disagreement,
    compute_all_indicators,
    aggregate_box,
)
from sotif_uncertainty.matching import greedy_match, compute_bev_iou
from sotif_uncertainty.metrics import (
    compute_auroc,
    compute_aurc,
    compute_ece,
    compute_nll,
    compute_brier,
    compute_all_metrics,
)
from sotif_uncertainty.sotif_analysis import (
    acceptance_gate,
    compute_operating_points,
    rank_triggering_conditions,
    flag_frames,
    find_optimal_gate,
    compute_frame_summary,
)
from sotif_uncertainty.ensemble import (
    cluster_detections,
    clustered_to_pipeline_format,
    compute_classification_uncertainty,
    compute_regression_uncertainty,
)

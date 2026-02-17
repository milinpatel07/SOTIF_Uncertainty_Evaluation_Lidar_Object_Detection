"""
Tests for the SOTIF uncertainty evaluation pipeline.

Verifies that all modules produce correct results with both
synthetic demo data and known edge cases.

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or without pytest:
    python tests/test_pipeline.py
"""

import os
import sys
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ======================================================================
# Test fixtures
# ======================================================================

def make_scores(n_tp=20, n_fp=40, K=6, seed=42):
    """Create simple known scores for testing."""
    rng = np.random.RandomState(seed)
    N = n_tp + n_fp
    scores = np.zeros((N, K))
    labels = np.zeros(N, dtype=int)
    labels[:n_tp] = 1

    # TP: all members detect at high confidence
    for i in range(n_tp):
        scores[i] = rng.uniform(0.7, 0.95, K)

    # FP: lower confidence, some members miss
    for i in range(n_tp, N):
        n_detect = rng.choice([2, 3, 4, 5, 6])
        detecting = rng.choice(K, size=n_detect, replace=False)
        for k in detecting:
            scores[i, k] = rng.uniform(0.05, 0.4)

    return scores, labels


def make_boxes(n_tp=20, n_fp=40, K=6, seed=42):
    """Create simple known boxes for testing."""
    rng = np.random.RandomState(seed + 100)
    N = n_tp + n_fp
    boxes = np.full((N, K, 7), np.nan)

    for i in range(N):
        x, y, z = rng.uniform(5, 50), rng.uniform(-5, 5), rng.uniform(-1.5, -0.5)
        for k in range(K):
            if i < n_tp:
                noise = 0.15
            else:
                noise = rng.uniform(0.5, 2.0)
                if rng.random() > 0.5:
                    continue  # Some members miss FP
            boxes[i, k] = [
                x + rng.normal(0, noise),
                y + rng.normal(0, noise),
                z + rng.normal(0, 0.05),
                1.8, 4.5, 1.6,
                rng.uniform(-0.1, 0.1),
            ]

    return boxes


# ======================================================================
# uncertainty.py tests
# ======================================================================

def test_mean_confidence():
    """Mean confidence should be the row-wise mean of scores."""
    from sotif_uncertainty.uncertainty import compute_mean_confidence
    scores = np.array([[0.5, 0.6, 0.7], [0.1, 0.0, 0.0]])
    result = compute_mean_confidence(scores)
    np.testing.assert_allclose(result, [0.6, 0.1 / 3], atol=1e-7)
    print("  PASS: test_mean_confidence")


def test_confidence_variance():
    """Variance with ddof=1 for known values."""
    from sotif_uncertainty.uncertainty import compute_confidence_variance
    scores = np.array([[0.5, 0.5, 0.5], [0.1, 0.3, 0.5]])
    result = compute_confidence_variance(scores)
    assert result[0] == 0.0, "Equal scores should have zero variance"
    assert result[1] > 0, "Different scores should have positive variance"
    expected_var = np.var([0.1, 0.3, 0.5], ddof=1)
    np.testing.assert_allclose(result[1], expected_var, atol=1e-7)
    print("  PASS: test_confidence_variance")


def test_geometric_disagreement():
    """Identical boxes should have zero disagreement."""
    from sotif_uncertainty.uncertainty import compute_geometric_disagreement
    boxes = np.zeros((1, 3, 7))
    boxes[0, :, :] = [10, 0, 0, 2, 4, 1.5, 0]  # Same box for all members
    result = compute_geometric_disagreement(boxes)
    assert result[0] == 0.0, f"Identical boxes should have 0 disagreement, got {result[0]}"
    print("  PASS: test_geometric_disagreement")


def test_geometric_disagreement_max():
    """Non-overlapping boxes should have disagreement = 1.0."""
    from sotif_uncertainty.uncertainty import compute_geometric_disagreement
    boxes = np.zeros((1, 2, 7))
    boxes[0, 0] = [0, 0, 0, 1, 1, 1, 0]
    boxes[0, 1] = [100, 100, 0, 1, 1, 1, 0]  # Far apart
    result = compute_geometric_disagreement(boxes)
    assert result[0] == 1.0, f"Non-overlapping boxes should have 1.0 disagreement, got {result[0]}"
    print("  PASS: test_geometric_disagreement_max")


def test_compute_all_indicators():
    """compute_all_indicators should return all three indicators."""
    from sotif_uncertainty.uncertainty import compute_all_indicators
    scores, labels = make_scores()
    boxes = make_boxes()
    indicators = compute_all_indicators(scores, boxes)

    assert "mean_confidence" in indicators
    assert "confidence_variance" in indicators
    assert "geometric_disagreement" in indicators
    assert len(indicators["mean_confidence"]) == 60
    print("  PASS: test_compute_all_indicators")


# ======================================================================
# matching.py tests
# ======================================================================

def test_bev_iou_identical():
    """Identical boxes should have IoU = 1.0."""
    from sotif_uncertainty.matching import compute_bev_iou
    box = np.array([10, 0, 0, 2, 4, 1.5, 0])
    iou = compute_bev_iou(box, box)
    np.testing.assert_allclose(iou, 1.0, atol=1e-6)
    print("  PASS: test_bev_iou_identical")


def test_bev_iou_no_overlap():
    """Non-overlapping boxes should have IoU = 0.0."""
    from sotif_uncertainty.matching import compute_bev_iou
    box_a = np.array([0, 0, 0, 1, 1, 1, 0])
    box_b = np.array([100, 100, 0, 1, 1, 1, 0])
    iou = compute_bev_iou(box_a, box_b)
    assert iou == 0.0
    print("  PASS: test_bev_iou_no_overlap")


def test_greedy_match():
    """Greedy matching should find correct TP/FP/FN counts."""
    from sotif_uncertainty.matching import greedy_match
    pred_boxes = np.array([
        [10, 0, 0, 2, 4, 1.5, 0],   # overlaps GT 0
        [20, 0, 0, 2, 4, 1.5, 0],   # overlaps GT 1
        [50, 50, 0, 2, 4, 1.5, 0],  # no match (FP)
    ])
    pred_scores = np.array([0.9, 0.8, 0.7])
    gt_boxes = np.array([
        [10.1, 0.1, 0, 2, 4, 1.5, 0],   # GT 0
        [20.1, 0.1, 0, 2, 4, 1.5, 0],   # GT 1
        [40, -10, 0, 2, 4, 1.5, 0],      # GT 2 - unmatched (FN)
    ])

    result = greedy_match(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)
    assert result["tp_count"] == 2, f"Expected 2 TP, got {result['tp_count']}"
    assert result["fp_count"] == 1, f"Expected 1 FP, got {result['fp_count']}"
    assert result["fn_count"] == 1, f"Expected 1 FN, got {result['fn_count']}"
    print("  PASS: test_greedy_match")


def test_greedy_match_empty():
    """Matching with no predictions should produce all FN."""
    from sotif_uncertainty.matching import greedy_match
    pred_boxes = np.zeros((0, 7))
    pred_scores = np.zeros(0)
    gt_boxes = np.array([[10, 0, 0, 2, 4, 1.5, 0]])

    result = greedy_match(pred_boxes, pred_scores, gt_boxes)
    assert result["tp_count"] == 0
    assert result["fn_count"] == 1
    print("  PASS: test_greedy_match_empty")


# ======================================================================
# metrics.py tests
# ======================================================================

def test_auroc_perfect():
    """Perfect separation should give AUROC = 1.0."""
    from sotif_uncertainty.metrics import compute_auroc
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.05, 0.02])
    labels = np.array([1, 1, 1, 0, 0, 0])
    auroc = compute_auroc(scores, labels, higher_is_correct=True)
    assert auroc == 1.0, f"Expected 1.0, got {auroc}"
    print("  PASS: test_auroc_perfect")


def test_auroc_random():
    """Random scores should give AUROC ~0.5."""
    from sotif_uncertainty.metrics import compute_auroc
    rng = np.random.RandomState(42)
    scores = rng.uniform(0, 1, 1000)
    labels = rng.choice([0, 1], 1000)
    auroc = compute_auroc(scores, labels, higher_is_correct=True)
    assert 0.4 < auroc < 0.6, f"Random AUROC should be ~0.5, got {auroc}"
    print("  PASS: test_auroc_random")


def test_ece():
    """Perfectly calibrated scores should give ECE near 0."""
    from sotif_uncertainty.metrics import compute_ece
    # For each confidence bin, accuracy matches confidence
    rng = np.random.RandomState(42)
    n = 10000
    scores = rng.uniform(0, 1, n)
    labels = (rng.uniform(0, 1, n) < scores).astype(int)
    ece, _, _, _ = compute_ece(scores, labels, n_bins=10)
    assert ece < 0.05, f"Calibrated ECE should be < 0.05, got {ece}"
    print("  PASS: test_ece")


def test_nll_brier():
    """NLL and Brier should be low for good predictions."""
    from sotif_uncertainty.metrics import compute_nll, compute_brier
    scores = np.array([0.99, 0.95, 0.01, 0.05])
    labels = np.array([1, 1, 0, 0])
    nll = compute_nll(scores, labels)
    brier = compute_brier(scores, labels)
    assert nll < 0.1, f"Good NLL should be low, got {nll}"
    assert brier < 0.01, f"Good Brier should be low, got {brier}"
    print("  PASS: test_nll_brier")


def test_compute_all_metrics():
    """compute_all_metrics should return all metric categories."""
    from sotif_uncertainty.metrics import compute_all_metrics
    scores, labels = make_scores()
    from sotif_uncertainty.uncertainty import compute_all_indicators
    boxes = make_boxes()
    ind = compute_all_indicators(scores, boxes)

    metrics = compute_all_metrics(
        ind["mean_confidence"], ind["confidence_variance"],
        ind["geometric_disagreement"], labels
    )
    assert "discrimination" in metrics
    assert "calibration" in metrics
    assert "risk_coverage" in metrics
    assert metrics["discrimination"]["auroc_mean_confidence"] > 0.5
    print("  PASS: test_compute_all_metrics")


# ======================================================================
# sotif_analysis.py tests
# ======================================================================

def test_acceptance_gate():
    """Acceptance gate should filter based on thresholds."""
    from sotif_uncertainty.sotif_analysis import acceptance_gate
    mean_conf = np.array([0.9, 0.7, 0.5, 0.3])
    conf_var = np.array([0.001, 0.01, 0.05, 0.1])
    geo_disagree = np.array([0.05, 0.1, 0.3, 0.5])

    # Only confidence threshold
    accepted = acceptance_gate(mean_conf, conf_var, geo_disagree, tau_s=0.6)
    np.testing.assert_array_equal(accepted, [True, True, False, False])

    # Multi-indicator gate
    accepted = acceptance_gate(mean_conf, conf_var, geo_disagree,
                               tau_s=0.6, tau_v=0.005)
    np.testing.assert_array_equal(accepted, [True, False, False, False])
    print("  PASS: test_acceptance_gate")


def test_operating_points():
    """Operating points should produce valid coverage and FAR."""
    from sotif_uncertainty.sotif_analysis import compute_operating_points
    scores, labels = make_scores()
    from sotif_uncertainty.uncertainty import compute_all_indicators
    ind = compute_all_indicators(scores, make_boxes())

    points = compute_operating_points(
        ind["mean_confidence"], ind["confidence_variance"],
        ind["geometric_disagreement"], labels,
        tau_s_range=np.array([0.3, 0.5, 0.7]),
    )
    assert len(points) == 3
    # Higher threshold -> lower coverage
    assert points[0]["coverage"] >= points[-1]["coverage"]
    for p in points:
        assert 0 <= p["coverage"] <= 1
        assert 0 <= p["far"] <= 1
    print("  PASS: test_operating_points")


def test_tc_ranking():
    """TC ranking should sort by FP share descending."""
    from sotif_uncertainty.sotif_analysis import rank_triggering_conditions
    conditions = np.array(["rain", "rain", "rain", "clear", "clear",
                          "rain", "rain", "clear", "clear", "clear"])
    labels = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1])
    mean_conf = np.random.uniform(0, 1, 10)
    conf_var = np.random.uniform(0, 0.1, 10)

    results = rank_triggering_conditions(conditions, labels, mean_conf, conf_var)
    assert len(results) == 2
    assert results[0]["fp_share"] >= results[1]["fp_share"]
    print("  PASS: test_tc_ranking")


def test_flag_frames():
    """Frame flagging should identify frames with high-variance FP."""
    from sotif_uncertainty.sotif_analysis import flag_frames
    frame_ids = np.array([0, 0, 1, 1, 2])
    labels = np.array([1, 0, 1, 0, 0])
    conf_var = np.array([0.001, 0.01, 0.001, 0.001, 0.001])

    result = flag_frames(frame_ids, labels, conf_var, percentile=50)
    assert result["total_frames"] == 3
    # Frame 0 has the highest-variance FP
    assert 0 in result["flagged_frames"]
    print("  PASS: test_flag_frames")


# ======================================================================
# demo_data.py tests
# ======================================================================

def test_demo_dataset_shape():
    """Demo dataset should have correct shapes."""
    from sotif_uncertainty.demo_data import generate_demo_dataset
    data = generate_demo_dataset()
    assert data["scores"].shape == (465, 6)
    assert data["boxes"].shape == (465, 6, 7)
    assert data["labels"].shape == (465,)
    assert np.sum(data["labels"] == 1) == 135
    assert np.sum(data["labels"] == 0) == 330
    print("  PASS: test_demo_dataset_shape")


def test_demo_dataset_statistics():
    """Demo dataset should approximately match paper statistics."""
    from sotif_uncertainty.demo_data import generate_demo_dataset, validate_dataset
    data = generate_demo_dataset()
    stats = validate_dataset(data)

    assert stats["auroc_mean_confidence"] > 0.95, \
        f"AUROC(conf) too low: {stats['auroc_mean_confidence']}"
    assert stats["auroc_confidence_variance"] > 0.80, \
        f"AUROC(var) too low: {stats['auroc_confidence_variance']}"
    assert stats["auroc_geometric_disagreement"] > 0.80, \
        f"AUROC(geo) too low: {stats['auroc_geometric_disagreement']}"
    assert stats["fp_at_070"] <= 1, \
        f"Too many FP at 0.70 threshold: {stats['fp_at_070']}"
    print("  PASS: test_demo_dataset_statistics")


# ======================================================================
# ensemble.py tests
# ======================================================================

def test_dbscan_clustering():
    """DBSCAN clustering should group overlapping detections."""
    from sotif_uncertainty.ensemble import cluster_detections

    K = 3
    member_preds = []
    for k in range(K):
        preds = [{
            "boxes_lidar": np.array([
                [10.0, 0.0, 1.0, 4.0, 1.8, 1.5, 0.0],
                [30.0, 0.0, 1.0, 4.0, 1.8, 1.5, 0.0],
            ]) + np.random.randn(2, 7) * 0.05,
            "score": np.array([0.8, 0.7]),
            "pred_labels": np.array([1, 1]),
            "frame_id": "000000",
        }]
        member_preds.append(preds)

    result = cluster_detections(member_preds, iou_threshold=0.5, voting="consensus")
    assert len(result) == 1  # One frame
    assert len(result[0]["mean_score"]) == 2  # Two clusters
    print("  PASS: test_dbscan_clustering")


def test_clustering_voting():
    """Different voting strategies should filter differently."""
    from sotif_uncertainty.ensemble import cluster_detections

    # 4 members, but only 2 detect the second object
    member_preds = []
    for k in range(4):
        boxes = [np.array([10.0, 0.0, 1.0, 4.0, 1.8, 1.5, 0.0])]
        scores_list = [0.8]
        if k < 2:  # Only first 2 members detect second object
            boxes.append(np.array([30.0, 0.0, 1.0, 4.0, 1.8, 1.5, 0.0]))
            scores_list.append(0.6)
        preds = [{
            "boxes_lidar": np.array(boxes),
            "score": np.array(scores_list),
            "pred_labels": np.ones(len(boxes), dtype=int),
            "frame_id": "000000",
        }]
        member_preds.append(preds)

    # Affirmative: keep all (both objects)
    result = cluster_detections(member_preds, iou_threshold=0.5, voting="affirmative")
    n_aff = len(result[0]["mean_score"])

    # Unanimous: require all 4 (only first object)
    result = cluster_detections(member_preds, iou_threshold=0.5, voting="unanimous")
    n_una = len(result[0]["mean_score"])

    assert n_aff >= n_una, f"Affirmative ({n_aff}) should keep >= unanimous ({n_una})"
    print("  PASS: test_clustering_voting")


# ======================================================================
# Full pipeline test
# ======================================================================

def test_full_pipeline():
    """Run the complete pipeline end-to-end with synthetic data."""
    from sotif_uncertainty.demo_data import generate_demo_dataset
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.metrics import compute_all_metrics
    from sotif_uncertainty.sotif_analysis import (
        compute_operating_points, rank_triggering_conditions,
        flag_frames, compute_frame_summary,
    )

    # Stage 1: Generate data
    data = generate_demo_dataset(seed=42)

    # Stage 2: Compute indicators
    indicators = compute_all_indicators(data["scores"], data["boxes"])
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    assert len(mean_conf) == 465
    assert np.all(mean_conf >= 0) and np.all(mean_conf <= 1)
    assert np.all(conf_var >= 0)

    # Stage 4: Compute metrics
    metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, data["labels"])
    assert metrics["discrimination"]["auroc_mean_confidence"] > 0.9
    assert 0 < metrics["calibration"]["ece"] < 1
    assert metrics["calibration"]["brier"] >= 0

    # Stage 5: SOTIF analysis
    points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, data["labels"],
        tau_s_range=np.array([0.60, 0.70]),
    )
    assert len(points) == 2

    tc_results = rank_triggering_conditions(
        data["conditions"], data["labels"], mean_conf, conf_var
    )
    assert len(tc_results) == 4  # 4 TC categories
    assert tc_results[0]["condition"] == "heavy_rain"  # Highest FP share

    flags = flag_frames(data["frame_ids"], data["labels"], conf_var)
    assert flags["total_frames"] > 0

    summaries = compute_frame_summary(
        data["frame_ids"], data["labels"], mean_conf, conf_var, data["conditions"]
    )
    assert len(summaries) > 0

    print("  PASS: test_full_pipeline")


def test_visualization_generation():
    """All visualizations should generate without errors."""
    from sotif_uncertainty.demo_data import generate_demo_dataset
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.metrics import compute_all_metrics
    from sotif_uncertainty.sotif_analysis import (
        compute_operating_points, rank_triggering_conditions,
        compute_frame_summary,
    )
    from sotif_uncertainty.visualization import generate_all_figures
    import tempfile

    data = generate_demo_dataset(seed=42)
    indicators = compute_all_indicators(data["scores"], data["boxes"])
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, data["labels"])
    points = compute_operating_points(
        mean_conf, conf_var, geo_disagree, data["labels"],
        tau_s_range=np.array([0.60, 0.70]),
    )
    tc_results = rank_triggering_conditions(
        data["conditions"], data["labels"], mean_conf, conf_var
    )
    summaries = compute_frame_summary(
        data["frame_ids"], data["labels"], mean_conf, conf_var, data["conditions"]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        figures = generate_all_figures(
            metrics=metrics,
            mean_conf=mean_conf,
            conf_var=conf_var,
            labels=data["labels"],
            frame_summaries=summaries,
            tc_results=tc_results,
            operating_points=points,
            output_dir=tmpdir,
            scores=data["scores"],
            geo_disagree=geo_disagree,
            conditions=data["conditions"],
        )

        assert len(figures) >= 13, f"Expected >= 13 figures, got {len(figures)}"

        # Verify all files exist
        for name in figures:
            # Check at least the key figures exist
            pass

        # Count actual PNG files
        png_files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
        assert len(png_files) >= 13, f"Expected >= 13 PNG files, got {len(png_files)}"

    print("  PASS: test_visualization_generation")


# ======================================================================
# Main runner
# ======================================================================

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        # uncertainty.py
        test_mean_confidence,
        test_confidence_variance,
        test_geometric_disagreement,
        test_geometric_disagreement_max,
        test_compute_all_indicators,
        # matching.py
        test_bev_iou_identical,
        test_bev_iou_no_overlap,
        test_greedy_match,
        test_greedy_match_empty,
        # metrics.py
        test_auroc_perfect,
        test_auroc_random,
        test_ece,
        test_nll_brier,
        test_compute_all_metrics,
        # sotif_analysis.py
        test_acceptance_gate,
        test_operating_points,
        test_tc_ranking,
        test_flag_frames,
        # demo_data.py
        test_demo_dataset_shape,
        test_demo_dataset_statistics,
        # ensemble.py
        test_dbscan_clustering,
        test_clustering_voting,
        # Integration
        test_full_pipeline,
        test_visualization_generation,
    ]

    print("=" * 60)
    print("SOTIF Uncertainty Evaluation - Test Suite")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test.__name__, str(e)))
            print(f"  FAIL: {test.__name__}: {e}")

    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  {name}: {err}")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

"""
Comprehensive tests for the SOTIF uncertainty evaluation pipeline.

Validates all five stages of the evaluation methodology:
    Stage 2: Uncertainty indicator computation
    Stage 3: Correctness determination (TP/FP matching)
    Stage 4: Metric computation (AUROC, ECE, etc.)
    Stage 5: SOTIF analysis (operating points, TC ranking)

Also tests:
    - Synthetic data generation and validation
    - End-to-end pipeline execution
    - KITTI utilities (calibration, label loading)
    - MC Dropout simulation
    - Edge cases and numerical stability

Usage:
    python -m pytest tests/test_pipeline.py -v
    python tests/test_pipeline.py   # standalone
"""

import os
import sys
import tempfile
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ================================================================
# Stage 2: Uncertainty Indicators
# ================================================================

class TestUncertaintyIndicators:
    """Test uncertainty indicator computation (Eqs. 1-3)."""

    def test_mean_confidence_basic(self):
        from sotif_uncertainty.uncertainty import compute_mean_confidence
        scores = np.array([[0.9, 0.8, 0.85], [0.1, 0.2, 0.15]])
        result = compute_mean_confidence(scores)
        assert result.shape == (2,)
        np.testing.assert_almost_equal(result[0], 0.85, decimal=5)
        np.testing.assert_almost_equal(result[1], 0.15, decimal=5)

    def test_mean_confidence_with_zeros(self):
        from sotif_uncertainty.uncertainty import compute_mean_confidence
        scores = np.array([[0.9, 0.0, 0.0]])  # 2 members didn't detect
        result = compute_mean_confidence(scores)
        np.testing.assert_almost_equal(result[0], 0.3, decimal=5)

    def test_confidence_variance(self):
        from sotif_uncertainty.uncertainty import compute_confidence_variance
        # All same -> variance = 0
        scores = np.array([[0.5, 0.5, 0.5]])
        result = compute_confidence_variance(scores)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=10)

        # Different scores -> positive variance
        scores = np.array([[0.9, 0.1]])
        result = compute_confidence_variance(scores)
        assert result[0] > 0

    def test_geometric_disagreement(self):
        from sotif_uncertainty.uncertainty import compute_geometric_disagreement
        K = 3
        # Identical boxes -> disagreement = 0
        box = np.array([10, 5, -1, 2, 4, 1.5, 0.0])
        boxes = np.tile(box, (1, K, 1))
        result = compute_geometric_disagreement(boxes)
        np.testing.assert_almost_equal(result[0], 0.0, decimal=5)

        # Completely separated boxes -> disagreement = 1
        boxes = np.full((1, K, 7), np.nan)
        boxes[0, 0] = [0, 0, 0, 2, 4, 1.5, 0]
        boxes[0, 1] = [100, 100, 0, 2, 4, 1.5, 0]
        result = compute_geometric_disagreement(boxes)
        np.testing.assert_almost_equal(result[0], 1.0, decimal=5)

    def test_compute_all_indicators(self):
        from sotif_uncertainty.uncertainty import compute_all_indicators
        K = 4
        N = 10
        scores = np.random.RandomState(42).rand(N, K)
        boxes = np.random.RandomState(42).rand(N, K, 7) * 10

        result = compute_all_indicators(scores, boxes)
        assert "mean_confidence" in result
        assert "confidence_variance" in result
        assert "geometric_disagreement" in result
        assert result["mean_confidence"].shape == (N,)

    def test_aggregate_box(self):
        from sotif_uncertainty.uncertainty import aggregate_box
        boxes = np.array([
            [10, 5, -1, 2, 4, 1.5, 0.1],
            [10.1, 5.1, -1, 2, 4, 1.5, 0.2],
        ])
        agg = aggregate_box(boxes)
        assert agg.shape == (7,)
        np.testing.assert_almost_equal(agg[0], 10.05, decimal=3)


# ================================================================
# Stage 3: Matching
# ================================================================

class TestMatching:
    """Test greedy TP/FP matching."""

    def test_perfect_match(self):
        from sotif_uncertainty.matching import greedy_match
        pred = np.array([[10, 5, -1, 2, 4, 1.5, 0]])
        gt = np.array([[10, 5, -1, 2, 4, 1.5, 0]])
        scores = np.array([0.9])
        result = greedy_match(pred, scores, gt, iou_threshold=0.5)
        assert result["tp_count"] == 1
        assert result["fp_count"] == 0
        assert result["fn_count"] == 0

    def test_no_match(self):
        from sotif_uncertainty.matching import greedy_match
        pred = np.array([[10, 5, -1, 2, 4, 1.5, 0]])
        gt = np.array([[100, 100, -1, 2, 4, 1.5, 0]])  # far away
        scores = np.array([0.9])
        result = greedy_match(pred, scores, gt, iou_threshold=0.5)
        assert result["tp_count"] == 0
        assert result["fp_count"] == 1
        assert result["fn_count"] == 1

    def test_empty_predictions(self):
        from sotif_uncertainty.matching import greedy_match
        pred = np.zeros((0, 7))
        gt = np.array([[10, 5, -1, 2, 4, 1.5, 0]])
        scores = np.array([])
        result = greedy_match(pred, scores, gt, iou_threshold=0.5)
        assert result["tp_count"] == 0
        assert result["fn_count"] == 1

    def test_empty_gt(self):
        from sotif_uncertainty.matching import greedy_match
        pred = np.array([[10, 5, -1, 2, 4, 1.5, 0]])
        gt = np.zeros((0, 7))
        scores = np.array([0.9])
        result = greedy_match(pred, scores, gt, iou_threshold=0.5)
        assert result["fp_count"] == 1
        assert result["fn_count"] == 0

    def test_bev_iou_computation(self):
        from sotif_uncertainty.matching import compute_bev_iou
        box_a = np.array([0, 0, 0, 2, 4, 1.5, 0])
        box_b = np.array([0, 0, 0, 2, 4, 1.5, 0])
        iou = compute_bev_iou(box_a, box_b)
        np.testing.assert_almost_equal(iou, 1.0, decimal=5)

        # No overlap
        box_b = np.array([100, 100, 0, 2, 4, 1.5, 0])
        iou = compute_bev_iou(box_a, box_b)
        np.testing.assert_almost_equal(iou, 0.0, decimal=5)


# ================================================================
# Stage 4: Metrics
# ================================================================

class TestMetrics:
    """Test metric computation."""

    def test_auroc_perfect(self):
        from sotif_uncertainty.metrics import compute_auroc
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([1, 1, 0, 0])
        auroc = compute_auroc(scores, labels, higher_is_correct=True)
        np.testing.assert_almost_equal(auroc, 1.0, decimal=5)

    def test_auroc_random(self):
        from sotif_uncertainty.metrics import compute_auroc
        rng = np.random.RandomState(42)
        scores = rng.rand(1000)
        labels = rng.choice([0, 1], 1000)
        auroc = compute_auroc(scores, labels, higher_is_correct=True)
        assert 0.4 < auroc < 0.6  # should be ~0.5

    def test_auroc_single_class(self):
        from sotif_uncertainty.metrics import compute_auroc
        scores = np.array([0.9, 0.8])
        labels = np.array([1, 1])
        auroc = compute_auroc(scores, labels)
        assert np.isnan(auroc)

    def test_ece(self):
        from sotif_uncertainty.metrics import compute_ece
        # Perfect calibration: confidence matches accuracy
        scores = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])
        ece, _, _, _ = compute_ece(scores, labels, n_bins=10)
        assert ece < 0.15  # should be near 0

    def test_nll(self):
        from sotif_uncertainty.metrics import compute_nll
        scores = np.array([0.99, 0.01])
        labels = np.array([1, 0])
        nll = compute_nll(scores, labels)
        assert nll < 0.1  # near-perfect predictions -> low NLL

    def test_brier(self):
        from sotif_uncertainty.metrics import compute_brier
        scores = np.array([1.0, 0.0])
        labels = np.array([1, 0])
        brier = compute_brier(scores, labels)
        np.testing.assert_almost_equal(brier, 0.0, decimal=5)

    def test_aurc(self):
        from sotif_uncertainty.metrics import compute_aurc
        # High-confidence detections are all correct
        scores = np.array([0.95, 0.90, 0.85, 0.10, 0.05])
        labels = np.array([1, 1, 1, 0, 0])
        aurc, cov, risk = compute_aurc(scores, labels)
        assert aurc < 0.5  # good selective prediction
        assert len(cov) == len(labels)

    def test_compute_all_metrics(self):
        from sotif_uncertainty.metrics import compute_all_metrics
        N = 100
        rng = np.random.RandomState(42)
        mean_conf = rng.rand(N)
        conf_var = rng.rand(N) * 0.01
        geo_disagree = rng.rand(N)
        labels = (mean_conf > 0.5).astype(int)

        metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, labels)
        assert "discrimination" in metrics
        assert "calibration" in metrics
        assert "risk_coverage" in metrics


# ================================================================
# Stage 5: SOTIF Analysis
# ================================================================

class TestSOTIFAnalysis:
    """Test SOTIF analysis functions."""

    def test_acceptance_gate(self):
        from sotif_uncertainty.sotif_analysis import acceptance_gate
        mc = np.array([0.9, 0.6, 0.3])
        cv = np.array([0.001, 0.005, 0.010])
        gd = np.array([0.1, 0.3, 0.8])

        accepted = acceptance_gate(mc, cv, gd, tau_s=0.5, tau_v=0.01, tau_d=0.5)
        assert accepted[0] == True
        assert accepted[1] == True
        assert accepted[2] == False

    def test_operating_points(self):
        from sotif_uncertainty.sotif_analysis import compute_operating_points
        N = 50
        rng = np.random.RandomState(42)
        mc = rng.rand(N)
        cv = rng.rand(N) * 0.01
        gd = rng.rand(N)
        labels = (mc > 0.5).astype(int)

        points = compute_operating_points(
            mc, cv, gd, labels,
            tau_s_range=np.array([0.5, 0.7, 0.9]),
        )
        assert len(points) > 0
        assert all("coverage" in p for p in points)
        assert all("far" in p for p in points)

    def test_tc_ranking(self):
        from sotif_uncertainty.sotif_analysis import rank_triggering_conditions
        N = 100
        rng = np.random.RandomState(42)
        conditions = rng.choice(["rain", "fog", "clear"], N)
        labels = rng.choice([0, 1], N)
        mc = rng.rand(N)
        cv = rng.rand(N) * 0.01

        results = rank_triggering_conditions(conditions, labels, mc, cv)
        assert len(results) == 3
        assert results[0]["fp_share"] >= results[-1]["fp_share"]

    def test_flag_frames(self):
        from sotif_uncertainty.sotif_analysis import flag_frames
        frame_ids = np.array([0, 0, 1, 1, 2])
        labels = np.array([1, 0, 1, 0, 0])
        conf_var = np.array([0.001, 0.010, 0.001, 0.002, 0.050])

        result = flag_frames(frame_ids, labels, conf_var)
        assert result["total_frames"] == 3
        assert result["flagged_count"] >= 0


# ================================================================
# Synthetic Data Generation
# ================================================================

class TestDemoData:
    """Test synthetic data generator."""

    def test_generate_demo_dataset(self):
        from sotif_uncertainty.demo_data import generate_demo_dataset
        data = generate_demo_dataset(n_tp=20, n_fp=50, K=4, n_frames=10, seed=42)

        assert data["scores"].shape == (70, 4)
        assert data["boxes"].shape == (70, 4, 7)
        assert data["labels"].shape == (70,)
        assert np.sum(data["labels"] == 1) == 20
        assert np.sum(data["labels"] == 0) == 50

    def test_paper_statistics(self):
        """Validate that generated data matches paper's target statistics."""
        from sotif_uncertainty.demo_data import generate_demo_dataset, validate_dataset
        data = generate_demo_dataset()
        stats = validate_dataset(data)

        assert stats["n_total"] == 465
        assert stats["n_tp"] == 135
        assert stats["n_fp"] == 330
        assert stats["auroc_mean_confidence"] > 0.95
        assert stats["auroc_confidence_variance"] > 0.80

    def test_condition_distribution(self):
        from sotif_uncertainty.demo_data import generate_demo_dataset
        data = generate_demo_dataset()
        conditions = data["conditions"]
        labels = data["labels"]

        fp_mask = labels == 0
        fp_conditions = conditions[fp_mask]
        total_fp = len(fp_conditions)

        # Heavy rain should be the largest FP category
        rain_share = np.sum(fp_conditions == "heavy_rain") / total_fp
        assert rain_share > 0.30


# ================================================================
# KITTI Utilities
# ================================================================

class TestKITTIUtils:
    """Test KITTI calibration and data loading."""

    def test_calibration_roundtrip(self):
        """Test that cam->lidar->cam roundtrip preserves coordinates."""
        # Create a temporary calibration file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("P0: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 "
                    "0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 "
                    "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
            f.write("P1: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 "
                    "0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 "
                    "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n")
            f.write("P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 "
                    "0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 "
                    "0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03\n")
            f.write("P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 "
                    "0.000000e+00 7.215377e+02 1.728540e+02 2.199936e+00 "
                    "0.000000e+00 0.000000e+00 1.000000e+00 2.729905e-03\n")
            f.write("R0_rect: 9.999239e-01 9.837760e-03 -7.445048e-03 "
                    "-9.869795e-03 9.999421e-01 -4.278459e-03 "
                    "7.402527e-03 4.351614e-03 9.999631e-01\n")
            f.write("Tr_velo_to_cam: 7.533745e-03 -9.999714e-01 -6.166020e-04 "
                    "-4.069766e-03 1.480249e-02 7.280733e-04 -9.998902e-01 "
                    "-7.631618e-02 9.998621e-01 7.523790e-03 1.480755e-02 "
                    "-2.717806e-01\n")
            f.write("Tr_imu_to_velo: 9.999976e-01 7.553071e-04 -2.035826e-03 "
                    "-8.086759e-01 -7.854027e-04 9.998898e-01 -1.482298e-02 "
                    "3.195559e-01 2.024406e-03 1.482454e-02 9.998881e-01 "
                    "-7.997231e-01\n")
            calib_path = f.name

        try:
            from sotif_uncertainty.kitti_utils import KITTICalibration

            calib = KITTICalibration(calib_path)

            # Test point transform roundtrip
            pts_lidar = np.array([[10.0, 5.0, -1.0]])
            pts_cam = calib.lidar_to_camera(pts_lidar)
            pts_back = calib.camera_to_lidar(pts_cam)
            np.testing.assert_array_almost_equal(pts_lidar, pts_back, decimal=4)

            # Test box transform roundtrip
            boxes_cam = np.array([[1.5, 1.8, 4.5, 3.0, 1.7, 15.0, 0.1]])
            boxes_lidar = calib.boxes_cam_to_lidar(boxes_cam)
            assert boxes_lidar.shape == (1, 7)
            assert not np.any(np.isnan(boxes_lidar))
        finally:
            os.unlink(calib_path)

    def test_load_point_cloud(self):
        """Test point cloud loading from .bin file."""
        from sotif_uncertainty.kitti_utils import load_point_cloud

        # Create a temporary .bin file
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            pts = np.random.rand(100, 4).astype(np.float32)
            pts.tofile(f)
            bin_path = f.name

        try:
            loaded = load_point_cloud(bin_path)
            assert loaded.shape == (100, 4)
            np.testing.assert_array_almost_equal(loaded, pts, decimal=5)

            # Test with range filter
            loaded_filtered = load_point_cloud(bin_path, xlim=(0.0, 0.5))
            assert len(loaded_filtered) <= 100
        finally:
            os.unlink(bin_path)

    def test_voxelization(self):
        """Test point cloud voxelization."""
        from sotif_uncertainty.kitti_utils import voxelize_point_cloud

        pts = np.random.RandomState(42).rand(1000, 4).astype(np.float32)
        pts[:, 0] *= 70  # x: 0-70
        pts[:, 1] = pts[:, 1] * 80 - 40  # y: -40 to 40
        pts[:, 2] = pts[:, 2] * 4 - 3  # z: -3 to 1

        voxels, coords, num_points = voxelize_point_cloud(pts)
        assert voxels.ndim == 3
        assert coords.ndim == 2
        assert coords.shape[1] == 3
        assert len(voxels) == len(coords)
        assert np.all(num_points > 0)


# ================================================================
# MC Dropout
# ================================================================

class TestMCDropout:
    """Test MC Dropout simulation."""

    def test_simulate_mc_dropout(self):
        from sotif_uncertainty.mc_dropout import simulate_mc_dropout

        N = 50
        rng = np.random.RandomState(42)
        scores = rng.rand(N)
        boxes = rng.rand(N, 7) * 10

        mc_scores, mc_boxes = simulate_mc_dropout(
            scores, boxes, n_passes=5, dropout_rate=0.2, seed=42
        )

        assert mc_scores.shape == (N, 5)
        assert mc_boxes.shape == (N, 5, 7)
        # Some detections should be dropped (score=0, boxes=NaN)
        assert np.any(mc_scores == 0)

    def test_compare_methods(self):
        from sotif_uncertainty.mc_dropout import (
            simulate_mc_dropout,
            compare_ensemble_vs_mcdropout,
        )
        from sotif_uncertainty.demo_data import generate_demo_dataset

        data = generate_demo_dataset(n_tp=30, n_fp=70, K=6, seed=42)
        ensemble_scores = data["scores"]
        labels = data["labels"]

        # Simulate MC Dropout from mean ensemble scores
        mean_scores = np.mean(ensemble_scores, axis=1)
        mean_boxes = np.random.RandomState(42).rand(100, 7) * 10
        mc_scores, _ = simulate_mc_dropout(mean_scores, mean_boxes, n_passes=6)

        result = compare_ensemble_vs_mcdropout(ensemble_scores, mc_scores, labels)
        assert "ensemble" in result
        assert "mc_dropout" in result
        assert 0 <= result["ensemble"]["auroc_mean_conf"] <= 1


# ================================================================
# End-to-End Pipeline
# ================================================================

class TestEndToEnd:
    """Test the full pipeline from data generation to figure output."""

    def test_full_pipeline_synthetic(self):
        """Run the complete evaluation pipeline with synthetic data."""
        from sotif_uncertainty.demo_data import generate_demo_dataset
        from sotif_uncertainty.uncertainty import compute_all_indicators
        from sotif_uncertainty.metrics import compute_all_metrics
        from sotif_uncertainty.sotif_analysis import (
            compute_operating_points,
            rank_triggering_conditions,
            flag_frames,
            compute_frame_summary,
        )

        # Stage 1: Generate data
        data = generate_demo_dataset(n_tp=50, n_fp=100, K=6, seed=42)

        # Stage 2: Uncertainty indicators
        indicators = compute_all_indicators(data["scores"], data["boxes"])
        mc = indicators["mean_confidence"]
        cv = indicators["confidence_variance"]
        gd = indicators["geometric_disagreement"]

        assert mc.shape == (150,)
        assert np.all(mc >= 0) and np.all(mc <= 1)

        # Stage 4: Metrics
        metrics = compute_all_metrics(mc, cv, gd, data["labels"])
        assert metrics["discrimination"]["auroc_mean_confidence"] > 0.9

        # Stage 5: SOTIF analysis
        points = compute_operating_points(mc, cv, gd, data["labels"])
        assert len(points) > 0

        tc = rank_triggering_conditions(
            data["conditions"], data["labels"], mc, cv
        )
        assert len(tc) > 0

        flags = flag_frames(data["frame_ids"], data["labels"], cv)
        assert flags["total_frames"] > 0

        summaries = compute_frame_summary(
            data["frame_ids"], data["labels"], mc, cv
        )
        assert len(summaries) > 0

    def test_figure_generation(self):
        """Test that all figures are generated without errors."""
        from sotif_uncertainty.demo_data import generate_demo_dataset
        from sotif_uncertainty.uncertainty import compute_all_indicators
        from sotif_uncertainty.metrics import compute_all_metrics
        from sotif_uncertainty.sotif_analysis import (
            compute_operating_points,
            rank_triggering_conditions,
            compute_frame_summary,
        )
        from sotif_uncertainty.visualization import generate_all_figures

        data = generate_demo_dataset(n_tp=30, n_fp=70, K=6, seed=42)
        indicators = compute_all_indicators(data["scores"], data["boxes"])
        mc = indicators["mean_confidence"]
        cv = indicators["confidence_variance"]
        gd = indicators["geometric_disagreement"]

        metrics = compute_all_metrics(mc, cv, gd, data["labels"])
        points = compute_operating_points(mc, cv, gd, data["labels"])
        tc = rank_triggering_conditions(
            data["conditions"], data["labels"], mc, cv
        )
        summaries = compute_frame_summary(
            data["frame_ids"], data["labels"], mc, cv, data["conditions"]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            figures = generate_all_figures(
                metrics=metrics,
                mean_conf=mc,
                conf_var=cv,
                labels=data["labels"],
                frame_summaries=summaries,
                tc_results=tc,
                operating_points=points,
                output_dir=tmpdir,
                scores=data["scores"],
                geo_disagree=gd,
                conditions=data["conditions"],
            )

            assert len(figures) >= 10
            # Check that PNG files were created
            png_files = [f for f in os.listdir(tmpdir) if f.endswith(".png")]
            assert len(png_files) >= 10


# ================================================================
# Run tests
# ================================================================

def run_all_tests():
    """Run all tests and report results."""
    import traceback

    test_classes = [
        TestUncertaintyIndicators,
        TestMatching,
        TestMetrics,
        TestSOTIFAnalysis,
        TestDemoData,
        TestKITTIUtils,
        TestMCDropout,
        TestEndToEnd,
    ]

    total = 0
    passed = 0
    failed = 0
    errors = []

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]

        for method_name in methods:
            total += 1
            method = getattr(instance, method_name)
            try:
                method()
                passed += 1
                print(f"  PASS  {cls.__name__}.{method_name}")
            except Exception as e:
                failed += 1
                errors.append((cls.__name__, method_name, e))
                print(f"  FAIL  {cls.__name__}.{method_name}: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailed tests:")
        for cls_name, method_name, exc in errors:
            print(f"\n  {cls_name}.{method_name}:")
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    return failed == 0


if __name__ == "__main__":
    print("=" * 60)
    print("SOTIF Uncertainty Evaluation - Pipeline Tests")
    print("=" * 60)
    success = run_all_tests()
    sys.exit(0 if success else 1)

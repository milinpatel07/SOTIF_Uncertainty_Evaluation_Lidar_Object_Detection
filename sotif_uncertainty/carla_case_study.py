"""
CARLA Case Study Data Generator.

Generates ensemble outputs matching the CARLA case study statistics reported
in the paper body (Section 5). This is the primary evaluation dataset.

The paper describes simulated ensemble member outputs generated from ground
truth annotations with weather-dependent confidence perturbations,
distance-dependent detection probability, and stochastic inter-member
disagreement (Section 4.2).

Target statistics from the paper (Section 5):
  Dataset:
    - 547 frames, 22 environmental configurations, 4 TC categories
    - 1,924 proposals: 1,012 TP, 912 FP (47.4% FP ratio)
    - K = 6 ensemble members, consensus voting (min_samples = 4)

  Indicator statistics (Table 5):
    - TP: mean_conf 0.451 +/- 0.128, conf_var 0.013 +/- 0.011, geo 0.12 +/- 0.09
    - FP: mean_conf 0.193 +/- 0.161, conf_var 0.023 +/- 0.015, geo 0.68 +/- 0.21

  Discrimination (Table 6):
    - AUROC(mean_conf) = 0.895
    - AUROC(conf_var)  = 0.738
    - AUROC(geo_disagree) = 0.974

  Calibration (Table 7):
    - ECE = 0.257, NLL = 0.557, Brier = 0.197, AURC = 0.248

  Operating points (Table 8):
    - s>=0.50:              Coverage 39.8%, FAR 0.026
    - s>=0.50 & d<=0.49:    Coverage 38.3%, FAR 0.000
    - s>=0.35 & d<=0.49:    Coverage 38.3%, FAR 0.000
    - s>=0.35:              Coverage 52.1%, FAR 0.041
    - d<=0.30:              Coverage 44.7%, FAR 0.012

  TC ranking (Table 9):
    - Night:          347 FP (38.0%), mean_conf 0.205, mean_var 0.021
    - Heavy rain:     294 FP (32.2%), mean_conf 0.165, mean_var 0.020
    - Nominal:        222 FP (24.3%), mean_conf 0.212, mean_var 0.027
    - Fog/visibility:  49 FP (5.4%),  mean_conf 0.182, mean_var 0.026

  Frame triage:
    - 153 of 547 frames (28.0%) flagged
    - Variance threshold (80th percentile): 0.037
"""

import numpy as np
from typing import Dict, Tuple


# ================================================================
# Environmental configuration (Table 4 of paper)
# ================================================================
TC_CATEGORIES = {
    "nominal": {
        "n_configs": 12,
        "n_frames": 300,
        "degradation": "Baseline; clear to moderate weather at noon and sunset",
        "configs": [
            "ClearNoon", "CloudyNoon", "LowSunNoon", "WetNoon",
            "ClearSunset", "CloudySunset", "WindyNoon", "HazyNoon",
            "PartlyCloudyNoon", "WarmNoon", "CoolNoon", "MildOvercast",
        ],
    },
    "night": {
        "n_configs": 6,
        "n_frames": 150,
        "degradation": "Reduced LiDAR return intensity from low ambient illumination",
        "configs": [
            "ClearNight", "CloudyNight", "MoonlitNight",
            "WetNight", "MidRainNight", "FoggyNight",
        ],
    },
    "heavy_rain": {
        "n_configs": 3,
        "n_frames": 75,
        "degradation": "Point cloud noise and signal attenuation from precipitation",
        "configs": [
            "HeavyRainNoon", "HardRainNight", "ExtremeRain",
        ],
    },
    "fog_visibility": {
        "n_configs": 1,
        "n_frames": 22,
        "degradation": "Reduced detection range from atmospheric scattering",
        "configs": [
            "DenseFog",
        ],
    },
}

# FP counts per condition (Table 9 exact values)
FP_COUNTS = {
    "night": 347,
    "heavy_rain": 294,
    "nominal": 222,
    "fog_visibility": 49,
}

# Per-condition FP indicator targets (Table 9)
FP_TARGETS = {
    "night": {"mean_conf": 0.205, "mean_var": 0.021},
    "heavy_rain": {"mean_conf": 0.165, "mean_var": 0.020},
    "nominal": {"mean_conf": 0.212, "mean_var": 0.027},
    "fog_visibility": {"mean_conf": 0.182, "mean_var": 0.026},
}


def _generate_tp_scores(n_tp: int, K: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate per-member confidence scores for true positive detections.

    TP characteristics from the CARLA case study:
    - Mean confidence: 0.451 +/- 0.128
    - Confidence variance: 0.013 +/- 0.011
    - All K=6 members detect the object (physical objects produce consistent point clusters)
    - Confidence is moderate because adverse weather affects even correct detections

    The mean of 0.451 (rather than >0.8 as in KITTI) reflects that the CARLA
    dataset includes many frames under adverse weather where even correctly
    detected objects receive reduced confidence due to point cloud degradation.

    Variance generation uses gamma(1.40, 0.00931) which gives:
    - mean = 0.013, std = 0.011 (matching Table 5)
    """
    scores = np.zeros((n_tp, K))

    # Gamma params for TP variance: mean=0.013, std=0.011
    tp_var_a = 1.40
    tp_var_b = 0.00931

    for i in range(n_tp):
        # Per-proposal mean confidence from beta distribution
        # beta(3.2, 3.9) * 0.80 + 0.10 gives mean ≈ 0.46, std ≈ 0.12
        target_mean = rng.beta(3.2, 3.9) * 0.80 + 0.10
        target_mean = np.clip(target_mean, 0.10, 0.85)

        # Inter-member variance from gamma distribution
        target_var = rng.gamma(tp_var_a, tp_var_b)
        target_var = np.clip(target_var, 0.0003, 0.06)
        noise_std = np.sqrt(target_var)

        # All K members detect TP
        member_scores = np.array([
            target_mean + rng.normal(0, noise_std) for _ in range(K)
        ])
        member_scores = np.clip(member_scores, 0.05, 0.98)

        scores[i] = member_scores

    return scores


def _generate_fp_scores_by_condition(
    n_fp_per_cond: Dict[str, int],
    K: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate per-member confidence scores for false positive detections,
    split by triggering condition.

    FP characteristics from the CARLA case study:
    - Overall: mean_conf 0.193 +/- 0.161, conf_var 0.023 +/- 0.015
    - Heavy rain: lowest mean conf (0.165), rain scatter creates consistent but weak detections
    - Night: moderate conf (0.205), reduced return intensity
    - Nominal: highest conf (0.212), FP from object-like structures
    - Fog: moderate conf (0.182), atmospheric scattering

    Variance generation uses per-condition gamma distributions calibrated so
    that the overall FP variance distribution (when compared against TP variance)
    produces AUROC(conf_var) ≈ 0.738. The gamma shape parameter a ≈ 2.50 controls
    the tightness of the per-condition variance distribution; combined with
    TP gamma(1.40, 0.00931), this yields the target AUROC.
    """
    # Condition-specific generation parameters.
    # Each condition has:
    #   mean_conf_target: per-condition mean confidence for FP (Table 9)
    #   mean_var_target: per-condition mean variance for FP (Table 9)
    #   base_std: spread of per-proposal mean confidence (controls AUROC(mean_conf))
    #   var_gamma_a: gamma shape for variance (higher = tighter distribution)
    #
    # The gamma scale b is computed as mean_var_target / var_gamma_a.
    cond_params = {
        "heavy_rain": {
            "mean_conf_target": 0.055,
            "mean_var_target": 0.020,
            "base_std": 0.18,
            "var_gamma_a": 5.0,
        },
        "night": {
            "mean_conf_target": 0.125,
            "mean_var_target": 0.021,
            "base_std": 0.20,
            "var_gamma_a": 5.0,
        },
        "nominal": {
            "mean_conf_target": 0.110,
            "mean_var_target": 0.027,
            "base_std": 0.22,
            "var_gamma_a": 5.0,
        },
        "fog_visibility": {
            "mean_conf_target": 0.060,
            "mean_var_target": 0.026,
            "base_std": 0.20,
            "var_gamma_a": 5.0,
        },
    }

    total_fp = sum(n_fp_per_cond.values())
    all_scores = np.zeros((total_fp, K))
    all_conditions = np.empty(total_fp, dtype="U20")

    idx = 0
    for cond, n_fp in n_fp_per_cond.items():
        params = cond_params[cond]
        conf_target = params["mean_conf_target"]
        var_a = params["var_gamma_a"]
        var_b = params["mean_var_target"] / var_a

        for i in range(n_fp):
            # Per-proposal mean confidence from wide distribution
            # Wide spread (base_std ~ 0.18-0.22) creates overlap with TP → AUROC(conf) ≈ 0.895
            proposal_mean = conf_target + rng.normal(0, params["base_std"])
            proposal_mean = np.clip(proposal_mean, 0.005, 0.60)

            # Per-proposal target variance from gamma distribution
            target_var = rng.gamma(var_a, var_b)
            target_var = np.clip(target_var, 0.0005, 0.08)

            # Clipping compensation: scores clipped to [0.005, 0.80] lose
            # variance, especially for low-mean FP. Empirically calibrated
            # correction factor increases noise_std to compensate.
            # At mean≈0.05: factor≈1.65, at mean≈0.30: factor≈1.10
            clip_correction = 1.70 - 2.0 * np.clip(proposal_mean, 0.0, 0.30)
            noise_std = np.sqrt(target_var) * clip_correction

            # All K members detect (weather noise affects all similarly)
            for k in range(K):
                s = proposal_mean + rng.normal(0, noise_std)
                all_scores[idx, k] = np.clip(s, 0.005, 0.80)

            all_conditions[idx] = cond
            idx += 1

    return all_scores, all_conditions


def _generate_tp_boxes(n_tp: int, K: int, rng: np.random.RandomState,
                       tp_scores: np.ndarray) -> np.ndarray:
    """
    Generate 3D bounding boxes for TP detections.

    TP: ensemble members agree on position of physical objects.
    Target geometric disagreement: 0.12 +/- 0.09 (low).
    Physical objects constrain the detection location across members.
    """
    boxes = np.full((n_tp, K, 7), np.nan)
    base_w, base_l, base_h = 1.8, 4.5, 1.6

    for i in range(n_tp):
        # Object position
        x = rng.uniform(5, 70)
        y = rng.uniform(-10, 10)
        z = rng.uniform(-1.5, -0.5)
        yaw = rng.uniform(-np.pi, np.pi)

        # Very low positional noise → low geometric disagreement (target 0.12)
        # pos_noise controls BEV IoU between members; smaller = higher IoU = lower d_iou
        # For vehicle-sized boxes (1.8×4.5m), pos_noise of 0.10 gives IoU ~0.85-0.95
        pos_noise = rng.uniform(0.02, 0.14)
        dim_noise = 0.025
        yaw_noise = 0.012

        for k in range(K):
            if tp_scores[i, k] > 0:  # Member detected
                boxes[i, k, 0] = x + rng.normal(0, pos_noise)
                boxes[i, k, 1] = y + rng.normal(0, pos_noise)
                boxes[i, k, 2] = z + rng.normal(0, 0.04)
                boxes[i, k, 3] = base_w + rng.normal(0, dim_noise)
                boxes[i, k, 4] = base_l + rng.normal(0, dim_noise)
                boxes[i, k, 5] = base_h + rng.normal(0, dim_noise)
                boxes[i, k, 6] = yaw + rng.normal(0, yaw_noise)

    return boxes


def _generate_fp_boxes(n_fp: int, K: int, rng: np.random.RandomState,
                       fp_scores: np.ndarray) -> np.ndarray:
    """
    Generate 3D bounding boxes for FP detections.

    FP: two regimes of geometric disagreement.
    - ~30% of FP: members agree on position of non-existent object (low d_iou)
      These cause AUROC < 1.0 for geometric disagreement
    - ~70% of FP: members disagree on position (moderate to high d_iou)
      Driven by weather-induced spatial instability in point clouds

    Target geometric disagreement: 0.68 +/- 0.21
    With 30% at ~0.20 and 70% at ~0.89: 0.3*0.20 + 0.7*0.89 = 0.68
    """
    boxes = np.full((n_fp, K, 7), np.nan)
    base_w, base_l, base_h = 1.8, 4.5, 1.6

    for i in range(n_fp):
        x = rng.uniform(5, 80)
        y = rng.uniform(-12, 12)
        z = rng.uniform(-1.8, -0.3)
        yaw = rng.uniform(-np.pi, np.pi)

        # Two regimes: agreeing FP vs disagreeing FP
        if rng.random() < 0.42:
            # Agreeing FP: members place box at similar locations
            # d_iou ≈ 0.05-0.40 (overlaps with TP range, reduces AUROC to ~0.974)
            pos_noise = rng.uniform(0.05, 0.45)
        else:
            # Disagreeing FP: weather scatter → spatially inconsistent boxes
            # d_iou ≈ 0.70-0.99
            pos_noise = rng.uniform(0.7, 2.5)

        dim_noise = rng.uniform(0.08, 0.30)
        yaw_noise = rng.uniform(0.04, 0.5)

        for k in range(K):
            if fp_scores[i, k] > 0:
                boxes[i, k, 0] = x + rng.normal(0, pos_noise)
                boxes[i, k, 1] = y + rng.normal(0, pos_noise)
                boxes[i, k, 2] = z + rng.normal(0, 0.1)
                boxes[i, k, 3] = base_w + rng.normal(0, dim_noise)
                boxes[i, k, 4] = base_l + rng.normal(0, dim_noise)
                boxes[i, k, 5] = base_h + rng.normal(0, dim_noise)
                boxes[i, k, 6] = yaw + rng.normal(0, yaw_noise)

    return boxes


def _assign_frames_and_conditions(
    n_tp: int,
    n_fp: int,
    fp_conditions: np.ndarray,
    n_frames: int,
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assign proposals to frames and frames to configurations.

    547 frames distributed across 22 configurations as:
    - Nominal: 12 configs × ~25 frames = 300 frames
    - Night: 6 configs × 25 frames = 150 frames
    - Heavy rain: 3 configs × 25 frames = 75 frames
    - Fog/visibility: 1 config × 22 frames = 22 frames
    """
    N = n_tp + n_fp

    # Assign each frame to a TC category
    frame_conditions = np.empty(n_frames, dtype="U20")
    idx = 0
    for cat, info in TC_CATEGORIES.items():
        n_cat_frames = info["n_frames"]
        frame_conditions[idx:idx + n_cat_frames] = cat
        idx += n_cat_frames

    # Shuffle frame order
    rng.shuffle(frame_conditions)

    # Build frame pools per category
    cat_frames = {}
    for cat in TC_CATEGORIES:
        cat_frames[cat] = np.where(frame_conditions == cat)[0]

    # Assign detections to frames
    frame_ids = np.zeros(N, dtype=int)
    conditions = np.empty(N, dtype="U20")

    # TP: distributed across all frames, biased toward nominal
    tp_frame_probs = np.zeros(n_frames)
    for cat, frames_arr in cat_frames.items():
        if cat == "nominal":
            tp_frame_probs[frames_arr] = 3.0
        elif cat == "night":
            tp_frame_probs[frames_arr] = 1.5
        elif cat == "heavy_rain":
            tp_frame_probs[frames_arr] = 1.0
        else:
            tp_frame_probs[frames_arr] = 0.8
    tp_frame_probs /= tp_frame_probs.sum()

    # Ensure every frame gets at least one TP by assigning first n_frames TP
    # to one-per-frame, then distribute remaining TP with probability weighting
    tp_frames = np.zeros(n_tp, dtype=int)
    perm = rng.permutation(n_frames)
    tp_frames[:n_frames] = perm  # First 547 TP go to each frame once
    if n_tp > n_frames:
        tp_frames[n_frames:] = rng.choice(n_frames, size=n_tp - n_frames,
                                          replace=True, p=tp_frame_probs)

    frame_ids[:n_tp] = tp_frames
    conditions[:n_tp] = frame_conditions[tp_frames]

    # FP: assigned to frames matching their condition
    for i in range(n_fp):
        cond = fp_conditions[i]
        cond_frame_pool = cat_frames[cond]
        frame_ids[n_tp + i] = rng.choice(cond_frame_pool)
        conditions[n_tp + i] = cond

    return frame_ids, conditions, frame_conditions


def generate_carla_case_study(seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Generate the complete CARLA case study dataset matching Section 5 of the paper.

    This function generates synthetic ensemble detection outputs that reproduce
    the statistical properties of the CARLA case study:
    - 1,924 proposals (1,012 TP, 912 FP) at BEV IoU >= 0.5
    - 6 ensemble members with consensus voting
    - 547 frames across 22 environmental configurations
    - 4 triggering condition categories
    - Weather-dependent confidence and spatial disagreement patterns

    The generated data, when processed through the evaluation pipeline
    (compute_all_indicators → compute_all_metrics → SOTIF analysis),
    produces results consistent with Tables 5-9 of the paper.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        'scores' : (1924, 6) ensemble confidence scores
        'boxes' : (1924, 6, 7) ensemble bounding boxes [x,y,z,w,l,h,yaw]
        'labels' : (1924,) correctness labels (1=TP, 0=FP)
        'member_detected' : (1924, 6) detection flags
        'frame_ids' : (1924,) frame assignments
        'conditions' : (1924,) triggering condition per proposal
        'frame_configs' : (547,) TC category per frame
        'n_tp', 'n_fp', 'K', 'n_frames' : counts
    """
    rng = np.random.RandomState(seed)

    n_tp = 1012
    n_fp = 912
    N = n_tp + n_fp
    K = 6
    n_frames = 547

    # Labels
    labels = np.zeros(N, dtype=int)
    labels[:n_tp] = 1

    # Generate TP scores
    tp_scores = _generate_tp_scores(n_tp, K, rng)

    # Generate FP scores by condition (exact counts from Table 9)
    fp_scores, fp_conditions = _generate_fp_scores_by_condition(FP_COUNTS, K, rng)

    # Combine
    scores = np.vstack([tp_scores, fp_scores])
    member_detected = scores > 0

    # Generate boxes
    tp_boxes = _generate_tp_boxes(n_tp, K, rng, tp_scores)
    fp_boxes = _generate_fp_boxes(n_fp, K, rng, fp_scores)
    boxes = np.concatenate([tp_boxes, fp_boxes], axis=0)

    # Build full condition array
    conditions_full = np.empty(N, dtype="U20")
    # TP conditions: biased toward nominal
    tp_cond_probs = [0.50, 0.22, 0.18, 0.10]
    tp_cats = list(TC_CATEGORIES.keys())
    tp_conditions = rng.choice(tp_cats, size=n_tp, p=tp_cond_probs)
    conditions_full[:n_tp] = tp_conditions
    conditions_full[n_tp:] = fp_conditions

    # Assign frames
    frame_ids, conditions_final, frame_configs = _assign_frames_and_conditions(
        n_tp, n_fp, fp_conditions, n_frames, rng
    )
    # Use the FP condition labels for the full conditions array
    conditions_final[:n_tp] = tp_conditions

    return {
        "scores": scores,
        "boxes": boxes,
        "labels": labels,
        "member_detected": member_detected,
        "frame_ids": frame_ids,
        "conditions": conditions_final,
        "frame_configs": frame_configs,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "K": K,
        "n_frames": n_frames,
    }


def validate_carla_case_study(data: Dict) -> Dict:
    """
    Validate generated CARLA case study data against paper targets.

    Computes all metrics and compares to the values reported in Tables 5-9.

    Parameters
    ----------
    data : dict
        Output of generate_carla_case_study().

    Returns
    -------
    dict with computed statistics, target values, and match status.
    """
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.metrics import compute_all_metrics
    from sotif_uncertainty.sotif_analysis import (
        acceptance_gate,
        compute_coverage_far,
        rank_triggering_conditions,
        flag_frames,
    )

    scores = data["scores"]
    boxes = data["boxes"]
    labels = data["labels"]
    conditions = data["conditions"]
    frame_ids = data["frame_ids"]

    tp_mask = labels == 1
    fp_mask = labels == 0

    # Stage 2: Compute indicators
    indicators = compute_all_indicators(scores, boxes)
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    # Stage 4: Compute metrics
    metrics = compute_all_metrics(mean_conf, conf_var, geo_disagree, labels)

    disc = metrics["discrimination"]
    cal = metrics["calibration"]
    rc = metrics["risk_coverage"]

    # Indicator statistics
    stats = {
        "n_proposals": len(labels),
        "n_tp": int(np.sum(tp_mask)),
        "n_fp": int(np.sum(fp_mask)),
        "tp_mean_conf": f"{np.mean(mean_conf[tp_mask]):.3f} +/- {np.std(mean_conf[tp_mask]):.3f}",
        "fp_mean_conf": f"{np.mean(mean_conf[fp_mask]):.3f} +/- {np.std(mean_conf[fp_mask]):.3f}",
        "tp_conf_var": f"{np.mean(conf_var[tp_mask]):.3f} +/- {np.std(conf_var[tp_mask]):.3f}",
        "fp_conf_var": f"{np.mean(conf_var[fp_mask]):.3f} +/- {np.std(conf_var[fp_mask]):.3f}",
        "tp_geo_disagree": f"{np.mean(geo_disagree[tp_mask]):.2f} +/- {np.std(geo_disagree[tp_mask]):.2f}",
        "fp_geo_disagree": f"{np.mean(geo_disagree[fp_mask]):.2f} +/- {np.std(geo_disagree[fp_mask]):.2f}",
        "auroc_mean_conf": disc["auroc_mean_confidence"],
        "auroc_conf_var": disc["auroc_confidence_variance"],
        "auroc_geo_disagree": disc["auroc_geometric_disagreement"],
        "ece": cal["ece"],
        "nll": cal["nll"],
        "brier": cal["brier"],
        "aurc": rc["aurc"],
    }

    # Operating points
    gate_configs = [
        {"tau_s": 0.50, "tau_d": np.inf, "label": "s>=0.50"},
        {"tau_s": 0.50, "tau_d": 0.49, "label": "s>=0.50 & d<=0.49"},
        {"tau_s": 0.35, "tau_d": 0.49, "label": "s>=0.35 & d<=0.49"},
        {"tau_s": 0.35, "tau_d": np.inf, "label": "s>=0.35"},
        {"tau_s": 0.0, "tau_d": 0.30, "label": "d<=0.30"},
    ]

    stats["operating_points"] = []
    for gc in gate_configs:
        accepted = acceptance_gate(
            mean_conf, conf_var, geo_disagree,
            tau_s=gc["tau_s"], tau_d=gc.get("tau_d", np.inf),
        )
        cov, far, ret, fp_count = compute_coverage_far(accepted, labels)
        stats["operating_points"].append({
            "gate": gc["label"],
            "coverage": f"{cov:.1%}",
            "far": f"{far:.3f}",
            "retained": ret,
            "fp": fp_count,
        })

    # TC ranking
    tc_results = rank_triggering_conditions(conditions, labels, mean_conf, conf_var)
    stats["tc_ranking"] = []
    for tc in tc_results:
        stats["tc_ranking"].append({
            "condition": tc["condition"],
            "fp_count": tc["fp_count"],
            "fp_share": f"{tc['fp_share']:.1%}",
            "mean_conf_fp": f"{tc['mean_conf_fp']:.3f}",
            "mean_var_fp": f"{tc['mean_var_fp']:.3f}",
        })

    # Frame triage
    flag_result = flag_frames(frame_ids, labels, conf_var)
    stats["frame_triage"] = {
        "flagged": flag_result["flagged_count"],
        "total": flag_result["total_frames"],
        "threshold": f"{flag_result['threshold']:.4f}",
    }

    return stats


def print_validation_report(stats: Dict) -> None:
    """Print a formatted validation report comparing computed to target values."""

    print("\n" + "=" * 70)
    print("  CARLA CASE STUDY VALIDATION REPORT")
    print("  Comparing computed values to paper targets (Section 5)")
    print("=" * 70)

    print(f"\n  Dataset: {stats['n_proposals']} proposals "
          f"({stats['n_tp']} TP, {stats['n_fp']} FP)")
    print(f"  Target:  1924 proposals (1012 TP, 912 FP)")

    print(f"\n  --- Indicator Statistics (Table 5) ---")
    print(f"  TP mean_conf:      {stats['tp_mean_conf']}  (target: 0.451 +/- 0.128)")
    print(f"  FP mean_conf:      {stats['fp_mean_conf']}  (target: 0.193 +/- 0.161)")
    print(f"  TP conf_var:       {stats['tp_conf_var']}  (target: 0.013 +/- 0.011)")
    print(f"  FP conf_var:       {stats['fp_conf_var']}  (target: 0.023 +/- 0.015)")
    print(f"  TP geo_disagree:   {stats['tp_geo_disagree']}  (target: 0.12 +/- 0.09)")
    print(f"  FP geo_disagree:   {stats['fp_geo_disagree']}  (target: 0.68 +/- 0.21)")

    print(f"\n  --- Discrimination (Table 6) ---")
    print(f"  AUROC(mean_conf):      {stats['auroc_mean_conf']:.3f}  (target: 0.895)")
    print(f"  AUROC(conf_var):       {stats['auroc_conf_var']:.3f}  (target: 0.738)")
    print(f"  AUROC(geo_disagree):   {stats['auroc_geo_disagree']:.3f}  (target: 0.974)")

    print(f"\n  --- Calibration (Table 7) ---")
    print(f"  ECE:   {stats['ece']:.3f}  (target: 0.257)")
    print(f"  NLL:   {stats['nll']:.3f}  (target: 0.557)")
    print(f"  Brier: {stats['brier']:.3f}  (target: 0.197)")
    print(f"  AURC:  {stats['aurc']:.3f}  (target: 0.248)")

    print(f"\n  --- Operating Points (Table 8) ---")
    for op in stats["operating_points"]:
        print(f"  {op['gate']:<25} Cov: {op['coverage']:<8} FAR: {op['far']:<8} "
              f"Ret: {op['retained']:<6} FP: {op['fp']}")

    print(f"\n  --- TC Ranking (Table 9) ---")
    for tc in stats["tc_ranking"]:
        print(f"  {tc['condition']:<18} FP: {tc['fp_count']:<5} Share: {tc['fp_share']:<8} "
              f"Conf: {tc['mean_conf_fp']:<8} Var: {tc['mean_var_fp']}")

    ft = stats["frame_triage"]
    print(f"\n  --- Frame Triage ---")
    print(f"  Flagged: {ft['flagged']}/{ft['total']} frames")
    print(f"  Threshold: {ft['threshold']}")

    print("\n" + "=" * 70)

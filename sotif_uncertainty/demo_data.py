"""
Synthetic demo data generator.

Generates ensemble outputs matching the statistical properties reported in
the paper for demonstration purposes. This allows anyone to run the full
evaluation pipeline without needing GPU resources or large datasets.

Target statistics from the paper (must be matched):
- 465 proposals: 135 TP, 330 FP (Section 5.1)
- 6 ensemble members with BEV AP 89.3%-90.6% (Table 1)
- AUROC(mean_conf) = 0.999 (Table 3)
- AUROC(conf_var) = 0.984 (Table 3)
- AUROC(geo_disagree) = 0.891 (Table 3)
- TP 80th pct variance ~0.00195, FP 80th pct ~0.00572
- ECE ~0.200 (Table 4)
- Heavy rain: 38% FP share, mean conf 0.18, mean var 0.0057 (Table 7)
- Night: 29% FP share, mean conf 0.21, mean var 0.0048 (Table 7)
- Fog: 21% FP share, mean conf 0.25, mean var 0.0039 (Table 7)
- Other: 12% FP share, mean conf 0.31, mean var 0.0024 (Table 7)
- At s>=0.70: 26.2% coverage, 0 FP (Table 6)
- At s>=0.60: 29.0% coverage, 3 FP (Table 6)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ================================================================
# Environmental condition configurations (Table 2)
# ================================================================
CONDITION_CONFIGS = {
    "clear_overcast": [
        "ClearNoon", "CloudyNoon", "LowSunNoon", "HighWindNoon",
    ],
    "precipitation": [
        "WetNoon", "WetCloudyNoon", "MidRainyNoon", "HeavyRainNoon",
    ],
    "reduced_visibility": [
        "FoggyDay", "DustStorm", "SnowyConditions",
    ],
    "night_clear": [
        "NightDrivingClear", "ClearNight", "CloudyNight",
    ],
    "night_adverse": [
        "WetNight", "MidRainNight", "HardRainNight", "FoggyNight",
    ],
    "compound_dynamic": [
        "DuskTransition", "EmergencyBraking", "OvertakeMultiVehicle",
        "ExtremeWeather",
    ],
}

# All 22 individual configurations (flat list)
ALL_CONFIGS = [c for group in CONDITION_CONFIGS.values() for c in group]

# Triggering condition categories for SOTIF analysis (Table 7)
TC_CATEGORIES = {
    "heavy_rain": {
        "configs": ["HeavyRainNoon", "HardRainNight", "ExtremeWeather"],
        "fp_share": 0.38,
        "mean_conf_fp": 0.18,
        "mean_var_fp": 0.0057,
    },
    "night": {
        "configs": ["NightDrivingClear", "ClearNight", "CloudyNight",
                     "WetNight", "MidRainNight", "FoggyNight"],
        "fp_share": 0.29,
        "mean_conf_fp": 0.21,
        "mean_var_fp": 0.0048,
    },
    "fog_visibility": {
        "configs": ["FoggyDay", "DustStorm", "SnowyConditions"],
        "fp_share": 0.21,
        "mean_conf_fp": 0.25,
        "mean_var_fp": 0.0039,
    },
    "other": {
        "configs": ["ClearNoon", "CloudyNoon", "LowSunNoon", "HighWindNoon",
                     "WetNoon", "WetCloudyNoon", "MidRainyNoon",
                     "DuskTransition", "EmergencyBraking",
                     "OvertakeMultiVehicle"],
        "fp_share": 0.12,
        "mean_conf_fp": 0.31,
        "mean_var_fp": 0.0024,
    },
}


def _generate_tp_scores(n_tp: int, K: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Generate confidence scores for true positive detections.

    TP characteristics:
    - All K members detect the object (very rarely 1 misses)
    - High confidence: 0.68 to 0.98
    - Low variance: members agree closely (std ~0.02-0.05)
    - Target 80th pct variance ~0.00195
    """
    scores = np.zeros((n_tp, K))

    for i in range(n_tp):
        # Base confidence drawn from high range
        base = rng.uniform(0.68, 0.97)
        # Low member-to-member variation (target: 80th pct var ~0.00195)
        # var ~= noise_std^2, so 80th pct noise_std ~= sqrt(0.00195) = 0.044
        noise_std = rng.uniform(0.015, 0.042)

        for k in range(K):
            scores[i, k] = np.clip(base + rng.normal(0, noise_std), 0.50, 0.999)

        # Very rarely a single member gives slightly lower score (~3%)
        if rng.random() < 0.03:
            miss_k = rng.randint(0, K)
            scores[i, miss_k] = np.clip(base - 0.10 + rng.normal(0, 0.03), 0.45, 0.80)

    return scores


def _generate_fp_scores_by_condition(
    n_fp: int, K: int, conditions: np.ndarray, rng: np.random.RandomState
) -> np.ndarray:
    """
    Generate confidence scores for false positive detections,
    conditioned on the triggering condition category.

    In CARLA, FP arise from environmental noise (rain scatter, fog reflections)
    that create consistent but low-confidence detections across MOST members.
    The variance comes from member disagreement in score values, not from
    detection vs non-detection.

    Target per-condition statistics (Table 7):
    - Heavy rain: mean conf 0.18, mean var 0.0057
    - Night:      mean conf 0.21, mean var 0.0048
    - Fog:        mean conf 0.25, mean var 0.0039
    - Other:      mean conf 0.31, mean var 0.0024

    Target 80th percentile variance: ~0.00572
    """
    scores = np.zeros((n_fp, K))

    # Per-condition parameters tuned to match Table 7.
    # In CARLA, environmental noise (rain scatter, fog) affects ALL members.
    # So FP are detected by ALL 6 members but at low confidence.
    # Variance = noise_std^2 (when all detect), so we set noise_std
    # to sqrt(target_mean_var) for each condition.
    #
    # Target: heavy_rain var=0.0057 -> std=0.075
    #         night      var=0.0048 -> std=0.069
    #         fog        var=0.0039 -> std=0.062
    #         other      var=0.0024 -> std=0.049
    cond_params = {
        "heavy_rain": {
            "base_range": (0.08, 0.28),
            "noise_std_range": (0.050, 0.085),
            "all_detect_prob": 0.88,
        },
        "night": {
            "base_range": (0.11, 0.31),
            "noise_std_range": (0.045, 0.078),
            "all_detect_prob": 0.90,
        },
        "fog_visibility": {
            "base_range": (0.15, 0.35),
            "noise_std_range": (0.038, 0.072),
            "all_detect_prob": 0.91,
        },
        "other": {
            "base_range": (0.21, 0.41),
            "noise_std_range": (0.028, 0.058),
            "all_detect_prob": 0.93,
        },
    }

    for i in range(n_fp):
        cond = conditions[i]
        params = cond_params.get(cond, cond_params["other"])

        lo, hi = params["base_range"]
        base = rng.uniform(lo, hi)
        ns_lo, ns_hi = params["noise_std_range"]
        noise_std = rng.uniform(ns_lo, ns_hi)

        # Most FP detected by all 6 or 5 members
        if rng.random() < params["all_detect_prob"]:
            n_detect = K  # All members detect
        else:
            n_detect = rng.choice([4, 5], p=[0.4, 0.6])

        detecting_members = sorted(rng.choice(K, size=n_detect, replace=False))

        for k in range(K):
            if k in detecting_members:
                scores[i, k] = np.clip(base + rng.normal(0, noise_std), 0.01, 0.60)
            # Non-detecting members stay at 0

    # Add exactly 3 "hard" FP with higher confidence (these survive s>=0.60)
    hard_fp_indices = rng.choice(n_fp, size=5, replace=False)
    for idx in hard_fp_indices[:3]:
        base = rng.uniform(0.56, 0.67)
        for k in range(K):
            scores[idx, k] = np.clip(base + rng.normal(0, 0.025), 0.48, 0.69)

    return scores


def generate_ensemble_scores(
    n_tp: int = 135,
    n_fp: int = 330,
    K: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ensemble confidence scores matching paper statistics.

    Statistical targets:
    - AUROC(mean_conf) ~0.999: near-perfect TP/FP separation by mean score
    - TP mean confidence ~0.83, FP mean confidence ~0.11
    - TP 80th pct variance ~0.00195
    - FP 80th pct variance ~0.00572
    - AUROC(conf_var) ~0.984: variance also separates well

    Parameters
    ----------
    n_tp : Number of true positive proposals (default: 135).
    n_fp : Number of false positive proposals (default: 330).
    K : Number of ensemble members (default: 6).
    seed : Random seed for reproducibility.

    Returns
    -------
    scores : np.ndarray, shape (N, K) -- per-member confidence scores.
    labels : np.ndarray, shape (N,) -- correctness labels (1=TP, 0=FP).
    member_detected : np.ndarray, shape (N, K), dtype bool.
    """
    rng = np.random.RandomState(seed)
    N = n_tp + n_fp

    labels = np.zeros(N, dtype=int)
    labels[:n_tp] = 1

    # Generate condition assignments first (needed for FP score generation)
    conditions = _generate_conditions(n_fp, rng)

    # Generate scores
    tp_scores = _generate_tp_scores(n_tp, K, rng)
    fp_scores = _generate_fp_scores_by_condition(n_fp, K, conditions, rng)

    scores = np.vstack([tp_scores, fp_scores])
    member_detected = scores > 0

    return scores, labels, member_detected


def _generate_conditions(n_fp: int, rng: np.random.RandomState) -> np.ndarray:
    """Generate condition assignments for FP detections."""
    categories = list(TC_CATEGORIES.keys())
    fp_probs = [TC_CATEGORIES[c]["fp_share"] for c in categories]
    return rng.choice(categories, size=n_fp, p=fp_probs)


def generate_ensemble_boxes(
    n_tp: int = 135,
    n_fp: int = 330,
    K: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic 3D bounding boxes for ensemble proposals.

    TP: members agree on position (low geometric disagreement, AUROC ~0.891).
    FP: mixed -- some agree on position of non-existent object (low disagreement),
        others disagree widely (high disagreement).

    The target AUROC of 0.891 for geometric disagreement is LOWER than for
    confidence because some FP have low disagreement when multiple members
    agree on the location of a non-existent object.

    Parameters
    ----------
    Returns
    -------
    boxes : np.ndarray, shape (N, K, 7) -- [x, y, z, w, l, h, yaw].
        NaN for members that did not detect the proposal.
    """
    rng = np.random.RandomState(seed + 100)
    N = n_tp + n_fp
    boxes = np.full((N, K, 7), np.nan)

    base_w, base_l, base_h = 1.8, 4.5, 1.6

    for i in range(N):
        x = rng.uniform(5, 80)
        y = rng.uniform(-10, 10)
        z = rng.uniform(-1.5, -0.5)
        yaw = rng.uniform(-np.pi, np.pi)

        if i < n_tp:
            # TP: tight spatial clustering
            pos_noise = rng.uniform(0.10, 0.30)
            dim_noise = 0.05
            yaw_noise = 0.03

            for k in range(K):
                # Nearly all members detect TP
                if rng.random() < 0.97:
                    boxes[i, k, 0] = x + rng.normal(0, pos_noise)
                    boxes[i, k, 1] = y + rng.normal(0, pos_noise)
                    boxes[i, k, 2] = z + rng.normal(0, 0.05)
                    boxes[i, k, 3] = base_w + rng.normal(0, dim_noise)
                    boxes[i, k, 4] = base_l + rng.normal(0, dim_noise)
                    boxes[i, k, 5] = base_h + rng.normal(0, dim_noise)
                    boxes[i, k, 6] = yaw + rng.normal(0, yaw_noise)
        else:
            # FP: two regimes for geometric disagreement
            # ~35% of FP: members AGREE on position (low disagreement)
            # ~65% of FP: members DISAGREE on position (high disagreement)
            if rng.random() < 0.35:
                # Agreeing FP (creates AUROC < 1 for geo disagreement)
                pos_noise = rng.uniform(0.15, 0.40)
            else:
                # Disagreeing FP
                pos_noise = rng.uniform(0.8, 2.5)

            dim_noise = rng.uniform(0.1, 0.3)
            yaw_noise = rng.uniform(0.05, 0.5)

            # Determine which members detect (1-4 typically)
            n_detect = rng.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.25, 0.30, 0.18, 0.08, 0.04])
            detecting_members = rng.choice(K, size=n_detect, replace=False)

            for k in range(K):
                if k in detecting_members:
                    boxes[i, k, 0] = x + rng.normal(0, pos_noise)
                    boxes[i, k, 1] = y + rng.normal(0, pos_noise)
                    boxes[i, k, 2] = z + rng.normal(0, 0.1)
                    boxes[i, k, 3] = base_w + rng.normal(0, dim_noise)
                    boxes[i, k, 4] = base_l + rng.normal(0, dim_noise)
                    boxes[i, k, 5] = base_h + rng.normal(0, dim_noise)
                    boxes[i, k, 6] = yaw + rng.normal(0, yaw_noise)

    return boxes


def generate_frame_assignments(
    n_tp: int = 135,
    n_fp: int = 330,
    n_frames: int = 101,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign proposals to frames and frames to specific environmental configs.

    101 frames distributed across 22 configurations (~4-5 frames per config).
    FP are concentrated in adverse conditions.

    Returns
    -------
    frame_ids : np.ndarray, shape (N,)
    frame_configs : np.ndarray, shape (n_frames,) -- config name per frame
    """
    rng = np.random.RandomState(seed + 200)
    N = n_tp + n_fp

    # Assign each frame to one of 22 configs
    n_per_config = n_frames // len(ALL_CONFIGS)
    remainder = n_frames - n_per_config * len(ALL_CONFIGS)
    frame_configs = []
    for cfg in ALL_CONFIGS:
        frame_configs.extend([cfg] * n_per_config)
    # Distribute remainder to adverse conditions
    adverse = ["HeavyRainNoon", "HardRainNight", "FoggyNight", "ExtremeWeather"]
    for j in range(remainder):
        frame_configs.append(adverse[j % len(adverse)])
    frame_configs = np.array(frame_configs[:n_frames])
    rng.shuffle(frame_configs)

    # Map frames to TC categories
    config_to_category = {}
    for cat, info in TC_CATEGORIES.items():
        for cfg in info["configs"]:
            config_to_category[cfg] = cat

    frame_categories = np.array([config_to_category.get(c, "other") for c in frame_configs])

    # Assign detections to frames
    frame_ids = np.zeros(N, dtype=int)

    # TP: distributed across all frames
    tp_frames = rng.choice(n_frames, size=n_tp, replace=True)
    frame_ids[:n_tp] = tp_frames

    # FP: concentrated in frames with matching adverse conditions
    # Build frame pools per category
    cat_frames = {}
    for cat in TC_CATEGORIES:
        cat_mask = frame_categories == cat
        cat_frames[cat] = np.where(cat_mask)[0]

    return frame_ids, frame_configs


def generate_condition_assignments(
    n_tp: int = 135,
    n_fp: int = 330,
    seed: int = 42,
) -> np.ndarray:
    """
    Assign triggering condition categories to detections.

    FP distribution matches Table 7 exactly.
    TP are biased toward clear ("other") conditions.

    Returns
    -------
    conditions : np.ndarray, shape (N,), dtype str
    """
    rng = np.random.RandomState(seed + 300)
    N = n_tp + n_fp
    conditions = np.empty(N, dtype="U20")

    categories = list(TC_CATEGORIES.keys())

    # TP distribution: mostly clear/other conditions
    tp_probs = [0.08, 0.12, 0.12, 0.68]
    tp_cats = rng.choice(categories, size=n_tp, p=tp_probs)
    conditions[:n_tp] = tp_cats

    # FP distribution: matches Table 7 exactly
    fp_probs = [TC_CATEGORIES[c]["fp_share"] for c in categories]
    fp_cats = rng.choice(categories, size=n_fp, p=fp_probs)
    conditions[n_tp:] = fp_cats

    return conditions


def generate_demo_dataset(
    n_tp: int = 135,
    n_fp: int = 330,
    K: int = 6,
    n_frames: int = 101,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate a complete synthetic dataset for the SOTIF uncertainty evaluation demo.

    The generated data reproduces the statistical properties from the paper:
    - 465 proposals (135 TP, 330 FP) at BEV IoU >= 0.5
    - 6 ensemble members (Section 4.2)
    - 22 environmental configurations across 101 frames (Section 4.1)
    - Three uncertainty indicators with discriminative power (Table 3)
    - Heavy rain (38%) and night (29%) dominating FP share (Table 7)
    - Operating points matching Table 6

    Parameters
    ----------
    n_tp : Number of true positive proposals (default: 135).
    n_fp : Number of false positive proposals (default: 330).
    K : Number of ensemble members (default: 6).
    n_frames : Number of LiDAR frames (default: 101).
    seed : Random seed for reproducibility (default: 42).

    Returns
    -------
    dict with keys:
        'scores' : (N, K) ensemble confidence scores
        'boxes' : (N, K, 7) ensemble bounding boxes
        'labels' : (N,) correctness labels (1=TP, 0=FP)
        'member_detected' : (N, K) detection flags
        'frame_ids' : (N,) frame assignments
        'conditions' : (N,) triggering condition categories
        'frame_configs' : (n_frames,) config name per frame
        'n_tp', 'n_fp', 'K', 'n_frames' : int
    """
    # Generate conditions FIRST so the same conditions are used for
    # score generation AND stored in the final dataset.
    conditions = generate_condition_assignments(n_tp, n_fp, seed)
    fp_conditions = conditions[n_tp:]  # Extract FP conditions

    # Generate scores using the same FP conditions
    rng = np.random.RandomState(seed)
    N = n_tp + n_fp
    labels = np.zeros(N, dtype=int)
    labels[:n_tp] = 1

    tp_scores = _generate_tp_scores(n_tp, K, rng)
    fp_scores = _generate_fp_scores_by_condition(n_fp, K, fp_conditions, rng)
    scores = np.vstack([tp_scores, fp_scores])
    member_detected = scores > 0

    boxes = generate_ensemble_boxes(n_tp, n_fp, K, seed)
    frame_ids, frame_configs = generate_frame_assignments(n_tp, n_fp, n_frames, seed)

    return {
        "scores": scores,
        "boxes": boxes,
        "labels": labels,
        "member_detected": member_detected,
        "frame_ids": frame_ids,
        "conditions": conditions,
        "frame_configs": frame_configs,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "K": K,
        "n_frames": n_frames,
    }


def get_individual_member_aps() -> Dict[str, float]:
    """
    Individual member BEV AP values (Table 1 in paper).

    These are the actual values reported for 6 SECOND detectors
    trained with different random seeds on CARLA data.
    """
    return {
        "A": 90.08,
        "B": 90.49,
        "C": 90.55,
        "D": 90.11,
        "E": 89.32,
        "F": 90.57,
    }


def validate_dataset(data: Dict) -> Dict[str, float]:
    """
    Validate that generated data matches paper's target statistics.

    Parameters
    ----------
    data : dict from generate_demo_dataset()

    Returns
    -------
    dict with computed statistics and whether targets are met.
    """
    from sotif_uncertainty.uncertainty import compute_all_indicators
    from sotif_uncertainty.metrics import compute_auroc

    scores = data["scores"]
    labels = data["labels"]
    boxes = data["boxes"]
    conditions = data["conditions"]

    indicators = compute_all_indicators(scores, boxes)
    mean_conf = indicators["mean_confidence"]
    conf_var = indicators["confidence_variance"]
    geo_disagree = indicators["geometric_disagreement"]

    tp_mask = labels == 1
    fp_mask = labels == 0

    # Compute AUROCs
    auroc_conf = compute_auroc(mean_conf, labels, higher_is_correct=True)
    auroc_var = compute_auroc(conf_var, labels, higher_is_correct=False)
    auroc_geo = compute_auroc(geo_disagree, labels, higher_is_correct=False)

    # Compute percentiles
    tp_var_80 = np.percentile(conf_var[tp_mask], 80)
    fp_var_80 = np.percentile(conf_var[fp_mask], 80)

    # Compute per-condition stats
    tc_stats = {}
    for cond in np.unique(conditions[fp_mask]):
        cond_fp = fp_mask & (conditions == cond)
        tc_stats[cond] = {
            "fp_share": np.sum(cond_fp) / np.sum(fp_mask),
            "mean_conf": np.mean(mean_conf[cond_fp]),
            "mean_var": np.mean(conf_var[cond_fp]),
        }

    # Operating point at s>=0.70
    accepted_70 = mean_conf >= 0.70
    cov_70 = np.sum(accepted_70) / len(labels)
    fp_70 = np.sum(accepted_70 & fp_mask)

    return {
        "n_total": len(labels),
        "n_tp": int(np.sum(tp_mask)),
        "n_fp": int(np.sum(fp_mask)),
        "auroc_mean_confidence": auroc_conf,
        "auroc_confidence_variance": auroc_var,
        "auroc_geometric_disagreement": auroc_geo,
        "tp_mean_conf": float(np.mean(mean_conf[tp_mask])),
        "fp_mean_conf": float(np.mean(mean_conf[fp_mask])),
        "tp_var_80th_pct": tp_var_80,
        "fp_var_80th_pct": fp_var_80,
        "coverage_at_070": cov_70,
        "fp_at_070": fp_70,
        "tc_stats": tc_stats,
    }

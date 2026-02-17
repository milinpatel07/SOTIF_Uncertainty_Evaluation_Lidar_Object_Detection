"""
Synthetic demo data generator.

Generates ensemble outputs matching the statistical properties reported in
the paper for demonstration purposes. This allows anyone to run the full
evaluation pipeline without needing GPU resources or large datasets.

The generated data mirrors:
- 465 proposals: 135 TP, 330 FP (Section 5.1)
- 6 ensemble members (Section 4.2)
- 22 environmental configurations across 101 frames (Section 4.1)
- AUROC ~0.999 for mean confidence (Table 3)
- Heavy rain (38%) and night (29%) dominating FP (Table 7)
"""

import numpy as np
from typing import Dict, Optional, Tuple


# Environmental condition configurations (Table 2)
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

# Triggering condition categories for SOTIF analysis
TC_CATEGORIES = {
    "heavy_rain": {
        "configs": ["HeavyRainNoon", "HardRainNight", "ExtremeWeather"],
        "fp_share": 0.38,
    },
    "night": {
        "configs": ["NightDrivingClear", "ClearNight", "CloudyNight",
                     "WetNight", "MidRainNight", "FoggyNight"],
        "fp_share": 0.29,
    },
    "fog_visibility": {
        "configs": ["FoggyDay", "DustStorm", "SnowyConditions"],
        "fp_share": 0.21,
    },
    "other": {
        "configs": ["ClearNoon", "CloudyNoon", "LowSunNoon", "HighWindNoon",
                     "WetNoon", "WetCloudyNoon", "MidRainyNoon",
                     "DuskTransition", "EmergencyBraking",
                     "OvertakeMultiVehicle"],
        "fp_share": 0.12,
    },
}


def generate_ensemble_scores(
    n_tp: int = 135,
    n_fp: int = 330,
    K: int = 6,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic ensemble confidence scores for TP and FP detections.

    TP detections: high confidence, low variance across members.
    FP detections: low confidence, high variance across members.

    Statistical targets (from paper):
    - TP mean confidence: ~0.75-0.95
    - FP mean confidence: ~0.05-0.45 (some up to 0.65)
    - TP 80th percentile variance: ~0.00195
    - FP 80th percentile variance: ~0.00572
    - AUROC(mean_conf) -> ~0.999

    Parameters
    ----------
    n_tp : int
        Number of true positive proposals.
    n_fp : int
        Number of false positive proposals.
    K : int
        Number of ensemble members.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    scores : np.ndarray, shape (N, K)
        Per-member confidence scores.
    labels : np.ndarray, shape (N,)
        Correctness labels (1=TP, 0=FP).
    member_detected : np.ndarray, shape (N, K), dtype bool
        Whether each member detected the proposal.
    """
    rng = np.random.RandomState(seed)
    N = n_tp + n_fp

    scores = np.zeros((N, K))
    labels = np.zeros(N, dtype=int)
    member_detected = np.ones((N, K), dtype=bool)

    # --- True Positives ---
    labels[:n_tp] = 1

    for i in range(n_tp):
        # Base confidence for this TP: high (0.70 to 0.98)
        base = rng.uniform(0.70, 0.98)
        # Small per-member variation (low epistemic uncertainty)
        noise_scale = rng.uniform(0.01, 0.06)
        for k in range(K):
            scores[i, k] = np.clip(base + rng.normal(0, noise_scale), 0.1, 0.999)
        # Most members detect TP (occasionally 1 member misses)
        if rng.random() < 0.05:
            miss_k = rng.randint(0, K)
            scores[i, miss_k] = 0.0
            member_detected[i, miss_k] = False

    # --- False Positives ---
    labels[n_tp:] = 0

    for i in range(n_tp, N):
        # Most FP have low confidence
        if rng.random() < 0.85:
            # Low-confidence FP
            base = rng.uniform(0.02, 0.35)
            noise_scale = rng.uniform(0.02, 0.10)
        elif rng.random() < 0.7:
            # Medium-confidence FP
            base = rng.uniform(0.35, 0.55)
            noise_scale = rng.uniform(0.03, 0.08)
        else:
            # Higher-confidence FP (harder cases, up to 0.65)
            base = rng.uniform(0.50, 0.65)
            noise_scale = rng.uniform(0.02, 0.06)

        # FP often detected by fewer members
        n_detect = rng.choice([1, 2, 3, 4, 5, 6], p=[0.15, 0.25, 0.25, 0.20, 0.10, 0.05])
        detecting_members = rng.choice(K, size=n_detect, replace=False)

        for k in range(K):
            if k in detecting_members:
                scores[i, k] = np.clip(base + rng.normal(0, noise_scale), 0.01, 0.999)
            else:
                scores[i, k] = 0.0
                member_detected[i, k] = False

    return scores, labels, member_detected


def generate_ensemble_boxes(
    n_tp: int = 135,
    n_fp: int = 330,
    K: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic 3D bounding boxes for ensemble proposals.

    TP: members agree on position (low geometric disagreement).
    FP: members disagree on position (high geometric disagreement).

    Parameters
    ----------
    Returns
    -------
    boxes : np.ndarray, shape (N, K, 7)
        Bounding boxes [x, y, z, w, l, h, yaw] per proposal per member.
        NaN for members that did not detect the proposal.
    """
    rng = np.random.RandomState(seed + 1)
    N = n_tp + n_fp
    boxes = np.full((N, K, 7), np.nan)

    # Typical vehicle dimensions
    base_w, base_l, base_h = 1.8, 4.5, 1.6

    for i in range(N):
        # Random base position
        x = rng.uniform(5, 80)
        y = rng.uniform(-10, 10)
        z = rng.uniform(-1.5, -0.5)
        yaw = rng.uniform(-np.pi, np.pi)

        if i < n_tp:
            # TP: tight clustering, all members close to true position
            pos_noise = 0.2
            dim_noise = 0.05
            yaw_noise = 0.05
        else:
            # FP: wider spread
            pos_noise = rng.uniform(0.5, 2.0)
            dim_noise = rng.uniform(0.1, 0.3)
            yaw_noise = rng.uniform(0.1, 0.5)

        for k in range(K):
            # Check if this member detected the proposal (use same seed logic)
            if i < n_tp:
                detected = True
                if rng.random() < 0.05:
                    detected = False
            else:
                detected = rng.random() < 0.55

            if detected:
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
) -> np.ndarray:
    """
    Assign proposals to frames.

    Distributes detections across frames with more FP concentrated
    in adverse condition frames.

    Returns
    -------
    frame_ids : np.ndarray, shape (N,)
    """
    rng = np.random.RandomState(seed + 2)
    N = n_tp + n_fp
    frame_ids = np.zeros(N, dtype=int)

    # Distribute TP roughly equally across frames
    tp_frames = rng.choice(n_frames, size=n_tp, replace=True)
    frame_ids[:n_tp] = tp_frames

    # Distribute FP with concentration in adverse frames
    # First 40 frames: "good" conditions (fewer FP)
    # Frames 40-70: "moderate" conditions
    # Frames 70-101: "adverse" conditions (more FP)
    fp_frame_probs = np.ones(n_frames)
    fp_frame_probs[:40] *= 0.5     # good conditions: fewer FP
    fp_frame_probs[40:70] *= 1.5   # moderate: average
    fp_frame_probs[70:] *= 3.0     # adverse: many FP
    fp_frame_probs /= fp_frame_probs.sum()

    fp_frames = rng.choice(n_frames, size=n_fp, replace=True, p=fp_frame_probs)
    frame_ids[n_tp:] = fp_frames

    return frame_ids


def generate_condition_assignments(
    n_tp: int = 135,
    n_fp: int = 330,
    seed: int = 42,
) -> np.ndarray:
    """
    Assign triggering condition categories to detections.

    FP distribution matches Table 7:
    - heavy_rain: 38%
    - night: 29%
    - fog_visibility: 21%
    - other: 12%

    TP are more evenly distributed with bias toward clear conditions.

    Returns
    -------
    conditions : np.ndarray, shape (N,), dtype str
    """
    rng = np.random.RandomState(seed + 3)
    N = n_tp + n_fp
    conditions = np.empty(N, dtype="U20")

    categories = list(TC_CATEGORIES.keys())

    # TP distribution: more in "other" (clear) conditions
    tp_probs = [0.10, 0.15, 0.15, 0.60]  # heavy_rain, night, fog, other
    tp_cats = rng.choice(categories, size=n_tp, p=tp_probs)
    conditions[:n_tp] = tp_cats

    # FP distribution: matches Table 7
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
    Generate a complete synthetic dataset for demonstration.

    The generated data matches the statistical properties from the paper:
    - 465 proposals (135 TP, 330 FP)
    - 6 ensemble members
    - 22 conditions across 101 frames
    - AUROC ~0.999 for mean confidence
    - Heavy rain and night dominate FP

    Parameters
    ----------
    n_tp : int
        Number of true positive proposals.
    n_fp : int
        Number of false positive proposals.
    K : int
        Number of ensemble members.
    n_frames : int
        Number of frames.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'scores': (N, K) ensemble confidence scores
        'boxes': (N, K, 7) ensemble bounding boxes
        'labels': (N,) correctness labels
        'member_detected': (N, K) detection flags
        'frame_ids': (N,) frame assignments
        'conditions': (N,) triggering condition categories
        'n_tp': int
        'n_fp': int
        'K': int
        'n_frames': int
    """
    scores, labels, member_detected = generate_ensemble_scores(n_tp, n_fp, K, seed)
    boxes = generate_ensemble_boxes(n_tp, n_fp, K, seed)
    frame_ids = generate_frame_assignments(n_tp, n_fp, n_frames, seed)
    conditions = generate_condition_assignments(n_tp, n_fp, seed)

    return {
        "scores": scores,
        "boxes": boxes,
        "labels": labels,
        "member_detected": member_detected,
        "frame_ids": frame_ids,
        "conditions": conditions,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "K": K,
        "n_frames": n_frames,
    }


def get_individual_member_aps(seed: int = 42) -> Dict[str, float]:
    """
    Return simulated individual member AP values (Table 1).

    These represent the BEV AP at IoU >= 0.5 for the vehicle class,
    matching the range reported in the paper (89.3% - 90.6%).
    """
    rng = np.random.RandomState(seed + 10)
    member_names = ["A", "B", "C", "D", "E", "F"]
    # Match reported values
    aps = {
        "A": 90.08,
        "B": 90.49,
        "C": 90.55,
        "D": 90.11,
        "E": 89.32,
        "F": 90.57,
    }
    return aps

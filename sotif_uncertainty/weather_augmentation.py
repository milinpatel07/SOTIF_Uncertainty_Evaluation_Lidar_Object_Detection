"""
Physics-Based Weather Augmentation for LiDAR Point Clouds.

Applies realistic weather degradation effects to LiDAR point clouds,
enabling SOTIF evaluation under adverse environmental conditions without
requiring a CARLA simulator instance.

Supported weather effects:
    - Rain: random scatter noise, intensity attenuation, point dropout
    - Fog: range-dependent attenuation using Beer-Lambert law, point loss
    - Snow: ground-level scatter, accumulation occlusion, intensity drop
    - Spray: vehicle-induced water spray (wet road conditions)
    - Combined: multiple simultaneous weather effects

The augmentation parameters are calibrated against real-world LiDAR
degradation studies and CARLA simulator measurements.

Key insight for SOTIF: CARLA's built-in LiDAR sensor does NOT model
weather effects on point returns. This module bridges that gap by
applying physics-based post-processing augmentation, following the
methodology of PCSim (PJLab-ADG) and related work.

References:
    Hahner et al. (2021). "Fog Simulation on Real LiDAR Point Clouds." ICCV.
    Hahner et al. (2022). "LiDAR Snowfall Simulation." CVPR.
    Li et al. (2023). "Realistic Rainy Weather Simulation for LiDARs." arxiv:2312.12772.
    ISO 21448:2022 -- Safety of the Intended Functionality.
"""

import numpy as np
from typing import Dict, Optional, Tuple


# =========================================================================
# Rain augmentation
# =========================================================================

def augment_rain(
    points: np.ndarray,
    rain_rate: float = 25.0,
    scatter_density: float = 0.05,
    attenuation_coeff: float = 0.02,
    dropout_prob: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply rain effects to a LiDAR point cloud.

    Rain degrades LiDAR in three ways:
    1. Random scatter: rain drops create spurious return points in air
    2. Attenuation: water reduces return intensity (Beer-Lambert)
    3. Dropout: some returns are lost entirely at high rain rates

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Input point cloud [x, y, z, intensity].
    rain_rate : float
        Rain intensity in mm/h. Light <2.5, moderate 2.5-10,
        heavy 10-50, extreme >50.
    scatter_density : float
        Fraction of additional scatter points relative to N.
        Scaled by rain_rate internally.
    attenuation_coeff : float
        Attenuation coefficient per meter. Scaled by rain_rate.
    dropout_prob : float
        Base probability of dropping a point. Scaled by rain_rate
        and distance. If 0, computed from rain_rate automatically.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Augmented point cloud.
    """
    rng = np.random.RandomState(seed)
    N = len(points)

    if N == 0:
        return points.copy()

    result = points.copy()

    # Scale parameters by rain rate (normalized to 50 mm/h = heavy rain)
    rate_factor = min(rain_rate / 50.0, 2.0)

    # 1. Intensity attenuation (Beer-Lambert: I = I0 * exp(-alpha * r))
    distances = np.sqrt(result[:, 0] ** 2 + result[:, 1] ** 2 + result[:, 2] ** 2)
    alpha = attenuation_coeff * rate_factor
    attenuation = np.exp(-alpha * distances)
    result[:, 3] *= attenuation

    # 2. Distance-dependent dropout
    if dropout_prob == 0.0:
        dropout_prob = min(0.15 * rate_factor, 0.30)

    # Farther points more likely to drop
    max_dist = np.max(distances) + 1e-6
    dist_factor = distances / max_dist
    drop_probs = dropout_prob * (0.3 + 0.7 * dist_factor)
    keep_mask = rng.random(N) > drop_probs
    result = result[keep_mask]

    # 3. Random scatter noise (rain drops in air)
    n_scatter = max(1, int(N * scatter_density * rate_factor))
    scatter_range = min(30.0, 10.0 + 20.0 * rate_factor)

    scatter_x = rng.uniform(-scatter_range, scatter_range, n_scatter)
    scatter_y = rng.uniform(-scatter_range, scatter_range, n_scatter)
    scatter_z = rng.uniform(-2.0, 3.0, n_scatter)
    scatter_intensity = rng.uniform(0.01, 0.10, n_scatter)

    scatter_points = np.stack(
        [scatter_x, scatter_y, scatter_z, scatter_intensity], axis=1
    ).astype(np.float32)

    return np.vstack([result, scatter_points]).astype(np.float32)


# =========================================================================
# Fog augmentation
# =========================================================================

def augment_fog(
    points: np.ndarray,
    visibility: float = 50.0,
    beta: Optional[float] = None,
    soft_threshold: float = 0.8,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply fog effects to a LiDAR point cloud.

    Fog attenuates LiDAR returns following the Beer-Lambert law.
    Points beyond the fog visibility range are progressively lost.

    The extinction coefficient beta is derived from meteorological
    visibility V using: beta = 3.912 / V (Koschmieder's law).

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Input point cloud [x, y, z, intensity].
    visibility : float
        Meteorological visibility in meters. Dense fog <50m,
        moderate fog 50-200m, light fog 200-1000m.
    beta : float, optional
        Override extinction coefficient. If None, computed from visibility.
    soft_threshold : float
        Fraction of visibility at which dropout begins (0-1).
        Points at this fraction of visibility start being dropped
        probabilistically.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Augmented point cloud.
    """
    rng = np.random.RandomState(seed)
    N = len(points)

    if N == 0:
        return points.copy()

    result = points.copy()

    # Compute extinction coefficient from visibility (Koschmieder's law)
    if beta is None:
        beta = 3.912 / max(visibility, 1.0)

    # Compute distances from sensor
    distances = np.sqrt(
        result[:, 0] ** 2 + result[:, 1] ** 2 + result[:, 2] ** 2
    )

    # Intensity attenuation (Beer-Lambert)
    attenuation = np.exp(-2.0 * beta * distances)  # factor 2 for round-trip
    result[:, 3] *= attenuation

    # Progressive point dropout beyond soft threshold
    dropout_start = visibility * soft_threshold
    dropout_probs = np.clip(
        (distances - dropout_start) / (visibility - dropout_start + 1e-6),
        0.0,
        0.95,
    )
    keep_mask = rng.random(N) > dropout_probs

    # Also drop points with very low remaining intensity
    keep_mask &= result[:, 3] > 0.005

    result = result[keep_mask]

    # Add fog backscatter (close-range noise from fog particles)
    n_backscatter = max(1, int(N * 0.01 * beta))
    if n_backscatter > 0:
        bs_dist = rng.exponential(scale=5.0 / beta, size=n_backscatter)
        bs_dist = np.clip(bs_dist, 0.5, visibility * 0.5)
        bs_angle_h = rng.uniform(-np.pi, np.pi, n_backscatter)
        bs_angle_v = rng.uniform(-0.4, 0.1, n_backscatter)

        bs_x = bs_dist * np.cos(bs_angle_v) * np.cos(bs_angle_h)
        bs_y = bs_dist * np.cos(bs_angle_v) * np.sin(bs_angle_h)
        bs_z = bs_dist * np.sin(bs_angle_v)
        bs_intensity = rng.uniform(0.01, 0.08, n_backscatter)

        backscatter = np.stack(
            [bs_x, bs_y, bs_z, bs_intensity], axis=1
        ).astype(np.float32)

        result = np.vstack([result, backscatter])

    return result.astype(np.float32)


# =========================================================================
# Snow augmentation
# =========================================================================

def augment_snow(
    points: np.ndarray,
    snowfall_rate: float = 2.5,
    accumulation_height: float = 0.05,
    scatter_density: float = 0.03,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply snowfall effects to a LiDAR point cloud.

    Snow affects LiDAR through:
    1. Scatter: snowflakes create false returns
    2. Attenuation: reduced return intensity
    3. Ground accumulation: raises ground plane, occludes low objects

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Input point cloud [x, y, z, intensity].
    snowfall_rate : float
        Snowfall intensity in mm/h water equivalent.
        Light <1, moderate 1-4, heavy >4.
    accumulation_height : float
        Snow ground accumulation in meters.
    scatter_density : float
        Density of snowflake scatter points (fraction of N).
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Augmented point cloud.
    """
    rng = np.random.RandomState(seed)
    N = len(points)

    if N == 0:
        return points.copy()

    result = points.copy()
    rate_factor = min(snowfall_rate / 5.0, 2.0)

    # 1. Intensity attenuation
    distances = np.sqrt(
        result[:, 0] ** 2 + result[:, 1] ** 2 + result[:, 2] ** 2
    )
    alpha = 0.01 * rate_factor
    result[:, 3] *= np.exp(-alpha * distances)

    # 2. Point dropout (distance dependent)
    dropout = 0.05 * rate_factor
    drop_probs = dropout * (0.2 + 0.8 * distances / (np.max(distances) + 1e-6))
    keep_mask = rng.random(N) > drop_probs
    result = result[keep_mask]

    # 3. Ground plane shift (accumulation)
    if accumulation_height > 0:
        ground_z = np.percentile(result[:, 2], 5)
        near_ground = result[:, 2] < ground_z + 0.3
        result[near_ground, 2] += accumulation_height
        result[near_ground, 3] *= 0.7

    # 4. Snowflake scatter
    n_scatter = max(1, int(N * scatter_density * rate_factor))
    flake_x = rng.uniform(-20, 40, n_scatter)
    flake_y = rng.uniform(-20, 20, n_scatter)
    flake_z = rng.uniform(-1.5, 3.0, n_scatter)
    flake_intensity = rng.uniform(0.02, 0.15, n_scatter)

    scatter = np.stack(
        [flake_x, flake_y, flake_z, flake_intensity], axis=1
    ).astype(np.float32)

    return np.vstack([result, scatter]).astype(np.float32)


# =========================================================================
# Spray augmentation (wet road)
# =========================================================================

def augment_spray(
    points: np.ndarray,
    spray_intensity: float = 0.5,
    vehicle_positions: Optional[np.ndarray] = None,
    spray_radius: float = 4.0,
    spray_height: float = 1.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply road spray effects from leading vehicles.

    Spray from vehicle tires on wet roads creates dense noise
    clouds behind vehicles, heavily degrading LiDAR perception.

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Input point cloud [x, y, z, intensity].
    spray_intensity : float
        Spray intensity factor (0-1). Higher = more spray.
    vehicle_positions : np.ndarray, shape (V, 3), optional
        Positions of vehicles generating spray [x, y, z].
        If None, spray is generated at random forward positions.
    spray_radius : float
        Lateral spread of spray cloud in meters.
    spray_height : float
        Maximum height of spray cloud in meters.
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Augmented point cloud.
    """
    rng = np.random.RandomState(seed)
    N = len(points)

    if N == 0:
        return points.copy()

    result = points.copy()

    if vehicle_positions is None:
        n_vehicles = rng.randint(1, 4)
        vehicle_positions = np.zeros((n_vehicles, 3))
        vehicle_positions[:, 0] = rng.uniform(10, 40, n_vehicles)
        vehicle_positions[:, 1] = rng.uniform(-4, 4, n_vehicles)
        vehicle_positions[:, 2] = -1.5

    n_spray_per_vehicle = max(1, int(200 * spray_intensity))

    spray_points = []
    for vpos in vehicle_positions:
        # Spray behind and around the vehicle
        sx = vpos[0] - rng.exponential(2.0, n_spray_per_vehicle)
        sy = vpos[1] + rng.normal(0, spray_radius * 0.5, n_spray_per_vehicle)
        sz = vpos[2] + rng.exponential(spray_height * 0.3, n_spray_per_vehicle)
        si = rng.uniform(0.01, 0.12, n_spray_per_vehicle)

        spray = np.stack([sx, sy, sz, si], axis=1)
        spray_points.append(spray)

    if spray_points:
        all_spray = np.vstack(spray_points).astype(np.float32)
        result = np.vstack([result, all_spray])

    return result.astype(np.float32)


# =========================================================================
# Combined weather augmentation
# =========================================================================

def augment_weather(
    points: np.ndarray,
    weather_config: Dict,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Apply combined weather augmentation based on a configuration dictionary.

    This is the primary entry point for weather augmentation. It accepts
    a configuration dict matching the CARLA weather parameters and applies
    the appropriate combination of effects.

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Input point cloud [x, y, z, intensity].
    weather_config : dict
        Weather parameters. Expected keys:
            'precipitation' : float (0-100, rain/snow intensity)
            'fog_density' : float (0-100, fog intensity)
            'wetness' : float (0-100, road wetness for spray)
            'cloudiness' : float (0-100, not directly used for LiDAR)
            'wind_intensity' : float (0-100, affects spray spread)
            'sun_altitude_angle' : float (-90 to 90, <0 = night)
        Additional optional keys:
            'snow' : bool (if True, precipitation is snow instead of rain)
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray, shape (M, 4)
        Augmented point cloud.
    """
    rng = np.random.RandomState(seed)
    result = points.copy()

    precip = weather_config.get("precipitation", 0.0)
    fog = weather_config.get("fog_density", 0.0)
    wetness = weather_config.get("wetness", 0.0)
    wind = weather_config.get("wind_intensity", 0.0)
    is_snow = weather_config.get("snow", False)
    sun_alt = weather_config.get("sun_altitude_angle", 70.0)

    # Determine if precipitation is snow based on config name or flag
    if is_snow or (precip > 20 and sun_alt < 5):
        # Snow conditions
        if precip > 20:
            result = augment_snow(
                result,
                snowfall_rate=precip / 20.0,
                accumulation_height=precip / 1000.0,
                seed=rng.randint(0, 2**31),
            )
    else:
        # Rain conditions
        if precip > 20:
            rain_rate = precip / 2.0  # Map 0-100 to 0-50 mm/h
            result = augment_rain(
                result,
                rain_rate=rain_rate,
                seed=rng.randint(0, 2**31),
            )

    # Fog effects
    if fog > 10:
        # Map fog_density 0-100 to visibility 1000m-10m
        visibility = max(10.0, 1000.0 * (1.0 - fog / 100.0))
        result = augment_fog(
            result,
            visibility=visibility,
            seed=rng.randint(0, 2**31),
        )

    # Spray effects from wet roads
    if wetness > 40 and precip > 10:
        spray_intensity = min(1.0, wetness / 100.0 * precip / 50.0)
        spray_radius = 2.0 + wind / 25.0
        result = augment_spray(
            result,
            spray_intensity=spray_intensity,
            spray_radius=spray_radius,
            seed=rng.randint(0, 2**31),
        )

    return result.astype(np.float32)


# =========================================================================
# Predefined weather presets matching CARLA configurations (Table 2)
# =========================================================================

WEATHER_PRESETS = {
    "clear": {
        "precipitation": 0.0,
        "fog_density": 0.0,
        "wetness": 0.0,
        "wind_intensity": 10.0,
        "sun_altitude_angle": 70.0,
    },
    "light_rain": {
        "precipitation": 30.0,
        "fog_density": 5.0,
        "wetness": 60.0,
        "wind_intensity": 30.0,
        "sun_altitude_angle": 50.0,
    },
    "heavy_rain": {
        "precipitation": 100.0,
        "fog_density": 30.0,
        "wetness": 100.0,
        "wind_intensity": 70.0,
        "sun_altitude_angle": 30.0,
    },
    "light_fog": {
        "precipitation": 0.0,
        "fog_density": 30.0,
        "wetness": 20.0,
        "wind_intensity": 10.0,
        "sun_altitude_angle": 45.0,
    },
    "dense_fog": {
        "precipitation": 0.0,
        "fog_density": 80.0,
        "wetness": 30.0,
        "wind_intensity": 10.0,
        "sun_altitude_angle": 45.0,
    },
    "snow": {
        "precipitation": 60.0,
        "fog_density": 30.0,
        "wetness": 40.0,
        "wind_intensity": 40.0,
        "sun_altitude_angle": 20.0,
        "snow": True,
    },
    "extreme": {
        "precipitation": 90.0,
        "fog_density": 60.0,
        "wetness": 100.0,
        "wind_intensity": 90.0,
        "sun_altitude_angle": 20.0,
    },
}


def get_weather_preset(name: str) -> Dict:
    """
    Get a predefined weather configuration by name.

    Parameters
    ----------
    name : str
        One of: clear, light_rain, heavy_rain, light_fog, dense_fog,
        snow, extreme.

    Returns
    -------
    dict
        Weather configuration dictionary.
    """
    if name not in WEATHER_PRESETS:
        available = ", ".join(WEATHER_PRESETS.keys())
        raise ValueError(
            f"Unknown weather preset '{name}'. Available: {available}"
        )
    return WEATHER_PRESETS[name].copy()


def compute_weather_severity(weather_config: Dict) -> Dict[str, float]:
    """
    Compute weather severity metrics for SOTIF triggering condition analysis.

    Maps weather configuration parameters to a normalized severity score
    per degradation category. Used for automated TC classification.

    Parameters
    ----------
    weather_config : dict
        Weather configuration dictionary.

    Returns
    -------
    dict with keys:
        'rain_severity' : float in [0, 1]
        'fog_severity' : float in [0, 1]
        'spray_severity' : float in [0, 1]
        'overall_severity' : float in [0, 1]
        'tc_category' : str (predicted triggering condition category)
    """
    precip = weather_config.get("precipitation", 0.0) / 100.0
    fog = weather_config.get("fog_density", 0.0) / 100.0
    wetness = weather_config.get("wetness", 0.0) / 100.0
    sun_alt = weather_config.get("sun_altitude_angle", 70.0)
    is_night = sun_alt < 0

    rain_sev = min(1.0, precip * 1.2)
    fog_sev = min(1.0, fog * 1.3)
    spray_sev = min(1.0, wetness * precip)
    night_sev = 1.0 if is_night else 0.0

    overall = max(rain_sev, fog_sev, spray_sev, night_sev * 0.5)

    # Classify triggering condition
    if rain_sev > 0.6:
        tc_category = "heavy_rain"
    elif is_night and (rain_sev > 0.2 or fog_sev > 0.2):
        tc_category = "night"
    elif is_night:
        tc_category = "night"
    elif fog_sev > 0.3:
        tc_category = "fog_visibility"
    else:
        tc_category = "other"

    return {
        "rain_severity": float(rain_sev),
        "fog_severity": float(fog_sev),
        "spray_severity": float(spray_sev),
        "night_severity": float(night_sev),
        "overall_severity": float(overall),
        "tc_category": tc_category,
    }

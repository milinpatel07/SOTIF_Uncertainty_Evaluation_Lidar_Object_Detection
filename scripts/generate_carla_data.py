"""
Generate LiDAR point cloud datasets from CARLA simulator.

This script automates data collection from CARLA in KITTI format,
with configurable weather/lighting conditions for SOTIF analysis.

The paper uses 22 environmental configurations across 6 categories
(Table 2), generating data in KITTI format for direct use with
OpenPCDet and this evaluation pipeline.

Prerequisites:
    - CARLA simulator (>= 0.9.13) running on localhost:2000
    - Python CARLA client: pip install carla
    - (Optional) CARLA-KITTI bridge for automated conversion

Usage:
    # Start CARLA first:
    #   ./CarlaUE4.sh -RenderOffScreen -quality-level=Low

    # Generate data with all 22 configurations
    python scripts/generate_carla_data.py --output_dir data/carla

    # Generate for specific conditions only
    python scripts/generate_carla_data.py \
        --output_dir data/carla \
        --conditions heavy_rain night fog

    # Use specific CARLA port
    python scripts/generate_carla_data.py \
        --output_dir data/carla \
        --carla_host localhost --carla_port 2000

    # Generate more frames per configuration
    python scripts/generate_carla_data.py \
        --output_dir data/carla \
        --frames_per_config 50

Data output structure (KITTI format):
    data/carla/
    ├── training/
    │   ├── velodyne/    # .bin point cloud files
    │   ├── label_2/     # .txt label files (KITTI format)
    │   ├── calib/       # .txt calibration files
    │   └── image_2/     # .png camera images
    ├── ImageSets/
    │   ├── train.txt
    │   └── val.txt
    └── conditions.json  # Per-frame condition metadata

Reference:
    Section 4.1 of the paper (Environmental configurations).
    Table 2: 22 configurations across 6 categories.
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path

# Environmental configuration definitions matching Table 2 in the paper
WEATHER_CONFIGS = {
    # Category 1: Clear / Overcast
    "ClearNoon": {
        "category": "clear_overcast",
        "description": "Clear sky, midday sun, dry roads",
        "carla_weather": {
            "cloudiness": 10.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 70.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 20.0,
        },
        "sun_azimuth": 180.0,
    },
    "CloudyNoon": {
        "category": "clear_overcast",
        "description": "Overcast sky, diffuse lighting",
        "carla_weather": {
            "cloudiness": 80.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 60.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 30.0,
        },
        "sun_azimuth": 200.0,
    },
    "LowSunNoon": {
        "category": "clear_overcast",
        "description": "Low sun angle creating long shadows",
        "carla_weather": {
            "cloudiness": 20.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 15.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 10.0,
        },
        "sun_azimuth": 270.0,
    },
    "HighWindNoon": {
        "category": "clear_overcast",
        "description": "High wind, airborne dust particles",
        "carla_weather": {
            "cloudiness": 30.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 65.0,
            "fog_density": 5.0,
            "wetness": 0.0,
            "wind_intensity": 90.0,
        },
        "sun_azimuth": 190.0,
    },
    # Category 2: Precipitation
    "WetNoon": {
        "category": "precipitation",
        "description": "Wet roads, no active rain, some spray",
        "carla_weather": {
            "cloudiness": 60.0,
            "precipitation": 10.0,
            "sun_altitude_angle": 50.0,
            "fog_density": 5.0,
            "wetness": 80.0,
            "wind_intensity": 30.0,
        },
        "sun_azimuth": 180.0,
    },
    "WetCloudyNoon": {
        "category": "precipitation",
        "description": "Wet roads under overcast sky",
        "carla_weather": {
            "cloudiness": 90.0,
            "precipitation": 20.0,
            "sun_altitude_angle": 40.0,
            "fog_density": 10.0,
            "wetness": 90.0,
            "wind_intensity": 40.0,
        },
        "sun_azimuth": 180.0,
    },
    "MidRainyNoon": {
        "category": "precipitation",
        "description": "Moderate rain, reduced visibility",
        "carla_weather": {
            "cloudiness": 90.0,
            "precipitation": 60.0,
            "sun_altitude_angle": 35.0,
            "fog_density": 15.0,
            "wetness": 100.0,
            "wind_intensity": 50.0,
        },
        "sun_azimuth": 180.0,
    },
    "HeavyRainNoon": {
        "category": "precipitation",
        "description": "Heavy rain, significant noise in LiDAR returns",
        "carla_weather": {
            "cloudiness": 100.0,
            "precipitation": 100.0,
            "sun_altitude_angle": 30.0,
            "fog_density": 30.0,
            "wetness": 100.0,
            "wind_intensity": 70.0,
        },
        "sun_azimuth": 180.0,
    },
    # Category 3: Reduced Visibility
    "FoggyDay": {
        "category": "reduced_visibility",
        "description": "Dense fog, severely reduced LiDAR range",
        "carla_weather": {
            "cloudiness": 80.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 45.0,
            "fog_density": 80.0,
            "wetness": 30.0,
            "wind_intensity": 10.0,
        },
        "sun_azimuth": 180.0,
    },
    "DustStorm": {
        "category": "reduced_visibility",
        "description": "Dust storm, particulate scatter",
        "carla_weather": {
            "cloudiness": 60.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 50.0,
            "fog_density": 60.0,
            "wetness": 0.0,
            "wind_intensity": 100.0,
        },
        "sun_azimuth": 220.0,
    },
    "SnowyConditions": {
        "category": "reduced_visibility",
        "description": "Snowfall with ground accumulation",
        "carla_weather": {
            "cloudiness": 100.0,
            "precipitation": 80.0,
            "sun_altitude_angle": 25.0,
            "fog_density": 40.0,
            "wetness": 50.0,
            "wind_intensity": 40.0,
        },
        "sun_azimuth": 180.0,
    },
    # Category 4: Night Clear
    "NightDrivingClear": {
        "category": "night_clear",
        "description": "Clear night, streetlights only",
        "carla_weather": {
            "cloudiness": 10.0,
            "precipitation": 0.0,
            "sun_altitude_angle": -30.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 10.0,
        },
        "sun_azimuth": 180.0,
    },
    "ClearNight": {
        "category": "night_clear",
        "description": "Clear night with moonlight",
        "carla_weather": {
            "cloudiness": 5.0,
            "precipitation": 0.0,
            "sun_altitude_angle": -45.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 5.0,
        },
        "sun_azimuth": 0.0,
    },
    "CloudyNight": {
        "category": "night_clear",
        "description": "Overcast night, minimal ambient light",
        "carla_weather": {
            "cloudiness": 90.0,
            "precipitation": 0.0,
            "sun_altitude_angle": -50.0,
            "fog_density": 5.0,
            "wetness": 0.0,
            "wind_intensity": 20.0,
        },
        "sun_azimuth": 0.0,
    },
    # Category 5: Night Adverse
    "WetNight": {
        "category": "night_adverse",
        "description": "Wet roads at night, reflections",
        "carla_weather": {
            "cloudiness": 70.0,
            "precipitation": 10.0,
            "sun_altitude_angle": -40.0,
            "fog_density": 10.0,
            "wetness": 80.0,
            "wind_intensity": 30.0,
        },
        "sun_azimuth": 0.0,
    },
    "MidRainNight": {
        "category": "night_adverse",
        "description": "Moderate rain at night",
        "carla_weather": {
            "cloudiness": 90.0,
            "precipitation": 60.0,
            "sun_altitude_angle": -35.0,
            "fog_density": 20.0,
            "wetness": 100.0,
            "wind_intensity": 50.0,
        },
        "sun_azimuth": 0.0,
    },
    "HardRainNight": {
        "category": "night_adverse",
        "description": "Heavy rain at night, severe conditions",
        "carla_weather": {
            "cloudiness": 100.0,
            "precipitation": 100.0,
            "sun_altitude_angle": -30.0,
            "fog_density": 35.0,
            "wetness": 100.0,
            "wind_intensity": 80.0,
        },
        "sun_azimuth": 0.0,
    },
    "FoggyNight": {
        "category": "night_adverse",
        "description": "Fog at night, extremely reduced visibility",
        "carla_weather": {
            "cloudiness": 80.0,
            "precipitation": 0.0,
            "sun_altitude_angle": -40.0,
            "fog_density": 70.0,
            "wetness": 20.0,
            "wind_intensity": 10.0,
        },
        "sun_azimuth": 0.0,
    },
    # Category 6: Compound / Dynamic
    "DuskTransition": {
        "category": "compound_dynamic",
        "description": "Dusk with rapidly changing light",
        "carla_weather": {
            "cloudiness": 40.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 0.0,
            "fog_density": 5.0,
            "wetness": 0.0,
            "wind_intensity": 15.0,
        },
        "sun_azimuth": 270.0,
    },
    "EmergencyBraking": {
        "category": "compound_dynamic",
        "description": "Clear conditions, emergency scenario",
        "carla_weather": {
            "cloudiness": 20.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 60.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 10.0,
        },
        "sun_azimuth": 180.0,
    },
    "OvertakeMultiVehicle": {
        "category": "compound_dynamic",
        "description": "Multi-vehicle overtake scenario",
        "carla_weather": {
            "cloudiness": 30.0,
            "precipitation": 0.0,
            "sun_altitude_angle": 55.0,
            "fog_density": 0.0,
            "wetness": 0.0,
            "wind_intensity": 20.0,
        },
        "sun_azimuth": 200.0,
    },
    "ExtremeWeather": {
        "category": "compound_dynamic",
        "description": "Extreme combination: rain + fog + wind",
        "carla_weather": {
            "cloudiness": 100.0,
            "precipitation": 90.0,
            "sun_altitude_angle": 20.0,
            "fog_density": 60.0,
            "wetness": 100.0,
            "wind_intensity": 90.0,
        },
        "sun_azimuth": 180.0,
    },
}

# Triggering condition category mappings (for SOTIF analysis)
TC_CATEGORY_MAP = {
    "HeavyRainNoon": "heavy_rain",
    "HardRainNight": "heavy_rain",
    "ExtremeWeather": "heavy_rain",
    "NightDrivingClear": "night",
    "ClearNight": "night",
    "CloudyNight": "night",
    "WetNight": "night",
    "MidRainNight": "night",
    "FoggyNight": "night",
    "FoggyDay": "fog_visibility",
    "DustStorm": "fog_visibility",
    "SnowyConditions": "fog_visibility",
}


def collect_carla_frame(
    world,
    config_name: str,
    config: dict,
    n_vehicles: int = 8,
    n_pedestrians: int = 3,
    seed: int = 42,
) -> dict:
    """
    Collect a single LiDAR frame from a running CARLA instance.

    Sets weather, spawns actors, captures LiDAR data, extracts ground
    truth bounding boxes, then cleans up.

    Parameters
    ----------
    world : carla.World
        CARLA world handle.
    config_name : str
        Name from WEATHER_CONFIGS.
    config : dict
        Weather configuration dictionary.
    n_vehicles : int
        Number of vehicles to spawn.
    n_pedestrians : int
        Number of pedestrians to spawn.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys matching generate_synthetic_lidar_frame output.
    """
    import carla
    import time

    rng = np.random.RandomState(seed)
    weather_params = config["carla_weather"]

    # Set weather
    weather = carla.WeatherParameters(
        cloudiness=weather_params["cloudiness"],
        precipitation=weather_params["precipitation"],
        sun_altitude_angle=weather_params.get("sun_altitude_angle",
                                               config.get("sun_azimuth", 70.0)),
        fog_density=weather_params["fog_density"],
        wetness=weather_params["wetness"],
        wind_intensity=weather_params["wind_intensity"],
    )
    if hasattr(weather, "sun_azimuth_angle"):
        weather.sun_azimuth_angle = config.get("sun_azimuth", 180.0)
    world.set_weather(weather)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    if len(spawn_points) == 0:
        raise RuntimeError("No spawn points available in CARLA map")

    actors_to_destroy = []

    try:
        # Spawn ego vehicle with LiDAR sensor
        ego_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        ego_transform = rng.choice(spawn_points)
        ego_vehicle = world.try_spawn_actor(ego_bp, ego_transform)
        if ego_vehicle is None:
            ego_transform = spawn_points[0]
            ego_vehicle = world.spawn_actor(ego_bp, ego_transform)
        actors_to_destroy.append(ego_vehicle)

        # Attach LiDAR sensor (64-channel, matching KITTI Velodyne HDL-64E)
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("range", "100.0")
        lidar_bp.set_attribute("points_per_second", "1300000")
        lidar_bp.set_attribute("rotation_frequency", "20")
        lidar_bp.set_attribute("upper_fov", "2.0")
        lidar_bp.set_attribute("lower_fov", "-24.8")
        # Weather effects on LiDAR
        lidar_bp.set_attribute("atmosphere_attenuation_rate",
                               str(0.004 + weather_params["fog_density"] * 0.001))
        lidar_bp.set_attribute("dropoff_general_rate",
                               str(0.45 + weather_params["precipitation"] * 0.003))

        lidar_transform = carla.Transform(
            carla.Location(x=0.0, y=0.0, z=1.8)
        )
        lidar_sensor = world.spawn_actor(
            lidar_bp, lidar_transform, attach_to=ego_vehicle
        )
        actors_to_destroy.append(lidar_sensor)

        # Spawn NPC vehicles
        vehicle_bps = blueprint_library.filter("vehicle.*")
        available_spawns = [sp for sp in spawn_points
                           if sp.location.distance(ego_transform.location) > 5.0]
        rng.shuffle(available_spawns)

        npc_vehicles = []
        for i in range(min(n_vehicles, len(available_spawns))):
            bp = rng.choice(vehicle_bps)
            actor = world.try_spawn_actor(bp, available_spawns[i])
            if actor is not None:
                actors_to_destroy.append(actor)
                npc_vehicles.append(actor)

        # Spawn NPC pedestrians
        ped_bps = blueprint_library.filter("walker.pedestrian.*")
        for i in range(n_pedestrians):
            bp = rng.choice(ped_bps)
            spawn_loc = carla.Transform()
            spawn_loc.location = world.get_random_location_from_navigation()
            if spawn_loc.location is not None:
                actor = world.try_spawn_actor(bp, spawn_loc)
                if actor is not None:
                    actors_to_destroy.append(actor)

        # Capture LiDAR data
        point_cloud_data = [None]

        def lidar_callback(data):
            point_cloud_data[0] = data

        lidar_sensor.listen(lidar_callback)

        # Tick the simulation to generate data
        world.tick()
        time.sleep(0.1)
        world.tick()
        time.sleep(0.5)

        lidar_sensor.stop()

        # Process point cloud
        if point_cloud_data[0] is not None:
            raw_data = np.frombuffer(
                point_cloud_data[0].raw_data, dtype=np.float32
            ).reshape(-1, 4)
            # CARLA format: x, y, z, intensity
            # Convert: CARLA uses left-handed, need right-handed for KITTI
            points = raw_data.copy()
            points[:, 1] = -points[:, 1]  # flip Y axis
        else:
            points = np.zeros((0, 4), dtype=np.float32)

        # Apply weather augmentation on point cloud (since CARLA LiDAR
        # doesn't natively model rain/fog effects on returns)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from sotif_uncertainty.weather_augmentation import augment_weather
        points = augment_weather(points, weather_params, seed=seed)

        # Extract ground truth bounding boxes
        gt_boxes = []
        gt_names = []
        ego_loc = ego_vehicle.get_transform()

        for npc in npc_vehicles:
            bb = npc.bounding_box
            transform = npc.get_transform()

            # Convert to ego LiDAR frame
            dx = transform.location.x - ego_loc.location.x
            dy = -(transform.location.y - ego_loc.location.y)
            dz = transform.location.z - ego_loc.location.z

            extent = bb.extent
            box_l = extent.x * 2
            box_w = extent.y * 2
            box_h = extent.z * 2
            yaw = -np.radians(transform.rotation.yaw - ego_loc.rotation.yaw)

            gt_boxes.append([dx, dy, dz, box_l, box_w, box_h, yaw])
            gt_names.append("Car")

        category = TC_CATEGORY_MAP.get(config_name, "other")

        return {
            "points": points.astype(np.float32),
            "gt_boxes": np.array(gt_boxes) if gt_boxes else np.zeros((0, 7)),
            "gt_names": np.array(gt_names) if gt_names else np.array([], dtype=str),
            "config": config_name,
            "category": category,
        }

    finally:
        # Clean up all spawned actors
        for actor in reversed(actors_to_destroy):
            try:
                actor.destroy()
            except Exception:
                pass


def make_kitti_calib_file(output_path: str):
    """
    Generate a KITTI-format calibration file for CARLA data.

    Uses default CARLA sensor mounting (LiDAR on roof, camera at front).
    These values match the standard CARLA-KITTI bridge configuration.
    """
    # Default CARLA camera intrinsics (800x600, FOV=90)
    fx = 400.0
    fy = 400.0
    cx = 400.0
    cy = 300.0

    P2 = np.array([
        [fx, 0, cx, 0],
        [0, fy, cy, 0],
        [0, 0, 1, 0],
    ])

    R0_rect = np.eye(3)

    # CARLA LiDAR to camera transform (default mounting)
    # LiDAR on roof, camera at front bumper
    Tr_velo_to_cam = np.array([
        [0, -1, 0, 0],
        [0, 0, -1, -0.08],
        [1, 0, 0, -0.27],
    ], dtype=np.float64)

    Tr_imu_to_velo = np.eye(3, 4)

    with open(output_path, "w") as f:
        f.write(f"P0: {' '.join(f'{v:.6e}' for v in P2.flatten())}\n")
        f.write(f"P1: {' '.join(f'{v:.6e}' for v in P2.flatten())}\n")
        f.write(f"P2: {' '.join(f'{v:.6e}' for v in P2.flatten())}\n")
        f.write(f"P3: {' '.join(f'{v:.6e}' for v in P2.flatten())}\n")
        f.write(f"R0_rect: {' '.join(f'{v:.6e}' for v in R0_rect.flatten())}\n")
        f.write(f"Tr_velo_to_cam: {' '.join(f'{v:.6e}' for v in Tr_velo_to_cam.flatten())}\n")
        f.write(f"Tr_imu_to_velo: {' '.join(f'{v:.6e}' for v in Tr_imu_to_velo.flatten())}\n")


def generate_synthetic_lidar_frame(
    config_name: str,
    n_vehicles: int = 8,
    n_pedestrians: int = 3,
    seed: int = 42,
) -> dict:
    """
    Generate a synthetic LiDAR frame with ground truth labels.

    This function creates realistic point cloud data without requiring
    a running CARLA instance, useful for testing the pipeline.

    The generated data includes:
    - Simulated point cloud with ground plane and objects
    - Ground truth 3D bounding boxes in LiDAR frame
    - Weather-dependent noise (rain scatter, fog attenuation)

    Parameters
    ----------
    config_name : str
        Name from WEATHER_CONFIGS.
    n_vehicles : int
        Number of vehicles to place.
    n_pedestrians : int
        Number of pedestrians to place.
    seed : int
        Random seed.

    Returns
    -------
    dict with keys:
        'points' : (N, 4) point cloud [x, y, z, intensity]
        'gt_boxes' : (M, 7) ground truth [x, y, z, dx, dy, dz, heading]
        'gt_names' : (M,) class names
        'config' : str config name
        'category' : str TC category
    """
    rng = np.random.RandomState(seed)
    config = WEATHER_CONFIGS.get(config_name, WEATHER_CONFIGS["ClearNoon"])

    weather = config["carla_weather"]
    fog = weather["fog_density"]
    rain = weather["precipitation"]

    all_points = []
    gt_boxes = []
    gt_names = []

    # 1. Generate ground plane points
    n_ground = rng.randint(15000, 25000)
    ground_x = rng.uniform(0, 70, n_ground)
    ground_y = rng.uniform(-40, 40, n_ground)
    ground_z = -1.73 + rng.normal(0, 0.02, n_ground)
    ground_i = rng.uniform(0.1, 0.4, n_ground)
    all_points.append(np.stack([ground_x, ground_y, ground_z, ground_i], axis=1))

    # 2. Generate vehicle point clouds
    for v in range(n_vehicles):
        vx = rng.uniform(5, 60)
        vy = rng.uniform(-15, 15)
        vz = -1.73 + 0.8  # center height
        vl = rng.uniform(3.5, 5.0)
        vw = rng.uniform(1.6, 2.0)
        vh = rng.uniform(1.4, 1.8)
        vyaw = rng.uniform(-np.pi, np.pi)

        gt_boxes.append([vx, vy, vz, vl, vw, vh, vyaw])
        gt_names.append("Car")

        # Generate points on vehicle surfaces
        n_pts = rng.randint(50, 300)
        # Front/back faces
        face_choice = rng.choice(4, n_pts)
        pts = np.zeros((n_pts, 4))
        for p in range(n_pts):
            if face_choice[p] == 0:  # front
                px = vl / 2 + rng.normal(0, 0.02)
                py = rng.uniform(-vw / 2, vw / 2)
                pz_local = rng.uniform(-vh / 2, vh / 2)
            elif face_choice[p] == 1:  # back
                px = -vl / 2 + rng.normal(0, 0.02)
                py = rng.uniform(-vw / 2, vw / 2)
                pz_local = rng.uniform(-vh / 2, vh / 2)
            elif face_choice[p] == 2:  # left
                px = rng.uniform(-vl / 2, vl / 2)
                py = vw / 2 + rng.normal(0, 0.02)
                pz_local = rng.uniform(-vh / 2, vh / 2)
            else:  # right
                px = rng.uniform(-vl / 2, vl / 2)
                py = -vw / 2 + rng.normal(0, 0.02)
                pz_local = rng.uniform(-vh / 2, vh / 2)

            # Rotate by yaw and translate
            cos_y, sin_y = np.cos(vyaw), np.sin(vyaw)
            rx = cos_y * px - sin_y * py + vx
            ry = sin_y * px + cos_y * py + vy
            rz = pz_local + vz

            pts[p] = [rx, ry, rz, rng.uniform(0.3, 0.8)]

        all_points.append(pts)

    # 3. Generate pedestrian point clouds
    for p_idx in range(n_pedestrians):
        px = rng.uniform(3, 40)
        py = rng.uniform(-10, 10)
        pz = -1.73 + 0.85
        pl, pw, ph = 0.6, 0.6, 1.7
        pyaw = rng.uniform(-np.pi, np.pi)

        gt_boxes.append([px, py, pz, pl, pw, ph, pyaw])
        gt_names.append("Pedestrian")

        # Sparse points for pedestrians
        n_pts = rng.randint(10, 60)
        pts = np.zeros((n_pts, 4))
        for i in range(n_pts):
            pts[i] = [
                px + rng.normal(0, 0.15),
                py + rng.normal(0, 0.15),
                pz + rng.uniform(-ph / 2, ph / 2),
                rng.uniform(0.15, 0.5),
            ]
        all_points.append(pts)

    # 4. Add weather effects
    points = np.vstack(all_points)

    # Rain: add scatter noise points
    if rain > 30:
        n_rain = int(rain * 20)
        rain_pts = np.zeros((n_rain, 4))
        rain_pts[:, 0] = rng.uniform(0, 50, n_rain)
        rain_pts[:, 1] = rng.uniform(-30, 30, n_rain)
        rain_pts[:, 2] = rng.uniform(-1.5, 2.0, n_rain)
        rain_pts[:, 3] = rng.uniform(0.01, 0.15, n_rain)
        points = np.vstack([points, rain_pts])

    # Fog: attenuate distant points
    if fog > 20:
        dist = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        fog_range = max(10, 70 - fog * 0.6)
        attenuation = np.exp(-dist / fog_range)
        points[:, 3] *= attenuation
        keep = attenuation > 0.05
        points = points[keep]

    category = TC_CATEGORY_MAP.get(config_name, "other")

    return {
        "points": points.astype(np.float32),
        "gt_boxes": np.array(gt_boxes) if gt_boxes else np.zeros((0, 7)),
        "gt_names": np.array(gt_names) if gt_names else np.array([], dtype=str),
        "config": config_name,
        "category": category,
    }


def save_frame_kitti_format(
    frame_data: dict,
    frame_id: str,
    output_dir: str,
    calib_path: str,
):
    """
    Save a single frame in KITTI format.

    Parameters
    ----------
    frame_data : dict
        Output from generate_synthetic_lidar_frame() or CARLA capture.
    frame_id : str
        6-digit frame ID (e.g., '000000').
    output_dir : str
        Root output directory.
    calib_path : str
        Path to calibration file template.
    """
    # Save point cloud
    vel_dir = os.path.join(output_dir, "training", "velodyne")
    os.makedirs(vel_dir, exist_ok=True)
    points = frame_data["points"].astype(np.float32)
    points.tofile(os.path.join(vel_dir, f"{frame_id}.bin"))

    # Save calibration (copy template)
    calib_dir = os.path.join(output_dir, "training", "calib")
    os.makedirs(calib_dir, exist_ok=True)
    import shutil
    calib_out = os.path.join(calib_dir, f"{frame_id}.txt")
    if not os.path.exists(calib_out):
        shutil.copy2(calib_path, calib_out)

    # Save labels (in KITTI camera format)
    label_dir = os.path.join(output_dir, "training", "label_2")
    os.makedirs(label_dir, exist_ok=True)

    # Load calibration for coordinate transform
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from sotif_uncertainty.kitti_utils import KITTICalibration

    calib = KITTICalibration(calib_path)

    with open(os.path.join(label_dir, f"{frame_id}.txt"), "w") as f:
        gt_boxes = frame_data["gt_boxes"]
        gt_names = frame_data["gt_names"]

        for i in range(len(gt_boxes)):
            box_lidar = gt_boxes[i]
            box_cam = calib.boxes_lidar_to_cam(box_lidar.reshape(1, 7))[0]

            # KITTI label format:
            # type trunc occ alpha bbox(4) dim(3) loc(3) ry [score]
            name = gt_names[i]
            h, w, l = box_cam[0], box_cam[1], box_cam[2]
            x, y, z = box_cam[3], box_cam[4], box_cam[5]
            ry = box_cam[6]

            # Dummy 2D bbox and alpha (not used for 3D evaluation)
            f.write(
                f"{name} 0.00 0 0.00 "
                f"0.00 0.00 100.00 100.00 "
                f"{h:.2f} {w:.2f} {l:.2f} "
                f"{x:.2f} {y:.2f} {z:.2f} {ry:.2f}\n"
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate LiDAR point cloud data from CARLA for SOTIF evaluation."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/carla",
        help="Output directory for generated data.",
    )
    parser.add_argument(
        "--conditions", nargs="+", default=None,
        help="TC categories to generate: heavy_rain, night, fog, other. "
             "Default: all 22 configurations.",
    )
    parser.add_argument(
        "--frames_per_config", type=int, default=5,
        help="Number of frames per weather configuration.",
    )
    parser.add_argument(
        "--mode", type=str, default="synthetic",
        choices=["synthetic", "carla"],
        help="'synthetic': generate data offline (no CARLA needed). "
             "'carla': connect to running CARLA instance.",
    )
    parser.add_argument(
        "--carla_host", type=str, default="localhost",
        help="CARLA server hostname.",
    )
    parser.add_argument(
        "--carla_port", type=int, default=2000,
        help="CARLA server port.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("CARLA Data Generation for SOTIF Evaluation")
    print("=" * 60)
    print(f"  Output:  {args.output_dir}")
    print(f"  Mode:    {args.mode}")
    print(f"  Frames:  {args.frames_per_config} per config")

    # Filter configurations
    if args.conditions:
        configs_to_use = {
            name: cfg for name, cfg in WEATHER_CONFIGS.items()
            if cfg["category"] in args.conditions
            or TC_CATEGORY_MAP.get(name, "other") in args.conditions
        }
    else:
        configs_to_use = WEATHER_CONFIGS

    print(f"  Configs: {len(configs_to_use)}")

    if args.mode == "carla":
        print("\n  NOTE: CARLA mode requires a running CARLA instance.")
        print("  For offline data generation, use --mode synthetic")
        try:
            import carla
            client = carla.Client(args.carla_host, args.carla_port)
            client.set_timeout(10.0)
            world = client.get_world()
            print(f"  Connected to CARLA: {world.get_map().name}")
        except ImportError:
            print("\n  ERROR: carla package not installed.")
            print("  Install: pip install carla")
            print("  Or use: --mode synthetic")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR: Could not connect to CARLA: {e}")
            print("  Start CARLA: ./CarlaUE4.sh -RenderOffScreen")
            print("  Or use: --mode synthetic")
            sys.exit(1)

    # Generate calibration template
    calib_template = os.path.join(args.output_dir, "_calib_template.txt")
    make_kitti_calib_file(calib_template)

    # Generate data
    frame_counter = 0
    metadata = {}
    all_frame_ids = []

    for config_name, config in sorted(configs_to_use.items()):
        category = config["category"]
        tc = TC_CATEGORY_MAP.get(config_name, "other")
        print(f"\n  [{config_name}] {config['description']} (TC: {tc})")

        for f_idx in range(args.frames_per_config):
            frame_id = f"{frame_counter:06d}"
            seed = args.seed + frame_counter * 7 + hash(config_name) % 1000

            if args.mode == "synthetic":
                frame_data = generate_synthetic_lidar_frame(
                    config_name,
                    n_vehicles=np.random.RandomState(seed).randint(3, 12),
                    n_pedestrians=np.random.RandomState(seed).randint(0, 5),
                    seed=seed,
                )
            else:
                # CARLA API data collection
                frame_data = collect_carla_frame(
                    world, config_name, config,
                    n_vehicles=np.random.RandomState(seed).randint(3, 12),
                    n_pedestrians=np.random.RandomState(seed).randint(0, 5),
                    seed=seed,
                )

            save_frame_kitti_format(
                frame_data, frame_id, args.output_dir, calib_template
            )

            n_pts = len(frame_data["points"])
            n_obj = len(frame_data["gt_boxes"])
            print(f"    Frame {frame_id}: {n_pts} points, {n_obj} objects")

            metadata[frame_id] = {
                "config": config_name,
                "category": category,
                "tc_category": tc,
                "n_points": n_pts,
                "n_objects": n_obj,
            }
            all_frame_ids.append(frame_id)
            frame_counter += 1

    # Create ImageSets
    imagesets_dir = os.path.join(args.output_dir, "ImageSets")
    os.makedirs(imagesets_dir, exist_ok=True)

    # 80/20 train/val split
    n_total = len(all_frame_ids)
    n_train = int(n_total * 0.8)
    np.random.RandomState(args.seed).shuffle(all_frame_ids)

    for name, ids in [("train", all_frame_ids[:n_train]),
                      ("val", all_frame_ids[n_train:])]:
        with open(os.path.join(imagesets_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(sorted(ids)) + "\n")
        print(f"\n  {name}.txt: {len(ids)} frames")

    # Save metadata
    meta_path = os.path.join(args.output_dir, "conditions.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to: {meta_path}")

    # Clean up template
    if os.path.exists(calib_template):
        os.remove(calib_template)

    print(f"\n{'=' * 60}")
    print(f"Generated {frame_counter} frames across {len(configs_to_use)} configurations.")
    print(f"Data saved to: {args.output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Create OpenPCDet info files:")
    print(f"     python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos \\")
    print(f"       tools/cfgs/dataset_configs/kitti_dataset.yaml")
    print(f"  2. Train ensemble: bash scripts/train_ensemble.sh")
    print(f"  3. Or use demo mode: python scripts/evaluate.py")
    print("=" * 60)


if __name__ == "__main__":
    main()

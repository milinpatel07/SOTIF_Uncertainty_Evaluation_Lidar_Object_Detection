"""
Generate LiDAR object detection data from CARLA simulator.

Creates point cloud data and ground truth labels in KITTI format
for use with the SOTIF uncertainty evaluation pipeline.

Prerequisites:
    - CARLA simulator 0.9.13+ running (https://carla.org/)
    - CARLA Python API installed: pip install carla

Usage:
    # Basic: generate data across all weather conditions
    python scripts/generate_carla_data.py --output_dir data/carla_kitti

    # Custom: specific weather, more frames
    python scripts/generate_carla_data.py \\
        --output_dir data/carla_kitti \\
        --host localhost --port 2000 \\
        --frames_per_config 10 \\
        --configs ClearNoon HeavyRainNoon FoggyDay ClearNight

    # Generate all 22 configurations from the paper (Table 2)
    python scripts/generate_carla_data.py \\
        --output_dir data/carla_kitti \\
        --frames_per_config 5 \\
        --all_configs

Output structure (KITTI format):
    data/carla_kitti/
    ├── training/
    │   ├── velodyne/       # .bin point cloud files
    │   ├── label_2/        # .txt ground truth labels
    │   ├── calib/          # .txt calibration files
    │   └── image_2/        # .png camera images (optional)
    ├── ImageSets/
    │   ├── train.txt
    │   └── val.txt
    └── conditions.json     # Environmental condition per frame

The 22 environmental configurations from Table 2:
    Clear/Overcast: ClearNoon, CloudyNoon, LowSunNoon, HighWindNoon
    Precipitation:  WetNoon, WetCloudyNoon, MidRainyNoon, HeavyRainNoon
    Reduced Visibility: FoggyDay, DustStorm, SnowyConditions
    Night Clear:    NightDrivingClear, ClearNight, CloudyNight
    Night Adverse:  WetNight, MidRainNight, HardRainNight, FoggyNight
    Compound:       DuskTransition, EmergencyBraking, OvertakeMultiVehicle, ExtremeWeather
"""

import argparse
import json
import os
import sys
import time
import struct
import numpy as np
from pathlib import Path


# ======================================================================
# Weather preset definitions matching Table 2 of the paper
# ======================================================================
WEATHER_PRESETS = {
    # Clear/Overcast
    "ClearNoon": {
        "cloudiness": 10, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": 70, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "CloudyNoon": {
        "cloudiness": 80, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 30, "sun_altitude_angle": 60, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "LowSunNoon": {
        "cloudiness": 20, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": 15, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "HighWindNoon": {
        "cloudiness": 40, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 100, "sun_altitude_angle": 60, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    # Precipitation
    "WetNoon": {
        "cloudiness": 60, "precipitation": 10, "precipitation_deposits": 40,
        "wind_intensity": 20, "sun_altitude_angle": 50, "fog_density": 0,
        "fog_distance": 0, "wetness": 50,
    },
    "WetCloudyNoon": {
        "cloudiness": 80, "precipitation": 30, "precipitation_deposits": 50,
        "wind_intensity": 30, "sun_altitude_angle": 40, "fog_density": 5,
        "fog_distance": 0, "wetness": 60,
    },
    "MidRainyNoon": {
        "cloudiness": 90, "precipitation": 60, "precipitation_deposits": 60,
        "wind_intensity": 50, "sun_altitude_angle": 30, "fog_density": 10,
        "fog_distance": 0, "wetness": 80,
    },
    "HeavyRainNoon": {
        "cloudiness": 100, "precipitation": 100, "precipitation_deposits": 90,
        "wind_intensity": 80, "sun_altitude_angle": 20, "fog_density": 30,
        "fog_distance": 0, "wetness": 100,
    },
    # Reduced visibility
    "FoggyDay": {
        "cloudiness": 70, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": 40, "fog_density": 70,
        "fog_distance": 10, "wetness": 0,
    },
    "DustStorm": {
        "cloudiness": 90, "precipitation": 0, "precipitation_deposits": 30,
        "wind_intensity": 100, "sun_altitude_angle": 50, "fog_density": 80,
        "fog_distance": 5, "wetness": 0,
    },
    "SnowyConditions": {
        "cloudiness": 100, "precipitation": 80, "precipitation_deposits": 80,
        "wind_intensity": 60, "sun_altitude_angle": 25, "fog_density": 40,
        "fog_distance": 20, "wetness": 30,
    },
    # Night clear
    "NightDrivingClear": {
        "cloudiness": 10, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": -30, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "ClearNight": {
        "cloudiness": 5, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 5, "sun_altitude_angle": -50, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "CloudyNight": {
        "cloudiness": 80, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 20, "sun_altitude_angle": -40, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    # Night adverse
    "WetNight": {
        "cloudiness": 70, "precipitation": 30, "precipitation_deposits": 50,
        "wind_intensity": 30, "sun_altitude_angle": -30, "fog_density": 10,
        "fog_distance": 0, "wetness": 60,
    },
    "MidRainNight": {
        "cloudiness": 90, "precipitation": 60, "precipitation_deposits": 70,
        "wind_intensity": 50, "sun_altitude_angle": -40, "fog_density": 20,
        "fog_distance": 0, "wetness": 80,
    },
    "HardRainNight": {
        "cloudiness": 100, "precipitation": 100, "precipitation_deposits": 100,
        "wind_intensity": 90, "sun_altitude_angle": -50, "fog_density": 40,
        "fog_distance": 0, "wetness": 100,
    },
    "FoggyNight": {
        "cloudiness": 80, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": -35, "fog_density": 80,
        "fog_distance": 5, "wetness": 0,
    },
    # Compound/dynamic
    "DuskTransition": {
        "cloudiness": 40, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 20, "sun_altitude_angle": 2, "fog_density": 10,
        "fog_distance": 0, "wetness": 0,
    },
    "EmergencyBraking": {
        "cloudiness": 30, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 10, "sun_altitude_angle": 50, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "OvertakeMultiVehicle": {
        "cloudiness": 20, "precipitation": 0, "precipitation_deposits": 0,
        "wind_intensity": 15, "sun_altitude_angle": 60, "fog_density": 0,
        "fog_distance": 0, "wetness": 0,
    },
    "ExtremeWeather": {
        "cloudiness": 100, "precipitation": 100, "precipitation_deposits": 100,
        "wind_intensity": 100, "sun_altitude_angle": -10, "fog_density": 60,
        "fog_distance": 5, "wetness": 100,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate CARLA LiDAR data in KITTI format."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/carla_kitti",
        help="Output directory for KITTI-format data.",
    )
    parser.add_argument(
        "--host", type=str, default="localhost",
        help="CARLA server hostname.",
    )
    parser.add_argument(
        "--port", type=int, default=2000,
        help="CARLA server port.",
    )
    parser.add_argument(
        "--configs", nargs="+", default=None,
        help="Specific weather configs to use. Defaults to a representative set.",
    )
    parser.add_argument(
        "--all_configs", action="store_true",
        help="Use all 22 weather configurations from Table 2.",
    )
    parser.add_argument(
        "--frames_per_config", type=int, default=5,
        help="Number of frames to capture per weather config.",
    )
    parser.add_argument(
        "--n_vehicles", type=int, default=30,
        help="Number of NPC vehicles to spawn.",
    )
    parser.add_argument(
        "--lidar_range", type=float, default=70.0,
        help="LiDAR sensor range in meters.",
    )
    parser.add_argument(
        "--lidar_channels", type=int, default=64,
        help="Number of LiDAR channels (vertical resolution).",
    )
    parser.add_argument(
        "--lidar_points_per_second", type=int, default=1200000,
        help="LiDAR points per second.",
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2,
        help="Fraction of frames for validation split.",
    )
    return parser.parse_args()


def set_weather(world, config_name):
    """Apply weather preset to CARLA world."""
    import carla

    params = WEATHER_PRESETS[config_name]
    weather = carla.WeatherParameters(
        cloudiness=params["cloudiness"],
        precipitation=params["precipitation"],
        precipitation_deposits=params["precipitation_deposits"],
        wind_intensity=params["wind_intensity"],
        sun_altitude_angle=params["sun_altitude_angle"],
        fog_density=params["fog_density"],
        fog_distance=params["fog_distance"],
        wetness=params["wetness"],
    )
    world.set_weather(weather)


def spawn_ego_with_lidar(world, blueprint_library, spawn_point, args):
    """Spawn ego vehicle with a LiDAR sensor attached."""
    import carla

    # Ego vehicle
    vehicle_bp = blueprint_library.filter("vehicle.tesla.model3")[0]
    ego = world.spawn_actor(vehicle_bp, spawn_point)

    # LiDAR sensor
    lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
    lidar_bp.set_attribute("range", str(args.lidar_range))
    lidar_bp.set_attribute("channels", str(args.lidar_channels))
    lidar_bp.set_attribute("points_per_second", str(args.lidar_points_per_second))
    lidar_bp.set_attribute("rotation_frequency", "20")
    lidar_bp.set_attribute("upper_fov", "2.0")
    lidar_bp.set_attribute("lower_fov", "-24.8")

    lidar_transform = carla.Transform(carla.Location(x=0.0, z=1.8))
    lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego)

    return ego, lidar


def lidar_to_kitti_bin(point_cloud_data):
    """Convert CARLA LiDAR data to KITTI .bin format (x, y, z, intensity)."""
    points = np.frombuffer(point_cloud_data.raw_data, dtype=np.float32)
    points = points.reshape(-1, 4)
    # CARLA uses (x, y, z, intensity) -- same as KITTI .bin
    # But CARLA's coordinate system differs: negate y for KITTI
    points[:, 1] = -points[:, 1]
    return points


def get_vehicle_bboxes_kitti(world, ego_vehicle, lidar_transform):
    """Get 3D bounding boxes for all vehicles in KITTI format."""
    import carla

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location

    bboxes = []
    actors = world.get_actors().filter("vehicle.*")

    for actor in actors:
        if actor.id == ego_vehicle.id:
            continue

        # Get bounding box in world frame
        bbox = actor.bounding_box
        actor_transform = actor.get_transform()

        # Distance from ego
        dx = actor_transform.location.x - ego_location.x
        dy = actor_transform.location.y - ego_location.y
        dz = actor_transform.location.z - ego_location.z
        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        if dist > 70.0:  # Skip far objects
            continue

        # Convert to LiDAR frame
        # KITTI format: x(forward), y(left), z(up) in camera frame
        # For LiDAR labels: x, y, z, dx, dy, dz, heading
        extent = bbox.extent
        h = extent.z * 2  # height
        w = extent.y * 2  # width
        l = extent.x * 2  # length

        # Transform to ego vehicle frame
        actor_loc = actor_transform.location
        ego_inv = ego_transform.get_inverse_matrix()

        # Simple relative position (LiDAR frame approximation)
        x_rel = dx * np.cos(np.radians(-ego_transform.rotation.yaw)) - \
                dy * np.sin(np.radians(-ego_transform.rotation.yaw))
        y_rel = dx * np.sin(np.radians(-ego_transform.rotation.yaw)) + \
                dy * np.cos(np.radians(-ego_transform.rotation.yaw))
        z_rel = dz

        # Heading
        heading = np.radians(actor_transform.rotation.yaw - ego_transform.rotation.yaw)

        # KITTI label format
        # type truncated occluded alpha bbox(4) dimensions(3) location(3) rotation_y
        label_line = (
            f"Car 0.00 0 0.00 "
            f"0 0 100 100 "
            f"{h:.2f} {w:.2f} {l:.2f} "
            f"{x_rel:.2f} {z_rel:.2f} {-y_rel:.2f} "
            f"{heading:.2f}"
        )
        bboxes.append(label_line)

    return bboxes


def write_calib_file(calib_path):
    """Write a default calibration file (identity-like for LiDAR-only eval)."""
    # For LiDAR-only detection, calibration is identity
    # These are placeholder matrices matching KITTI format
    P = "0 0 0 0 0 0 0 0 0 0 0 0"
    R0 = "1 0 0 0 1 0 0 0 1"
    Tr = "1 0 0 0 0 1 0 0 0 0 1 0"

    with open(calib_path, "w") as f:
        f.write(f"P0: {P}\n")
        f.write(f"P1: {P}\n")
        f.write(f"P2: {P}\n")
        f.write(f"P3: {P}\n")
        f.write(f"R0_rect: {R0}\n")
        f.write(f"Tr_velo_to_cam: {Tr}\n")
        f.write(f"Tr_imu_to_velo: {Tr}\n")


def main():
    args = parse_args()

    # Determine configs
    if args.all_configs:
        configs = list(WEATHER_PRESETS.keys())
    elif args.configs:
        configs = args.configs
        for c in configs:
            if c not in WEATHER_PRESETS:
                print(f"ERROR: Unknown config '{c}'. Available:")
                for name in WEATHER_PRESETS:
                    print(f"  {name}")
                sys.exit(1)
    else:
        # Representative set
        configs = [
            "ClearNoon", "HeavyRainNoon", "FoggyDay",
            "ClearNight", "HardRainNight", "ExtremeWeather",
        ]

    total_frames = len(configs) * args.frames_per_config

    print("=" * 60)
    print("CARLA Data Generation for SOTIF Evaluation")
    print("=" * 60)
    print(f"  Server: {args.host}:{args.port}")
    print(f"  Configs: {len(configs)}")
    print(f"  Frames/config: {args.frames_per_config}")
    print(f"  Total frames: {total_frames}")
    print(f"  Output: {args.output_dir}")
    print(f"  LiDAR: {args.lidar_channels}ch, {args.lidar_range}m range")
    print()

    # Create output directories
    for subdir in ["training/velodyne", "training/label_2", "training/calib",
                   "training/image_2", "ImageSets"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Try to connect to CARLA
    try:
        import carla
    except ImportError:
        print("=" * 60)
        print("CARLA Python API not installed.")
        print("=" * 60)
        print()
        print("To install CARLA:")
        print("  1. Download CARLA 0.9.13+: https://carla.org/")
        print("  2. Install Python API: pip install carla")
        print("  3. Start CARLA server: ./CarlaUE4.sh")
        print("  4. Run this script")
        print()
        print("Alternatively, use the synthetic demo data:")
        print("  python scripts/evaluate.py")
        print()
        print("Or download pre-recorded data:")
        print("  See README.md for available datasets")
        sys.exit(1)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        print(f"  Connected to CARLA {client.get_server_version()}")
    except Exception as e:
        print(f"ERROR: Cannot connect to CARLA at {args.host}:{args.port}")
        print(f"  {e}")
        print()
        print("Make sure CARLA is running:")
        print("  ./CarlaUE4.sh  (or CarlaUE4.exe on Windows)")
        sys.exit(1)

    # Set synchronous mode for deterministic data collection
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    frame_counter = 0
    condition_map = {}  # frame_id -> config name
    actors_to_destroy = []

    try:
        spawn_points = world.get_map().get_spawn_points()

        for config_name in configs:
            print(f"\n  [{config_name}] Setting weather...")
            set_weather(world, config_name)

            # Spawn ego vehicle
            ego_spawn = np.random.choice(len(spawn_points))
            ego, lidar = spawn_ego_with_lidar(
                world, blueprint_library, spawn_points[ego_spawn], args
            )
            actors_to_destroy.extend([ego, lidar])

            # Spawn NPC vehicles
            npc_actors = []
            available_points = [p for i, p in enumerate(spawn_points) if i != ego_spawn]
            np.random.shuffle(available_points)
            vehicle_bps = blueprint_library.filter("vehicle.*")

            for i in range(min(args.n_vehicles, len(available_points))):
                bp = np.random.choice(vehicle_bps)
                try:
                    npc = world.try_spawn_actor(bp, available_points[i])
                    if npc:
                        npc.set_autopilot(True)
                        npc_actors.append(npc)
                except Exception:
                    pass

            actors_to_destroy.extend(npc_actors)

            # Enable autopilot for ego
            ego.set_autopilot(True)

            # Let simulation warm up
            for _ in range(20):
                world.tick()

            # Collect frames
            lidar_data_buffer = [None]

            def lidar_callback(data):
                lidar_data_buffer[0] = data

            lidar.listen(lidar_callback)

            for frame_idx in range(args.frames_per_config):
                world.tick()
                time.sleep(0.1)

                if lidar_data_buffer[0] is None:
                    world.tick()
                    time.sleep(0.1)

                if lidar_data_buffer[0] is not None:
                    frame_id = f"{frame_counter:06d}"

                    # Save point cloud
                    points = lidar_to_kitti_bin(lidar_data_buffer[0])
                    bin_path = os.path.join(args.output_dir, "training", "velodyne", f"{frame_id}.bin")
                    points.astype(np.float32).tofile(bin_path)

                    # Save labels
                    bboxes = get_vehicle_bboxes_kitti(world, ego, lidar.get_transform())
                    label_path = os.path.join(args.output_dir, "training", "label_2", f"{frame_id}.txt")
                    with open(label_path, "w") as f:
                        f.write("\n".join(bboxes) + "\n" if bboxes else "")

                    # Save calibration
                    calib_path = os.path.join(args.output_dir, "training", "calib", f"{frame_id}.txt")
                    write_calib_file(calib_path)

                    condition_map[frame_id] = config_name
                    frame_counter += 1

                    n_pts = len(points)
                    n_objs = len(bboxes)
                    print(f"    Frame {frame_id}: {n_pts} points, {n_objs} objects")

            lidar.stop()

            # Clean up NPC vehicles
            for npc in npc_actors:
                npc.destroy()
            actors_to_destroy = [a for a in actors_to_destroy if a not in npc_actors]

            # Destroy ego + lidar for this config
            lidar.destroy()
            ego.destroy()
            actors_to_destroy = [a for a in actors_to_destroy
                                 if a.id != ego.id and a.id != lidar.id]

    finally:
        # Restore original settings
        world.apply_settings(original_settings)
        for actor in actors_to_destroy:
            try:
                actor.destroy()
            except Exception:
                pass

    # Create ImageSets
    all_frames = sorted(condition_map.keys())
    n_val = int(len(all_frames) * args.val_ratio)
    np.random.shuffle(all_frames)
    val_frames = sorted(all_frames[:n_val])
    train_frames = sorted(all_frames[n_val:])

    for name, frame_list in [("train", train_frames), ("val", val_frames)]:
        path = os.path.join(args.output_dir, "ImageSets", f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(frame_list) + "\n")

    # Save condition metadata
    cond_path = os.path.join(args.output_dir, "conditions.json")
    with open(cond_path, "w") as f:
        json.dump(condition_map, f, indent=2)

    print()
    print("=" * 60)
    print(f"Generated {frame_counter} frames across {len(configs)} configs")
    print(f"  Train: {len(train_frames)} frames")
    print(f"  Val: {len(val_frames)} frames")
    print(f"  Output: {args.output_dir}")
    print(f"  Conditions: {cond_path}")
    print("=" * 60)
    print()
    print("Next steps:")
    print(f"  1. Create OpenPCDet info files for this data")
    print(f"  2. Train ensemble: bash scripts/train_ensemble.sh")
    print(f"  3. Run inference: python scripts/run_inference.py --ckpt_dirs ...")


if __name__ == "__main__":
    main()

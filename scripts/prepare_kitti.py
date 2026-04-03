"""
Download and prepare the KITTI 3D Object Detection dataset for use
with the SOTIF uncertainty evaluation pipeline.

KITTI provides LiDAR point clouds, camera images, calibration matrices,
and 3D bounding box annotations for autonomous driving.

Website: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

Usage:
    # Download and prepare KITTI data
    python scripts/prepare_kitti.py --data_root data/kitti

    # Only download specific components
    python scripts/prepare_kitti.py --data_root data/kitti --components velodyne labels calib

    # Skip download (data already exists), only create OpenPCDet info files
    python scripts/prepare_kitti.py --data_root data/kitti --skip_download
"""

import argparse
import os
import sys
import subprocess


# KITTI 3D Object Detection download URLs
KITTI_URLS = {
    "velodyne": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip",
        "size": "29 GB",
        "description": "Velodyne point clouds (training + testing)",
    },
    "labels": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
        "size": "5 MB",
        "description": "Training labels (3D bounding boxes)",
    },
    "calib": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
        "size": "16 MB",
        "description": "Calibration files (LiDAR-camera projection matrices)",
    },
    "images": {
        "url": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
        "size": "12 GB",
        "description": "Left color images (optional for LiDAR-only detection)",
    },
}

# Expected KITTI directory structure after extraction
EXPECTED_STRUCTURE = """
data/kitti/
├── training/
│   ├── velodyne/     # 7481 .bin point cloud files
│   ├── label_2/      # 7481 .txt label files
│   ├── calib/        # 7481 .txt calibration files
│   └── image_2/      # 7481 .png images (optional)
├── testing/
│   ├── velodyne/     # 7518 .bin point cloud files
│   ├── calib/        # 7518 .txt calibration files
│   └── image_2/      # 7518 .png images (optional)
└── ImageSets/
    ├── train.txt     # Training frame IDs
    ├── val.txt       # Validation frame IDs
    └── test.txt      # Test frame IDs
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and prepare KITTI 3D Object Detection dataset."
    )
    parser.add_argument(
        "--data_root", type=str, default="data/kitti",
        help="Root directory for KITTI data.",
    )
    parser.add_argument(
        "--components", nargs="+",
        default=["velodyne", "labels", "calib"],
        choices=list(KITTI_URLS.keys()),
        help="Which components to download.",
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip downloading, only verify and create splits.",
    )
    return parser.parse_args()


def download_component(name, url, data_root):
    """Download and extract a KITTI component."""
    zip_path = os.path.join(data_root, f"{name}.zip")

    if os.path.exists(zip_path):
        print(f"  {name}: zip already exists, skipping download")
    else:
        print(f"  {name}: downloading from {url}")
        print(f"  Size: {KITTI_URLS[name]['size']}")
        try:
            subprocess.run(
                ["wget", "-c", "-O", zip_path, url],
                check=True,
            )
        except FileNotFoundError:
            print("  ERROR: wget not found. Install wget or download manually:")
            print(f"    URL: {url}")
            print(f"    Save to: {zip_path}")
            return False

    # Extract
    print(f"  {name}: extracting...")
    subprocess.run(["unzip", "-o", "-q", zip_path, "-d", data_root], check=True)
    print(f"  {name}: done")
    return True


def create_imagesets(data_root):
    """Create train/val/test split files for OpenPCDet."""
    imagesets_dir = os.path.join(data_root, "ImageSets")
    os.makedirs(imagesets_dir, exist_ok=True)

    # Check how many training frames exist
    velodyne_dir = os.path.join(data_root, "training", "velodyne")
    if not os.path.exists(velodyne_dir):
        print("  WARNING: training/velodyne not found. Cannot create splits.")
        return

    frames = sorted([f[:-4] for f in os.listdir(velodyne_dir) if f.endswith(".bin")])
    n_total = len(frames)
    print(f"  Found {n_total} training frames")

    # Standard KITTI split: 3712 train, 3769 val
    # Using the standard Chen et al. split
    n_train = 3712
    train_frames = frames[:n_train]
    val_frames = frames[n_train:]

    for name, frame_list in [("train", train_frames), ("val", val_frames)]:
        path = os.path.join(imagesets_dir, f"{name}.txt")
        with open(path, "w") as f:
            f.write("\n".join(frame_list) + "\n")
        print(f"  Created {path} ({len(frame_list)} frames)")

    # Test split
    test_velodyne = os.path.join(data_root, "testing", "velodyne")
    if os.path.exists(test_velodyne):
        test_frames = sorted([f[:-4] for f in os.listdir(test_velodyne) if f.endswith(".bin")])
        path = os.path.join(imagesets_dir, "test.txt")
        with open(path, "w") as f:
            f.write("\n".join(test_frames) + "\n")
        print(f"  Created {path} ({len(test_frames)} frames)")


def verify_structure(data_root):
    """Verify KITTI directory structure."""
    print("\n  Verifying directory structure:")
    required = [
        ("training/velodyne", 7481),
        ("training/label_2", 7481),
        ("training/calib", 7481),
    ]
    all_ok = True
    for subdir, expected_count in required:
        path = os.path.join(data_root, subdir)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if not f.startswith(".")])
            status = "OK" if count >= expected_count else f"INCOMPLETE ({count}/{expected_count})"
            print(f"    {subdir}: {count} files [{status}]")
            if count < expected_count:
                all_ok = False
        else:
            print(f"    {subdir}: MISSING")
            all_ok = False

    return all_ok


def main():
    args = parse_args()
    os.makedirs(args.data_root, exist_ok=True)

    print("=" * 60)
    print("KITTI 3D Object Detection Dataset Preparation")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Components: {args.components}")
    print()

    if not args.skip_download:
        print("Downloading components:")
        for comp in args.components:
            info = KITTI_URLS[comp]
            print(f"\n  [{comp}] {info['description']} ({info['size']})")
            download_component(comp, info["url"], args.data_root)

    print("\nCreating ImageSets (train/val/test splits):")
    create_imagesets(args.data_root)

    ok = verify_structure(args.data_root)

    print("\n" + "=" * 60)
    if ok:
        print("KITTI dataset is ready.")
        print(f"\nNext steps:")
        print(f"  1. Install OpenPCDet: https://github.com/open-mmlab/OpenPCDet")
        print(f"  2. Create OpenPCDet data info files:")
        print(f"     cd OpenPCDet && python -m pcdet.datasets.kitti.kitti_dataset \\")
        print(f"       create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml")
        print(f"  3. Train ensemble:")
        print(f"     bash scripts/train_ensemble.sh --seeds 0 1 2 3 4 5")
    else:
        print("KITTI dataset is INCOMPLETE. Check the missing components above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

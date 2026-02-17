"""
Unified Dataset Adapter for KITTI, CARLA, and Custom Formats.

Provides a consistent interface for loading point clouds, ground truth
labels, and calibration data from different dataset formats. Handles
format detection, coordinate transforms, and weather metadata.

Supported formats:
    - KITTI: Standard 3D object detection benchmark
    - CARLA: CARLA simulator output (KITTI-format with conditions.json)
    - Custom: User-defined format with configurable paths

This adapter enables the evaluation pipeline to work seamlessly across
different data sources without code changes.

Reference:
    Geiger et al. (2012). KITTI Vision Benchmark Suite.
    Dosovitskiy et al. (2017). CARLA: An Open Urban Driving Simulator.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class DatasetAdapter:
    """
    Unified adapter for loading 3D object detection datasets.

    Automatically detects dataset format and provides consistent access
    to point clouds, ground truth boxes, calibration, and metadata.

    Parameters
    ----------
    data_root : str
        Root directory of the dataset.
    split : str
        Dataset split: 'train', 'val', or 'test'.
    format : str, optional
        Dataset format: 'auto', 'kitti', 'carla', or 'custom'.
        If 'auto', format is detected from directory structure.
    classes : list of str, optional
        Object classes to include. Default: ['Car', 'Pedestrian', 'Cyclist'].

    Usage
    -----
    >>> adapter = DatasetAdapter('data/kitti', split='val')
    >>> frame_ids = adapter.get_frame_ids()
    >>> points = adapter.load_points(frame_ids[0])
    >>> gt_boxes = adapter.load_gt_boxes(frame_ids[0])
    >>> conditions = adapter.get_conditions()  # Only for CARLA data
    """

    def __init__(
        self,
        data_root: str,
        split: str = "val",
        format: str = "auto",
        classes: Optional[List[str]] = None,
    ):
        self.data_root = os.path.abspath(data_root)
        self.split = split
        self.classes = classes or ["Car", "Pedestrian", "Cyclist"]

        if format == "auto":
            self.format = self._detect_format()
        else:
            self.format = format

        self._setup_paths()
        self._frame_ids = None
        self._conditions = None

    def _detect_format(self) -> str:
        """Detect dataset format from directory structure."""
        # Check for CARLA conditions metadata
        if os.path.exists(os.path.join(self.data_root, "conditions.json")):
            return "carla"

        # Check for standard KITTI structure
        if os.path.exists(os.path.join(self.data_root, "training", "velodyne")):
            return "kitti"

        # Check for flat structure (velodyne directly in root)
        if os.path.exists(os.path.join(self.data_root, "velodyne")):
            return "custom"

        return "kitti"

    def _setup_paths(self):
        """Set up data paths based on format."""
        if self.format in ("kitti", "carla"):
            base = os.path.join(self.data_root, "training")
            self.velodyne_dir = os.path.join(base, "velodyne")
            self.label_dir = os.path.join(base, "label_2")
            self.calib_dir = os.path.join(base, "calib")
            self.image_dir = os.path.join(base, "image_2")
            self.imageset_dir = os.path.join(self.data_root, "ImageSets")
        else:
            self.velodyne_dir = os.path.join(self.data_root, "velodyne")
            self.label_dir = os.path.join(self.data_root, "label_2")
            self.calib_dir = os.path.join(self.data_root, "calib")
            self.image_dir = os.path.join(self.data_root, "image_2")
            self.imageset_dir = os.path.join(self.data_root, "ImageSets")

    def get_frame_ids(self) -> List[str]:
        """
        Get list of frame IDs for the current split.

        Returns
        -------
        list of str
            Frame IDs (e.g., ['000000', '000001', ...]).
        """
        if self._frame_ids is not None:
            return self._frame_ids

        # Try ImageSets first
        split_file = os.path.join(self.imageset_dir, f"{self.split}.txt")
        if os.path.exists(split_file):
            with open(split_file, "r") as f:
                self._frame_ids = [line.strip() for line in f if line.strip()]
            return self._frame_ids

        # Fall back to listing velodyne directory
        if os.path.exists(self.velodyne_dir):
            self._frame_ids = sorted([
                f[:-4] for f in os.listdir(self.velodyne_dir)
                if f.endswith(".bin")
            ])
            return self._frame_ids

        # Fall back to listing label directory
        if os.path.exists(self.label_dir):
            self._frame_ids = sorted([
                f[:-4] for f in os.listdir(self.label_dir)
                if f.endswith(".txt")
            ])
            return self._frame_ids

        return []

    def load_points(
        self,
        frame_id: str,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        zlim: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """
        Load point cloud for a frame.

        Parameters
        ----------
        frame_id : str
            Frame identifier.
        xlim, ylim, zlim : tuple of (min, max), optional
            Crop ranges in LiDAR frame.

        Returns
        -------
        np.ndarray, shape (N, 4)
            Point cloud [x, y, z, intensity].
        """
        from sotif_uncertainty.kitti_utils import load_point_cloud

        bin_path = os.path.join(self.velodyne_dir, f"{frame_id}.bin")
        if not os.path.exists(bin_path):
            return np.zeros((0, 4), dtype=np.float32)

        return load_point_cloud(bin_path, xlim=xlim, ylim=ylim, zlim=zlim)

    def load_gt_boxes(
        self,
        frame_id: str,
        as_lidar: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load ground truth bounding boxes for a frame.

        Parameters
        ----------
        frame_id : str
            Frame identifier.
        as_lidar : bool
            If True, transform to LiDAR coordinates.

        Returns
        -------
        boxes : np.ndarray, shape (M, 7)
            Ground truth boxes [x, y, z, dx, dy, dz, heading].
        names : np.ndarray, shape (M,)
            Class names per box.
        """
        label_path = os.path.join(self.label_dir, f"{frame_id}.txt")
        if not os.path.exists(label_path):
            return np.zeros((0, 7)), np.array([], dtype=str)

        from sotif_uncertainty.kitti_utils import load_kitti_label

        label_data = load_kitti_label(label_path, self.classes)

        if len(label_data["boxes_cam"]) == 0:
            return np.zeros((0, 7)), np.array([], dtype=str)

        if as_lidar:
            calib_path = os.path.join(self.calib_dir, f"{frame_id}.txt")
            if os.path.exists(calib_path):
                from sotif_uncertainty.kitti_utils import KITTICalibration
                calib = KITTICalibration(calib_path)
                boxes_lidar = calib.boxes_cam_to_lidar(label_data["boxes_cam"])
            else:
                # No calibration: assume labels are already in LiDAR frame
                boxes_lidar = label_data["boxes_cam"]
        else:
            boxes_lidar = label_data["boxes_cam"]

        return boxes_lidar, label_data["names"]

    def load_calibration(self, frame_id: str):
        """
        Load calibration data for a frame.

        Returns
        -------
        KITTICalibration or None
        """
        calib_path = os.path.join(self.calib_dir, f"{frame_id}.txt")
        if not os.path.exists(calib_path):
            return None

        from sotif_uncertainty.kitti_utils import KITTICalibration
        return KITTICalibration(calib_path)

    def get_conditions(self) -> Optional[Dict[str, Dict]]:
        """
        Load weather/environment conditions metadata (CARLA datasets only).

        Returns
        -------
        dict or None
            Mapping from frame_id to condition metadata.
        """
        if self._conditions is not None:
            return self._conditions

        conditions_path = os.path.join(self.data_root, "conditions.json")
        if os.path.exists(conditions_path):
            with open(conditions_path, "r") as f:
                self._conditions = json.load(f)
            return self._conditions

        return None

    def get_frame_condition(self, frame_id: str) -> str:
        """
        Get triggering condition category for a specific frame.

        Parameters
        ----------
        frame_id : str
            Frame identifier.

        Returns
        -------
        str
            Triggering condition category (e.g., 'heavy_rain', 'night').
        """
        conditions = self.get_conditions()
        if conditions is None:
            return "other"

        frame_meta = conditions.get(str(frame_id), {})
        return frame_meta.get("tc_category", "other")

    def get_condition_array(self, frame_ids: np.ndarray) -> np.ndarray:
        """
        Get condition category for an array of frame IDs.

        Parameters
        ----------
        frame_ids : np.ndarray
            Frame identifiers.

        Returns
        -------
        np.ndarray of str
            Condition category per frame.
        """
        conditions = np.empty(len(frame_ids), dtype="U20")
        for i, fid in enumerate(frame_ids):
            conditions[i] = self.get_frame_condition(fid)
        return conditions

    def summary(self) -> Dict:
        """
        Get dataset summary statistics.

        Returns
        -------
        dict with dataset information.
        """
        frame_ids = self.get_frame_ids()
        n_frames = len(frame_ids)

        info = {
            "format": self.format,
            "data_root": self.data_root,
            "split": self.split,
            "n_frames": n_frames,
            "classes": self.classes,
            "has_velodyne": os.path.exists(self.velodyne_dir),
            "has_labels": os.path.exists(self.label_dir),
            "has_calib": os.path.exists(self.calib_dir),
            "has_conditions": self.get_conditions() is not None,
        }

        # Count objects
        if os.path.exists(self.label_dir) and n_frames > 0:
            total_objects = 0
            sample_ids = frame_ids[:min(10, n_frames)]
            for fid in sample_ids:
                boxes, names = self.load_gt_boxes(fid)
                total_objects += len(boxes)
            info["avg_objects_per_frame"] = total_objects / len(sample_ids)

        # Condition distribution for CARLA
        conditions = self.get_conditions()
        if conditions is not None:
            categories = {}
            for fid, meta in conditions.items():
                tc = meta.get("tc_category", "other")
                categories[tc] = categories.get(tc, 0) + 1
            info["condition_distribution"] = categories

        return info


def load_dataset(
    data_root: str,
    split: str = "val",
    format: str = "auto",
) -> DatasetAdapter:
    """
    Convenience function to create a DatasetAdapter.

    Parameters
    ----------
    data_root : str
        Root directory of the dataset.
    split : str
        Dataset split.
    format : str
        Format override ('auto', 'kitti', 'carla', 'custom').

    Returns
    -------
    DatasetAdapter
    """
    return DatasetAdapter(data_root, split=split, format=format)

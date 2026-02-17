"""
KITTI Dataset Utilities.

Provides calibration loading, coordinate transforms (camera <-> LiDAR),
label parsing, and point cloud I/O for the KITTI 3D Object Detection
benchmark.

The KITTI dataset stores 3D bounding box annotations in the **camera**
coordinate frame. To compare detector outputs (which operate in the
**LiDAR** coordinate frame), annotations must be transformed using the
calibration matrices provided with each frame.

Coordinate frames (KITTI convention):
    Camera:  x-right, y-down,  z-forward
    LiDAR:   x-forward, y-left, z-up
    Velodyne = LiDAR in KITTI terminology

Reference:
    Geiger et al. (2012). "Are we ready for autonomous driving?
    The KITTI vision benchmark suite." CVPR.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple


class KITTICalibration:
    """
    Load and apply KITTI calibration matrices for a single frame.

    Calibration file format (7 lines):
        P0: 3x4 projection matrix (left grayscale camera)
        P1: 3x4 projection matrix (right grayscale camera)
        P2: 3x4 projection matrix (left color camera)
        P3: 3x4 projection matrix (right color camera)
        R0_rect: 3x3 rectification rotation matrix
        Tr_velo_to_cam: 3x4 Velodyne-to-camera transformation
        Tr_imu_to_velo: 3x4 IMU-to-Velodyne transformation

    Usage:
        calib = KITTICalibration('data/kitti/training/calib/000000.txt')
        boxes_lidar = calib.boxes_cam_to_lidar(boxes_cam)
    """

    def __init__(self, calib_filepath: str):
        calib = self._load_calib_file(calib_filepath)
        self.P2 = calib["P2"].reshape(3, 4)
        self.R0_rect = np.eye(4)
        self.R0_rect[:3, :3] = calib["R0_rect"].reshape(3, 3)
        self.Tr_velo_to_cam = np.eye(4)
        self.Tr_velo_to_cam[:3, :] = calib["Tr_velo_to_cam"].reshape(3, 4)

        # Pre-compute inverse for cam->lidar
        self.Tr_cam_to_velo = np.linalg.inv(self.Tr_velo_to_cam)
        self.R0_rect_inv = np.linalg.inv(self.R0_rect)

    @staticmethod
    def _load_calib_file(filepath: str) -> Dict[str, np.ndarray]:
        """Parse a KITTI calibration file into a dictionary of arrays."""
        data = {}
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                data[key.strip()] = np.array(
                    [float(x) for x in value.strip().split()]
                )
        return data

    def lidar_to_camera(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Transform points from LiDAR to rectified camera coordinates.

        Parameters
        ----------
        pts_lidar : np.ndarray, shape (N, 3)
            Points in LiDAR frame [x, y, z].

        Returns
        -------
        np.ndarray, shape (N, 3)
            Points in rectified camera frame.
        """
        N = pts_lidar.shape[0]
        pts_hom = np.hstack([pts_lidar, np.ones((N, 1))])
        pts_cam = (self.R0_rect @ self.Tr_velo_to_cam @ pts_hom.T).T
        return pts_cam[:, :3]

    def camera_to_lidar(self, pts_cam: np.ndarray) -> np.ndarray:
        """
        Transform points from rectified camera to LiDAR coordinates.

        Parameters
        ----------
        pts_cam : np.ndarray, shape (N, 3)
            Points in rectified camera frame [x, y, z].

        Returns
        -------
        np.ndarray, shape (N, 3)
            Points in LiDAR frame.
        """
        N = pts_cam.shape[0]
        pts_hom = np.hstack([pts_cam, np.ones((N, 1))])
        pts_lidar = (self.Tr_cam_to_velo @ self.R0_rect_inv @ pts_hom.T).T
        return pts_lidar[:, :3]

    def boxes_cam_to_lidar(self, boxes_cam: np.ndarray) -> np.ndarray:
        """
        Convert 3D bounding boxes from camera to LiDAR coordinates.

        KITTI annotation format (camera frame):
            [h, w, l, x, y, z, ry]
            where (x, y, z) is the 3D center in camera coords,
            and ry is rotation around Y-axis (camera Y = down).

        Output format (LiDAR frame):
            [x, y, z, dx, dy, dz, heading]
            where (x, y, z) is center in LiDAR coords,
            (dx, dy, dz) = (l, w, h) in LiDAR frame,
            heading is rotation around Z-axis (LiDAR Z = up).

        Parameters
        ----------
        boxes_cam : np.ndarray, shape (N, 7)
            Boxes in camera frame [h, w, l, x, y, z, ry].

        Returns
        -------
        np.ndarray, shape (N, 7)
            Boxes in LiDAR frame [x, y, z, dx, dy, dz, heading].
        """
        if len(boxes_cam) == 0:
            return np.zeros((0, 7))

        h = boxes_cam[:, 0]
        w = boxes_cam[:, 1]
        l = boxes_cam[:, 2]
        x_cam = boxes_cam[:, 3]
        y_cam = boxes_cam[:, 4]
        z_cam = boxes_cam[:, 5]
        ry = boxes_cam[:, 6]

        # KITTI annotation convention: y is the bottom of the box
        # Convert to center: shift up by h/2
        y_center = y_cam - h / 2.0

        # Transform center from camera to LiDAR
        centers_cam = np.stack([x_cam, y_center, z_cam], axis=1)
        centers_lidar = self.camera_to_lidar(centers_cam)

        # Dimensions: camera (h, w, l) -> LiDAR (l, w, h)
        # because camera z-forward maps to lidar x-forward
        dx = l  # length along lidar x
        dy = w  # width along lidar y
        dz = h  # height along lidar z

        # Heading: camera ry (rotation around Y-down) -> LiDAR heading
        # (rotation around Z-up). ry=0 means forward in camera = forward
        # in LiDAR, but axes differ: heading = -(ry + pi/2)
        # Standard conversion: heading = -(ry + pi/2), then normalize
        heading = -(ry + np.pi / 2.0)
        # Normalize to [-pi, pi]
        heading = np.arctan2(np.sin(heading), np.cos(heading))

        boxes_lidar = np.stack(
            [centers_lidar[:, 0], centers_lidar[:, 1], centers_lidar[:, 2],
             dx, dy, dz, heading],
            axis=1,
        )
        return boxes_lidar

    def boxes_lidar_to_cam(self, boxes_lidar: np.ndarray) -> np.ndarray:
        """
        Convert 3D bounding boxes from LiDAR to camera coordinates.

        Parameters
        ----------
        boxes_lidar : np.ndarray, shape (N, 7)
            Boxes in LiDAR frame [x, y, z, dx, dy, dz, heading].

        Returns
        -------
        np.ndarray, shape (N, 7)
            Boxes in camera frame [h, w, l, x, y, z, ry].
        """
        if len(boxes_lidar) == 0:
            return np.zeros((0, 7))

        centers_lidar = boxes_lidar[:, :3]
        dx, dy, dz = boxes_lidar[:, 3], boxes_lidar[:, 4], boxes_lidar[:, 5]
        heading = boxes_lidar[:, 6]

        centers_cam = self.lidar_to_camera(centers_lidar)
        h, w, l = dz, dy, dx
        ry = -(heading + np.pi / 2.0)
        ry = np.arctan2(np.sin(ry), np.cos(ry))

        # Camera: y is bottom of box
        y_bottom = centers_cam[:, 1] + h / 2.0

        boxes_cam = np.stack(
            [h, w, l, centers_cam[:, 0], y_bottom, centers_cam[:, 2], ry],
            axis=1,
        )
        return boxes_cam


def load_kitti_label(
    label_filepath: str,
    classes: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load KITTI 3D object detection label file.

    KITTI label format (each line, 15+ fields):
        type truncated occluded alpha
        bbox_2d(4) dimensions(3) location(3) rotation_y [score]

    Parameters
    ----------
    label_filepath : str
        Path to .txt label file.
    classes : list of str, optional
        Filter to these classes. Default: ['Car', 'Pedestrian', 'Cyclist'].

    Returns
    -------
    dict with keys:
        'names' : (N,) str array of class names
        'boxes_cam' : (N, 7) [h, w, l, x, y, z, ry] in camera frame
        'difficulty' : (N,) int -- 0=easy, 1=moderate, 2=hard
        'truncated' : (N,) float
        'occluded' : (N,) int
        'bbox_2d' : (N, 4) [left, top, right, bottom]
    """
    if classes is None:
        classes = ["Car", "Pedestrian", "Cyclist"]

    names, boxes, difficulties = [], [], []
    truncations, occlusions, bboxes_2d = [], [], []

    if not os.path.exists(label_filepath):
        return {
            "names": np.array([], dtype=str),
            "boxes_cam": np.zeros((0, 7)),
            "difficulty": np.zeros(0, dtype=int),
            "truncated": np.zeros(0),
            "occluded": np.zeros(0, dtype=int),
            "bbox_2d": np.zeros((0, 4)),
        }

    with open(label_filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue

            cls = parts[0]
            if cls not in classes and cls != "DontCare":
                continue
            if cls == "DontCare":
                continue

            truncated = float(parts[1])
            occluded = int(parts[2])
            bbox = [float(parts[4]), float(parts[5]),
                    float(parts[6]), float(parts[7])]
            h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
            x, y, z = float(parts[11]), float(parts[12]), float(parts[13])
            ry = float(parts[14])

            # Determine difficulty level
            bbox_height = bbox[3] - bbox[1]
            if occluded == 0 and truncated <= 0.15 and bbox_height >= 40:
                difficulty = 0  # Easy
            elif occluded <= 1 and truncated <= 0.30 and bbox_height >= 25:
                difficulty = 1  # Moderate
            elif occluded <= 2 and truncated <= 0.50 and bbox_height >= 25:
                difficulty = 2  # Hard
            else:
                difficulty = 3  # Unknown / very hard

            names.append(cls)
            boxes.append([h, w, l, x, y, z, ry])
            difficulties.append(difficulty)
            truncations.append(truncated)
            occlusions.append(occluded)
            bboxes_2d.append(bbox)

    return {
        "names": np.array(names),
        "boxes_cam": np.array(boxes) if boxes else np.zeros((0, 7)),
        "difficulty": np.array(difficulties, dtype=int),
        "truncated": np.array(truncations),
        "occluded": np.array(occlusions, dtype=int),
        "bbox_2d": np.array(bboxes_2d) if bboxes_2d else np.zeros((0, 4)),
    }


def load_kitti_labels_as_lidar(
    label_filepath: str,
    calib_filepath: str,
    classes: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Load KITTI labels and convert to LiDAR frame coordinates.

    This is the primary function for ground truth loading during evaluation.

    Parameters
    ----------
    label_filepath : str
        Path to label .txt file.
    calib_filepath : str
        Path to calibration .txt file.
    classes : list of str, optional
        Filter to these classes.

    Returns
    -------
    np.ndarray, shape (N, 7)
        Ground truth boxes in LiDAR frame [x, y, z, dx, dy, dz, heading].
    """
    label_data = load_kitti_label(label_filepath, classes)
    if len(label_data["boxes_cam"]) == 0:
        return np.zeros((0, 7))

    calib = KITTICalibration(calib_filepath)
    return calib.boxes_cam_to_lidar(label_data["boxes_cam"])


def load_point_cloud(
    bin_filepath: str,
    remove_close: float = 0.0,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    zlim: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Load a KITTI Velodyne point cloud from a .bin file.

    Each point has 4 values: [x, y, z, intensity].

    Parameters
    ----------
    bin_filepath : str
        Path to .bin file.
    remove_close : float
        Remove points closer than this distance from sensor (default: 0).
    xlim, ylim, zlim : tuple of (min, max), optional
        Crop point cloud to these ranges.

    Returns
    -------
    np.ndarray, shape (N, 4)
        Point cloud [x, y, z, intensity].
    """
    points = np.fromfile(bin_filepath, dtype=np.float32).reshape(-1, 4)

    if remove_close > 0:
        dist = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        points = points[dist > remove_close]

    if xlim is not None:
        mask = (points[:, 0] >= xlim[0]) & (points[:, 0] <= xlim[1])
        points = points[mask]
    if ylim is not None:
        mask = (points[:, 1] >= ylim[0]) & (points[:, 1] <= ylim[1])
        points = points[mask]
    if zlim is not None:
        mask = (points[:, 2] >= zlim[0]) & (points[:, 2] <= zlim[1])
        points = points[mask]

    return points


def voxelize_point_cloud(
    points: np.ndarray,
    voxel_size: Tuple[float, float, float] = (0.05, 0.05, 0.1),
    point_cloud_range: Tuple[float, ...] = (0.0, -40.0, -3.0, 70.4, 40.0, 1.0),
    max_points_per_voxel: int = 5,
    max_voxels: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Voxelize a point cloud for use with voxel-based detectors (SECOND, etc.).

    This is a simplified CPU-based voxelization. For production use,
    spconv's VoxelGenerator or OpenPCDet's DataProcessor is faster.

    Parameters
    ----------
    points : np.ndarray, shape (N, 4)
        Point cloud [x, y, z, intensity].
    voxel_size : tuple of 3 floats
        Voxel dimensions [dx, dy, dz] in meters.
    point_cloud_range : tuple of 6 floats
        [x_min, y_min, z_min, x_max, y_max, z_max].
    max_points_per_voxel : int
        Maximum points per voxel.
    max_voxels : int
        Maximum number of non-empty voxels.

    Returns
    -------
    voxels : np.ndarray, shape (M, max_points_per_voxel, 4)
        Voxel features.
    coords : np.ndarray, shape (M, 3)
        Voxel coordinates [z_idx, y_idx, x_idx].
    num_points : np.ndarray, shape (M,)
        Number of points in each voxel.
    """
    pcr = np.array(point_cloud_range)
    vs = np.array(voxel_size)

    # Filter points to range
    mask = (
        (points[:, 0] >= pcr[0]) & (points[:, 0] < pcr[3]) &
        (points[:, 1] >= pcr[1]) & (points[:, 1] < pcr[4]) &
        (points[:, 2] >= pcr[2]) & (points[:, 2] < pcr[5])
    )
    points = points[mask]

    # Compute voxel indices
    grid_size = np.round((pcr[3:] - pcr[:3]) / vs).astype(np.int32)
    indices = np.floor((points[:, :3] - pcr[:3]) / vs).astype(np.int32)

    # Clip to valid range
    indices = np.clip(indices, 0, grid_size - 1)

    # Use dictionary for voxel accumulation
    voxel_dict = {}
    for i in range(len(points)):
        key = (indices[i, 0], indices[i, 1], indices[i, 2])
        if key not in voxel_dict:
            voxel_dict[key] = []
        if len(voxel_dict[key]) < max_points_per_voxel:
            voxel_dict[key].append(points[i])

        if len(voxel_dict) >= max_voxels:
            break

    # Convert to arrays
    M = len(voxel_dict)
    voxels = np.zeros((M, max_points_per_voxel, 4), dtype=np.float32)
    coords = np.zeros((M, 3), dtype=np.int32)
    num_points = np.zeros(M, dtype=np.int32)

    for idx, (key, pts) in enumerate(voxel_dict.items()):
        n = len(pts)
        voxels[idx, :n] = np.array(pts)
        coords[idx] = [key[2], key[1], key[0]]  # z, y, x order for spconv
        num_points[idx] = n

    return voxels, coords, num_points


def get_kitti_frame_ids(
    split_file: str,
) -> List[str]:
    """
    Read frame IDs from a KITTI ImageSets split file.

    Parameters
    ----------
    split_file : str
        Path to train.txt, val.txt, or test.txt.

    Returns
    -------
    list of str
        Frame IDs (e.g., ['000000', '000001', ...]).
    """
    with open(split_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def compute_3d_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute approximate 3D IoU between two axis-aligned boxes.

    Parameters
    ----------
    box_a, box_b : np.ndarray, shape (7,)
        [x, y, z, dx, dy, dz, heading] in LiDAR frame.

    Returns
    -------
    float
        3D IoU in [0, 1].
    """
    # Use axis-aligned approximation (ignore heading)
    xa, ya, za = box_a[:3]
    dxa, dya, dza = box_a[3:6]
    xb, yb, zb = box_b[:3]
    dxb, dyb, dzb = box_b[3:6]

    # Intersection
    x_overlap = max(0, min(xa + dxa / 2, xb + dxb / 2) - max(xa - dxa / 2, xb - dxb / 2))
    y_overlap = max(0, min(ya + dya / 2, yb + dyb / 2) - max(ya - dya / 2, yb - dyb / 2))
    z_overlap = max(0, min(za + dza / 2, zb + dzb / 2) - max(za - dza / 2, zb - dzb / 2))

    inter = x_overlap * y_overlap * z_overlap
    vol_a = dxa * dya * dza
    vol_b = dxb * dyb * dzb
    union = vol_a + vol_b - inter

    if union < 1e-8:
        return 0.0
    return float(inter / union)

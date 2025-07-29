#!/usr/bin/env python3
"""
lidar_cam_calib.py

Core functionality for LiDAR→Camera extrinsic calibration using:
- Camera intrinsics (OpenCV YAML)
- Board→camera poses (JSON from your extract_cam_poses.py)
- Ouster LiDAR CSVs (x,y,z columns)
- Checkerboard geometry (pattern, square size)

Implements:
- CSV loader with column auto-detection
- Plane fit (Open3D RANSAC) + PCA orientation
- Checkerboard cropping in board coordinates
- Hand–eye solver (AX = X B)
- Projection & fat depth-colored overlay
"""

import re
import json
import yaml
import csv
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# ----------------------------- CSV utilities ----------------------------- #
def load_lidar_points_from_csv(csv_path: Path) -> np.ndarray:
    """
    Load x,y,z columns from an Ouster Studio CSV. Unit auto-detected (mm->m).
    """
    import pandas as pd
    df = pd.read_csv(csv_path, sep=None, engine="python")

    def pick(prefix):
        pattern = re.compile(rf"^{prefix}\d*\s*(\(mm\))?$", re.IGNORECASE)
        for c in df.columns:
            if pattern.match(c.strip()):
                return c
        for c in df.columns:
            if prefix.lower() in c.lower():
                return c
        raise ValueError(f"{csv_path} missing {prefix} column. Columns: {df.columns.tolist()}")

    cx, cy, cz = pick("x"), pick("y"), pick("z")
    pts = df[[cx, cy, cz]].to_numpy(dtype=np.float64)
    # if mm -> convert to m
    if "(mm)" in cx.lower() or "(mm)" in cy.lower() or "(mm)" in cz.lower():
        pts /= 1000.0
    pts = pts[~np.isnan(pts).any(axis=1)]
    return pts.astype(np.float32)

# ----------------------------- Intrinsics ----------------------------- #
def load_intrinsics(yaml_path: Path):
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeff").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError(f"Could not read 'camera_matrix' or 'dist_coeff' from {yaml_path}")
    return K, D

# ----------------------------- Board Poses ----------------------------- #
def load_board_pose(board_json: Path, stem: str) -> np.ndarray:
    """Return 4x4 T_board_cam for the entry matching image stem."""
    data = json.load(open(board_json, "r"))
    for e in data:
        s = Path(e.get("image", str(e.get("timestamp", "")))).stem
        if s == stem:
            return np.array(e["T_board_cam"], dtype=np.float64)
    raise RuntimeError(f"No board pose for stem {stem} in {board_json}")

# ----------------------------- Plane Fit & Board Frame ----------------------------- #
def fit_plane_pose(points: np.ndarray,
                   dist_thresh: float = 0.02,
                   max_pca_pts: int = 5000):
    """
    Fit a plane (RANSAC), then build a right-handed frame:
      z = plane normal
      x = principal in-plane direction
      y = z × x
    Returns:
        T_board_lidar (4x4): board->lidar
        plane_pts (Nx3): inlier points
    """
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=3,
                                             num_iterations=1000)
    a, b, c, d = plane_model
    plane_pts = points[inliers]
    if plane_pts.shape[0] == 0:
        raise RuntimeError("Plane segmentation returned 0 inliers.")

    centroid = plane_pts.mean(axis=0)

    # downsample for PCA
    if plane_pts.shape[0] > max_pca_pts:
        idx = np.random.choice(plane_pts.shape[0], max_pca_pts, replace=False)
        pca_pts = plane_pts[idx]
    else:
        pca_pts = plane_pts

    pca_pts_centered = (pca_pts - centroid).astype(np.float64)

    z_axis = np.array([a, b, c], dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)

    cov = pca_pts_centered.T @ pca_pts_centered  # 3x3
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_sorted = np.argsort(eigvals)[::-1]
    x_candidate = eigvecs[:, idx_sorted[0]]
    x_axis = x_candidate - (x_candidate @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    if np.linalg.det(np.stack([x_axis, y_axis, z_axis], axis=1)) < 0:
        x_axis *= -1
        y_axis *= -1

    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3]  = centroid
    return T, plane_pts

def filter_points_on_board(points_lidar: np.ndarray,
                           T_board_lidar: np.ndarray,
                           board_w: float,
                           board_h: float,
                           z_tol: float = 0.01,
                           margin: float = 0.02):
    """
    Keep points near plane (|z_board|<z_tol) & inside board rectangle (|x|<w+M, |y|<h+M).
    Returns filtered points_lidar.
    """
    TbL_inv = np.linalg.inv(T_board_lidar)
    pts_h = np.hstack([points_lidar, np.ones((points_lidar.shape[0],1), dtype=points_lidar.dtype)])
    pts_board = (TbL_inv @ pts_h.T).T[:, :3]

    w = board_w + margin
    h = board_h + margin
    mask = (np.abs(pts_board[:,0]) <= w) & \
           (np.abs(pts_board[:,1]) <= h) & \
           (np.abs(pts_board[:,2]) <= z_tol)
    return points_lidar[mask]

# ----------------------------- Hand–Eye ----------------------------- #
def solve_hand_eye(A_list, B_list):
    """
    Classic AX = X B (Tsai-like) solver.
    A: board->camera inverse (cam->board)
    B: board->lidar
    Returns X = lidar->camera
    """
    RA = [A[:3, :3] for A in A_list]
    RB = [B[:3, :3] for B in B_list]
    tA = [A[:3, 3]   for A in A_list]
    tB = [B[:3, 3]   for B in B_list]

    # rotation
    M = np.zeros((3,3))
    for Ra, Rb in zip(RA, RB):
        M += Rb @ Rb.T - Ra @ Ra.T
    U, _, Vt = np.linalg.svd(M)
    R_x = U @ Vt
    if np.linalg.det(R_x) < 0:
        U[:, -1] *= -1
        R_x = U @ Vt

    # translation
    I = np.eye(3)
    A_mat = []
    b_vec = []
    for Ra, Rb, ta, tb in zip(RA, RB, tA, tB):
        A_mat.append(Ra - I)
        b_vec.append(R_x @ tb - ta)
    A_mat = np.vstack(A_mat)
    b_vec = np.hstack(b_vec)
    t_x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    X = np.eye(4)
    X[:3,:3] = R_x
    X[:3, 3] = t_x
    return X

def average_transforms(T_list):
    Rs = R.from_matrix([T[:3,:3] for T in T_list])
    R_avg = Rs.mean().as_matrix()
    t_avg = np.mean([T[:3,3] for T in T_list], axis=0)
    T_avg = np.eye(4)
    T_avg[:3,:3] = R_avg
    T_avg[:3,3]  = t_avg
    return T_avg

# ----------------------------- Projection ----------------------------- #
def project_points(points_lidar, T_lidar_cam, K, D, img_shape, use_distortion=True):
    """
    Transform lidar points to camera, project, clean NaN/Inf/out-of-bounds.
    Returns (pts_xy, depths).
    """
    MAX_PIXEL_VAL = 1e6
    MIN_DEPTH_M   = 1e-6
    MAX_DEPTH_M   = 1000.0

    pts_h   = np.hstack([points_lidar, np.ones((points_lidar.shape[0],1), dtype=points_lidar.dtype)])
    pts_cam = (T_lidar_cam @ pts_h.T).T[:, :3]

    mask_z  = pts_cam[:,2] > 0
    pts_cam = pts_cam[mask_z]
    if pts_cam.size == 0:
        return np.empty((0,2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    rvec = np.zeros((3,1), dtype=np.float64)
    tvec = np.zeros((3,1), dtype=np.float64)
    dist = D if use_distortion else np.zeros_like(D)

    img_pts_raw, _ = cv2.projectPoints(pts_cam, rvec, tvec, K, dist)
    img_pts = np.asarray(img_pts_raw.reshape(-1,2), dtype=np.float64)

    finite = np.isfinite(img_pts).all(axis=1)
    img_pts = img_pts[finite]
    pts_cam = pts_cam[finite]

    depth_mask = (pts_cam[:,2] > MIN_DEPTH_M) & (pts_cam[:,2] < MAX_DEPTH_M)
    mag_mask   = (np.abs(img_pts[:,0]) < MAX_PIXEL_VAL) & (np.abs(img_pts[:,1]) < MAX_PIXEL_VAL)
    keep_mask  = depth_mask & mag_mask

    img_pts = img_pts[keep_mask]
    pts_cam = pts_cam[keep_mask]

    if img_pts.size == 0:
        return np.empty((0,2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    img_pts = np.rint(img_pts).astype(np.int32, copy=False)
    h, w = img_shape[:2]
    in_bounds = (img_pts[:,0] >= 0) & (img_pts[:,0] < w) & (img_pts[:,1] >= 0) & (img_pts[:,1] < h)
    return img_pts[in_bounds], pts_cam[in_bounds, 2].astype(np.float32)

def draw_points_depth(img, pts_xy, depths, radius=5):
    """Colorize by depth + fat circles."""
    if pts_xy.size == 0:
        return img
    d = depths.astype(np.float32)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET).reshape(-1,3)
    for (x, y), c in zip(pts_xy, colors):
        cv2.circle(img, (int(x), int(y)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return img

def save_mask(h, w, pts_xy, radius=3):
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y in pts_xy:
        cv2.circle(mask, (int(x), int(y)), radius, 255, -1)
    return mask

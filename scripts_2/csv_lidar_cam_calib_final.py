#!/usr/bin/env python3
"""
csv_lidar_cam_calib_final.py

Robust LiDAR -> Camera extrinsic calibration from:
- Camera intrinsics (OpenCV yaml)
- Board->camera poses json (from your PnP extraction script)
- LiDAR CSV frames (exported from Ouster Studio)

Outputs:
- <cam>_lidar_extrinsics.yaml
- <cam>_overlay.png (depth-colored fat points)
- <cam>_mask.png (binary projection mask)

Usage (Windows CMD, one line):
  python csv_lidar_cam_calib_final.py ^
    --intrinsics "C:\...\calib_imgs_cam0_intrinsics.yaml" ^
    --board_json "C:\...\calib_imgs_cam0_board_poses.json" ^
    --img_dir "C:\...\calib_imgs_cam0" ^
    --csv_dir "C:\...\csv_cam0" ^
    --out_dir "C:\...\calib"

"""

import argparse
import re
import json
import yaml
import numpy as np
import pandas as pd
import cv2
import open3d as o3d
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# --------------------------- Helpers --------------------------- #
def load_intrinsics(yaml_path: Path):
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeff").mat()
    fs.release()
    if K is None or D is None:
        raise ValueError(f"Can't read camera_matrix/dist_coeff in {yaml_path}")
    return K, D

def load_board_pose(board_json: Path, stem: str) -> np.ndarray:
    data = json.load(open(board_json, "r"))
    for e in data:
        s = Path(e.get("image", str(e.get("timestamp", "")))).stem
        if s == stem:
            return np.array(e["T_board_cam"], dtype=np.float64)
    raise RuntimeError(f"No board pose for stem {stem}")

def load_lidar_csv(csv_path: Path) -> np.ndarray:
    df = pd.read_csv(csv_path, sep=None, engine="python")
    def pick(prefix):
        pat = re.compile(rf"^{prefix}\d*\s*(\(mm\))?$", re.IGNORECASE)
        for c in df.columns:
            if pat.match(c.strip()):
                return c
        for c in df.columns:
            if prefix.lower() in c.lower():
                return c
        raise ValueError(f"{csv_path} missing {prefix} col. Got: {df.columns.tolist()}")
    cx, cy, cz = pick("x"), pick("y"), pick("z")
    pts = df[[cx, cy, cz]].to_numpy(dtype=np.float64)
    if "(mm)" in cx.lower() or "(mm)" in cy.lower() or "(mm)" in cz.lower():
        pts /= 1000.0
    pts = pts[~np.isnan(pts).any(axis=1)]
    return pts.astype(np.float32)

def fit_plane_pose(points, dist_thresh=0.02, max_pca_pts=5000):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    plane_pts = points[inliers]
    if plane_pts.shape[0] == 0:
        raise RuntimeError("Plane segmentation returned 0 inliers.")
    centroid = plane_pts.mean(axis=0)

    if plane_pts.shape[0] > max_pca_pts:
        idx = np.random.choice(plane_pts.shape[0], max_pca_pts, replace=False)
        pca_pts = plane_pts[idx]
    else:
        pca_pts = plane_pts

    pca_c = (pca_pts - centroid).astype(np.float64)

    z_axis = np.array([a, b, c], dtype=np.float64); z_axis /= np.linalg.norm(z_axis)
    cov = pca_c.T @ pca_c
    eigvals, eigvecs = np.linalg.eigh(cov)
    i0 = np.argsort(eigvals)[::-1][0]
    x_candidate = eigvecs[:, i0]
    x_axis = x_candidate - (x_candidate @ z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)

    if np.linalg.det(np.stack([x_axis, y_axis, z_axis], axis=1)) < 0:
        x_axis *= -1; y_axis *= -1

    T = np.eye(4)
    T[:3, :3] = np.stack([x_axis, y_axis, z_axis], axis=1)
    T[:3, 3]  = centroid
    return T, plane_pts

def filter_points_on_board(points_lidar, T_board_lidar, board_w, board_h, z_tol=0.01, margin=0.02):
    TbL_inv = np.linalg.inv(T_board_lidar)
    pts_h = np.hstack([points_lidar, np.ones((points_lidar.shape[0],1), dtype=points_lidar.dtype)])
    pts_b = (TbL_inv @ pts_h.T).T[:, :3]

    w = board_w + margin
    h = board_h + margin
    mask = (np.abs(pts_b[:,0]) <= w) & (np.abs(pts_b[:,1]) <= h) & (np.abs(pts_b[:,2]) <= z_tol)
    return points_lidar[mask]

def project_points(points_lidar, T_lidar_cam, K, D, img_shape, use_dist=True):
    MAX_PIXEL_VAL = 1e6; MIN_DEPTH=1e-6; MAX_DEPTH=1000.0
    pts_h = np.hstack([points_lidar, np.ones((points_lidar.shape[0],1), dtype=points_lidar.dtype)])
    pts_cam = (T_lidar_cam @ pts_h.T).T[:, :3]
    mask = pts_cam[:,2] > 0
    pts_cam = pts_cam[mask]
    if pts_cam.size == 0:
        return np.empty((0,2), dtype=np.int32), np.empty((0,), dtype=np.float32)
    rvec = np.zeros((3,1)); tvec = np.zeros((3,1))
    dist = D if use_dist else np.zeros_like(D)
    img_pts_raw,_ = cv2.projectPoints(pts_cam, rvec, tvec, K, dist)
    img_pts = np.asarray(img_pts_raw.reshape(-1,2), dtype=np.float64)
    finite = np.isfinite(img_pts).all(axis=1)
    img_pts = img_pts[finite]; pts_cam = pts_cam[finite]

    ok_depth = (pts_cam[:,2] > MIN_DEPTH) & (pts_cam[:,2] < MAX_DEPTH)
    ok_mag = (np.abs(img_pts[:,0]) < MAX_PIXEL_VAL) & (np.abs(img_pts[:,1]) < MAX_PIXEL_VAL)
    keep = ok_depth & ok_mag
    img_pts = img_pts[keep]; pts_cam = pts_cam[keep]
    if img_pts.size == 0:
        return np.empty((0,2), dtype=np.int32), np.empty((0,), dtype=np.float32)

    img_pts = np.rint(img_pts).astype(np.int32, copy=False)
    h, w = img_shape[:2]
    inb = (img_pts[:,0]>=0)&(img_pts[:,0]<w)&(img_pts[:,1]>=0)&(img_pts[:,1]<h)
    return img_pts[inb], pts_cam[inb,2].astype(np.float32)

def draw_points_depth(img, pts_xy, depths, radius=5):
    if pts_xy.size == 0:
        return img
    d = depths.astype(np.float32)
    d_norm = (d - d.min()) / (d.max() - d.min() + 1e-6)
    d_uint8 = (d_norm * 255).astype(np.uint8)
    colors = cv2.applyColorMap(d_uint8, cv2.COLORMAP_JET).reshape(-1,3)
    for (x,y), c in zip(pts_xy, colors):
        cv2.circle(img, (int(x), int(y)), radius, (int(c[0]), int(c[1]), int(c[2])), -1)
    return img

def save_mask(h,w,pts_xy,radius=3):
    mask = np.zeros((h,w), dtype=np.uint8)
    for x,y in pts_xy:
        cv2.circle(mask, (int(x),int(y)), radius, 255, -1)
    return mask

def average_transforms(T_list):
    Rs = R.from_matrix([T[:3,:3] for T in T_list])
    R_avg = Rs.mean().as_matrix()
    t_avg = np.mean([T[:3,3] for T in T_list], axis=0)
    T_avg = np.eye(4)
    T_avg[:3,:3] = R_avg
    T_avg[:3,3]  = t_avg
    return T_avg

# Try all axis flip combos (det=+1) to fix board frame ambiguity
def generate_flip_mats():
    flips = []
    signs = [-1,1]
    for sx in signs:
        for sy in signs:
            for sz in signs:
                M = np.diag([sx,sy,sz])
                if np.linalg.det(M) > 0:  # keep right-handed
                    flips.append(M)
    return flips

# --------------------------- Main --------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intrinsics", required=True)
    ap.add_argument("--board_json", required=True)
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--squares_x", type=int, default=7)
    ap.add_argument("--squares_y", type=int, default=5)
    ap.add_argument("--square_size", type=float, default=0.0376)
    ap.add_argument("--plane_thresh", type=float, default=0.02)
    ap.add_argument("--z_tol", type=float, default=0.01)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--max_pca_pts", type=int, default=5000)
    ap.add_argument("--min_pairs", type=int, default=3)
    ap.add_argument("--no_flip_search", action="store_true", help="disable axis-flip symmetry search")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    board_w = (args.squares_x - 1) * args.square_size
    board_h = (args.squares_y - 1) * args.square_size

    K, D = load_intrinsics(Path(args.intrinsics))

    images = sorted(Path(args.img_dir).glob("*.png"))
    if not images:
        raise SystemExit("No images found.")

    # Gather per-pair transforms
    per_pair = []
    used = []

    for img_path in images:
        stem = img_path.stem
        csv_path = Path(args.csv_dir)/f"{stem}.csv"
        if not csv_path.exists():
            continue

        # board->camera
        T_board_cam = load_board_pose(Path(args.board_json), stem)

        pts_full = load_lidar_csv(csv_path)
        if pts_full.shape[0] < 100:
            continue

        try:
            T_board_lidar, plane_pts = fit_plane_pose(pts_full,
                                                      dist_thresh=args.plane_thresh,
                                                      max_pca_pts=args.max_pca_pts)
        except RuntimeError:
            continue

        board_pts = filter_points_on_board(plane_pts, T_board_lidar,
                                           board_w, board_h,
                                           z_tol=args.z_tol,
                                           margin=args.margin)
        if board_pts.shape[0] < 20:
            # fallback to plane points
            board_pts = plane_pts

        # Two candidate transforms
        cand1 = T_board_cam @ np.linalg.inv(T_board_lidar)
        cand2 = np.linalg.inv(T_board_cam) @ T_board_lidar

        img = cv2.imread(str(img_path))
        pts_xy_1, _ = project_points(board_pts, cand1, K, D, img.shape, use_dist=True)
        pts_xy_2, _ = project_points(board_pts, cand2, K, D, img.shape, use_dist=True)

        pick = cand1 if pts_xy_1.shape[0] >= pts_xy_2.shape[0] else cand2
        # store also count for later robust filtering
        count = max(pts_xy_1.shape[0], pts_xy_2.shape[0])
        per_pair.append((pick, count))
        used.append((stem, img_path, csv_path, board_pts))

    if len(per_pair) < args.min_pairs:
        raise SystemExit(f"Only {len(per_pair)} usable pairs – need >= {args.min_pairs}")

    # Remove outliers by count (keep top 70%)
    per_pair.sort(key=lambda x: x[1], reverse=True)
    keep_n = max(args.min_pairs, int(len(per_pair)*0.7))
    per_pair_trimmed = [p[0] for p in per_pair[:keep_n]]

    # Optional axis flip search
    flips = [np.eye(3)] if args.no_flip_search else generate_flip_mats()

    best_T = None
    best_score = -1

    for F in flips:
        # Reconstruct per-pair with flips on board frame (effectively flip T_board_lidar)
        Ts_adj = []
        for (T, c) in per_pair[:keep_n]:
            # T came from T_board_cam @ inv(T_board_lidar). If we flip board axes, X -> X * F
            # Actually that's equivalent to: new_T = T * diag(F,1) because it's lidar->cam
            T_adj = np.eye(4)
            T_adj[:3,:3] = T[:3,:3] @ F
            T_adj[:3,3]  = T[:3,3]
            Ts_adj.append(T_adj)

        T_avg = average_transforms(Ts_adj)

        # Score on first frame overlay count
        stem, img_path, csv_path, board_pts = used[0]
        img = cv2.imread(str(img_path))
        pts_xy, _ = project_points(board_pts, T_avg, K, D, img.shape, use_dist=True)
        score = pts_xy.shape[0]
        if score > best_score:
            best_score = score
            best_T = T_avg

    # Save result
    name = Path(args.intrinsics).stem.replace("_intrinsics","")
    out_yaml = out_dir / f"{name}_lidar_extrinsics.yaml"
    with open(out_yaml, "w") as f:
        yaml.dump({"T_lidar_cam": {"rows":4,"cols":4,"data": best_T.flatten().tolist()}}, f)
    print("Saved extrinsic:", out_yaml)

    # Overlay visualization
    stem, img_path, csv_path, board_pts = used[0]
    img = cv2.imread(str(img_path))
    pts_xy, depths = project_points(board_pts, best_T, K, D, img.shape, use_dist=True)
    if pts_xy.shape[0] == 0:
        pts_xy, depths = project_points(board_pts, best_T, K, D, img.shape, use_dist=False)
    if pts_xy.shape[0] == 0:
        # fallback full cloud
        full_pts = load_lidar_csv(csv_path)
        pts_xy, depths = project_points(full_pts, best_T, K, D, img.shape, use_dist=True)
        if pts_xy.shape[0] == 0:
            pts_xy, depths = project_points(full_pts, best_T, K, D, img.shape, use_dist=False)

    overlay = draw_points_depth(img.copy(), pts_xy, depths, radius=5)
    mask = save_mask(img.shape[0], img.shape[1], pts_xy, radius=3)

    out_overlay = out_dir / f"{name}_overlay.png"
    out_mask    = out_dir / f"{name}_mask.png"
    cv2.imwrite(str(out_overlay), overlay)
    cv2.imwrite(str(out_mask), mask)
    print("Overlay:", out_overlay)
    print("Mask   :", out_mask)

    cv2.imshow("overlay", overlay)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

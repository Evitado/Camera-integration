#!/usr/bin/env python3
"""
run_calibration.py

CLI tool to calibrate LiDAR->Camera for one camera folder (cam0/cam1/...)
using:
- Intrinsics YAML
- Board pose JSON
- Image folder
- CSV folder

Outputs:
- <out_dir>/<cam>_lidar_extrinsics.yaml
- <out_dir>/<cam>_overlay.png
- <out_dir>/<cam>_mask.png
"""

import argparse
import glob
from pathlib import Path
import yaml
import cv2
import numpy as np

from ex_cali import (
    load_intrinsics,
    load_board_pose,
    load_lidar_points_from_csv,
    fit_plane_pose,
    filter_points_on_board,
    solve_hand_eye,
    average_transforms,
    project_points,
    draw_points_depth,
    save_mask,
)

def main():
    parser = argparse.ArgumentParser(description="LiDAR -> Camera extrinsic calibration from CSVs")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics YAML")
    parser.add_argument("--board_json", required=True, help="Path to board poses JSON")
    parser.add_argument("--img_dir", required=True, help="Directory with camera images (png)")
    parser.add_argument("--csv_dir", required=True, help="Directory with CSV LiDAR frames")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--squares_x", type=int, default=7, help="Number of squares on checkerboard in X")
    parser.add_argument("--squares_y", type=int, default=5, help="Number of squares on checkerboard in Y")
    parser.add_argument("--square_size", type=float, default=0.0376, help="Square size in meters")
    parser.add_argument("--plane_thresh", type=float, default=0.02, help="RANSAC plane distance threshold (m)")
    parser.add_argument("--z_tol", type=float, default=0.01, help="Max |z| in board frame (m) for cropping")
    parser.add_argument("--margin", type=float, default=0.02, help="Extra margin (m) around board extents")
    parser.add_argument("--max_pca_pts", type=int, default=5000, help="Max plane points for PCA")
    parser.add_argument("--min_pairs", type=int, default=3, help="Minimum pose pairs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Derived board size from inner corners (pattern = (squares_x-1, squares_y-1))
    board_w = (args.squares_x - 1) * args.square_size
    board_h = (args.squares_y - 1) * args.square_size

    # Load intrinsics
    K, D = load_intrinsics(Path(args.intrinsics))

    # get list of PNG images
    images = sorted(Path(args.img_dir).glob("*.png"))
    if not images:
        raise SystemExit("No PNG images found in img_dir.")

    A_list, B_list = [], []
    used = []

    for img_path in images:
        stem = img_path.stem
        csv_path = Path(args.csv_dir) / f"{stem}.csv"
        if not csv_path.exists():
            continue

        # board->camera
        T_board_cam = load_board_pose(Path(args.board_json), stem)
        A = np.linalg.inv(T_board_cam)  # cam->board for AX = X B
        # lidar CSV
        pts_full = load_lidar_points_from_csv(csv_path)
        if pts_full.shape[0] < 100:
            continue

        # plane fit
        try:
            T_board_lidar, plane_pts = fit_plane_pose(pts_full,
                                                      dist_thresh=args.plane_thresh,
                                                      max_pca_pts=args.max_pca_pts)
        except RuntimeError:
            continue

        # filter to board rectangle
        board_pts = filter_points_on_board(
            plane_pts, T_board_lidar,
            board_w, board_h,
            z_tol=args.z_tol, margin=args.margin
        )
        if board_pts.shape[0] < 20:
            # fallback to plane_pts if cropping left too few
            board_pts = plane_pts

        B = T_board_lidar  # board->lidar
        A_list.append(A)
        B_list.append(B)
        used.append((stem, img_path, csv_path))

    if len(A_list) < args.min_pairs:
        raise SystemExit(f"Only {len(A_list)} valid pose pairs. Need >= {args.min_pairs}.")

    print(f"Solving hand-eye with {len(A_list)} pairs...")
    T_lidar_cam = solve_hand_eye(A_list, B_list)
    # (Optional) refine by averaging per-pair X_i = A_i^-1 B_i
    # You can compute per_pair_X and average_transforms if you prefer:
    # per_pair = [np.linalg.inv(A_list[i]) @ B_list[i] for i in range(len(A_list))]
    # T_lidar_cam = average_transforms(per_pair)

    # Save transform as OpenCV-style YAML
    out_yml = out_dir / f"{Path(args.intrinsics).stem.replace('_intrinsics','')}_lidar_extrinsics.yaml"
    with open(out_yml, "w") as f:
        yaml.dump({"T_lidar_cam": {"rows": 4, "cols": 4, "data": T_lidar_cam.flatten().tolist()}}, f)
    print("Saved extrinsic to", out_yml)

    # Generate overlay using the first used pair
    stem, img_path, csv_path = used[0]
    img = cv2.imread(str(img_path))
    pts_full = load_lidar_points_from_csv(csv_path)
    T_board_lidar, plane_pts = fit_plane_pose(pts_full,
                                              dist_thresh=args.plane_thresh,
                                              max_pca_pts=args.max_pca_pts)
    board_pts = filter_points_on_board(
        plane_pts, T_board_lidar,
        board_w, board_h,
        z_tol=args.z_tol, margin=args.margin
    )
    if board_pts.shape[0] < 20:
        board_pts = plane_pts

    pts_xy, depths = project_points(board_pts, T_lidar_cam, K, D, img.shape, use_distortion=True)
    if pts_xy.shape[0] == 0:
        pts_xy, depths = project_points(board_pts, T_lidar_cam, K, D, img.shape, use_distortion=False)
    if pts_xy.shape[0] == 0:
        # fallback full cloud
        pts_xy, depths = project_points(pts_full, T_lidar_cam, K, D, img.shape, use_distortion=True)
        if pts_xy.shape[0] == 0:
            pts_xy, depths = project_points(pts_full, T_lidar_cam, K, D, img.shape, use_distortion=False)

    overlay = draw_points_depth(img.copy(), pts_xy, depths, radius=5)
    out_overlay = out_dir / f"{Path(args.intrinsics).stem.replace('_intrinsics','')}_overlay.png"
    cv2.imwrite(str(out_overlay), overlay)
    print("Wrote overlay →", out_overlay)

    mask = save_mask(img.shape[0], img.shape[1], pts_xy, radius=3)
    out_mask = out_dir / f"{Path(args.intrinsics).stem.replace('_intrinsics','')}_mask.png"
    cv2.imwrite(str(out_mask), mask)

    cv2.imshow("overlay", overlay)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()

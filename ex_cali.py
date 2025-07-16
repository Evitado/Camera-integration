#!/usr/bin/env python3
"""
LiDAR ↔ camera extrinsic calibration (Ouster OS-1 + checkerboard)

• Reads matched_camX.csv, camera intrinsics, LiDAR scan CSVs
• Crops LiDAR by depth, fits checkerboard plane, resolves 8 orientation variants
• Accumulates 3-D ↔ 2-D correspondences, solves global PnP
• Writes camX_extrinsics.yaml  (R, t, K, dist)

pip install opencv-python open3d numpy pyyaml pandas
"""

from pathlib import Path
from typing  import Tuple, List

import cv2, numpy as np, open3d as o3d, yaml, pandas as pd

# ─────────── CONFIG ────────────────────────────────────────────────────
ROOT          = Path("/Users/parthmehta/Downloads/mybox-selected")
CAMERAS       = [0, 1, 2, 3]
PATTERN_SIZE  = (9, 6)            # interior corners  (cols, rows)
SQUARE_MM     = 30.0              # printed square size
NEAR, FAR     = 0.40, 2.50        # depth crop [m] around board
PLANE_THRESH  = 0.010             # plane RANSAC threshold [m]
MIN_INLIERS   = 120               # discard planes with fewer inliers
MAX_FRAMES    = 30                # per-cam processing cap
SAVE_DIR      = ROOT / "extrinsics"
# ───────────────────────────────────────────────────────────────────────


# ---------- helpers ----------------------------------------------------
def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    K   = (d["camera_matrix"] if "camera_matrix" in d else d["K"]).reshape(3,3)
    dist = (d["dist_coeff"]   if "dist_coeff"   in d else d["dist"]).reshape(-1,1)
    return K.astype(np.float64), dist.astype(np.float64)

def chessboard_object_points() -> np.ndarray:
    w, h = PATTERN_SIZE
    objp = np.zeros((w*h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:w, 0:h].T.reshape(-1,2)
    objp *= SQUARE_MM / 1000.0                          # mm→m
    objp[:,0] -= (w-1)*(SQUARE_MM/1000)/2
    objp[:,1] -= (h-1)*(SQUARE_MM/1000)/2
    return objp

def detect_chessboard(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    ok, corners = cv2.findChessboardCorners(img, PATTERN_SIZE,
                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    cv2.cornerSubPix(img, corners, (11,11), (-1,-1),
                     (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return corners.reshape(-1,2)

def load_pointcloud(csv_path: Path):
    # columns: TIMESTAMP, X, Y, Z, RANGE, ...
    data = np.loadtxt(csv_path, delimiter=",", usecols=(1,2,3,4))
    xyz   = data[:, :3]                 # metres
    rng   = data[:,  3] * 1e-3          # mm → m
    return xyz, rng

def pca_basis(pts: np.ndarray) -> np.ndarray:
    cov = np.cov((pts - pts.mean(axis=0)).T)
    _, eigvecs = np.linalg.eigh(cov)    # col-wise, small→large var
    return eigvecs

def lidar_corners_in_world(
        centroid: np.ndarray, R_plane: np.ndarray,
        K: np.ndarray, dist: np.ndarray, img_corners: np.ndarray
    ) -> np.ndarray:
    objp = chessboard_object_points()
    ROT_Z = [
        np.eye(3),
        np.array([[0,-1,0],[1,0,0],[0,0,1]]),
        np.array([[-1,0,0],[0,-1,0],[0,0,1]]),
        np.array([[0,1,0],[-1,0,0],[0,0,1]])
    ]
    best_err, best_pts = np.inf, None
    for z_sign in (+1,-1):
        Rn = R_plane.copy(); Rn[:,2] *= z_sign
        for A in ROT_Z:
            Rv = Rn @ A
            pts = (Rv @ objp.T).T + centroid
            ok, rvec, tvec = cv2.solvePnP(
                    pts.astype(np.float32), img_corners.astype(np.float32),
                    K, dist, flags=cv2.SOLVEPNP_EPNP)
            if not ok: continue
            proj,_ = cv2.projectPoints(pts, rvec, tvec, K, dist)
            err = np.mean(np.linalg.norm(img_corners - proj.squeeze(), axis=1))
            if err < best_err: best_err, best_pts = err, pts
    if best_pts is None:
        raise RuntimeError("board orientation unresolved")
    return best_pts

def save_yaml(path: Path, K, dist, rvec, tvec):
    R,_ = cv2.Rodrigues(rvec)
    data = dict(camera_matrix=K.tolist(),
                dist_coeff=dist.flatten().tolist(),
                rotation_matrix=R.tolist(),
                translation_vec=tvec.flatten().tolist())
    with open(path,"w") as fh: yaml.safe_dump(data, fh)

# ---------- main routine per cam ---------------------------------------
def process_camera(cam:int)->None:
    print(f"\n— Cam {cam} —")

    paired = ROOT / f"matched/matched_cam{cam}.csv"
    intr   = ROOT / f"intrinsics_cam{cam}.npz"
    if not paired.exists(): print("  ⚠ no matched CSV"); return

    K, dist = load_intrinsics(intr)
    pairs   = pd.read_csv(paired)

    img_pts, obj_pts = [], []
    good = 0

    for _, row in pairs.iterrows():
        img_path  = ROOT / f"intrinsics/cam{cam}" / row.filename
        lidar_csv = ROOT / f"cam{cam}_csv" / row.lidar_csv

        corners = detect_chessboard(img_path)
        if corners is None: continue

        xyz, rng = load_pointcloud(lidar_csv)
        mask = (rng > NEAR) & (rng < FAR)
        if good < 3:      # only for a few first frames
            dmin, dmax = rng[mask].min(), rng[mask].max()
            print(f"    depth window kept {mask.sum()} pts "
                f"(min {dmin:.2f} m, max {dmax:.2f} m)")
        if mask.sum() < 2000: continue
        P = xyz[mask]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))

        plane, inl = pc.segment_plane(distance_threshold=PLANE_THRESH,
                                      ransac_n=3, num_iterations=2000)
        if len(inl) < MIN_INLIERS: continue
        board_pts = P[inl]
        centroid  = board_pts.mean(axis=0)
        R_plane   = pca_basis(board_pts)

        lidar_c = lidar_corners_in_world(centroid, R_plane, K, dist, corners)
        if lidar_c.shape[0] != corners.shape[0]: continue

        img_pts.append(corners); obj_pts.append(lidar_c)
        good += 1
        print(f"  + {row.filename}   inliers={len(inl)}")
        if good >= MAX_FRAMES: break

    if good < 5: print("  ⚠ too few frames"); return

    img_all = np.vstack(img_pts).astype(np.float32)
    obj_all = np.vstack(obj_pts).astype(np.float32)
    ok,rvec,tvec,_ = cv2.solvePnPRansac(
        obj_all, img_all, K, dist, flags=cv2.SOLVEPNP_EPNP)
    if not ok: print("  ❌ PnP failed"); return
    rvec,tvec = cv2.solvePnPRefineLM(obj_all,img_all,K,dist,rvec,tvec)

    SAVE_DIR.mkdir(exist_ok=True)
    save_yaml(SAVE_DIR/f"cam{cam}_extrinsics.yaml",K,dist,rvec,tvec)
    print(f"  ✅ saved cam{cam}_extrinsics.yaml   (frames used: {good})")

# ---------- driver ------------------------------------------------------
def main():
    for cam in CAMERAS:
        process_camera(cam)

if __name__ == "__main__":
    main()

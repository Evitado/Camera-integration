#!/usr/bin/env python3
"""
Visualise LiDAR → camera projection and report reprojection error.

Dependencies
────────────
pip install opencv-python open3d numpy pyyaml matplotlib
"""

from pathlib import Path
import yaml, cv2, open3d as o3d, numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ───────── CONFIG ────────────────────────────────────────────────────────
ROOT        = Path("/Users/parthmehta/Downloads/mybox-selected")
CAM         = 0                            # 0,1,2,3
IMG_FILE    = "frame_04.png"               # from matched_camX.csv
LIDAR_FILE  = "cam0_scan_252.csv"          # matching scan
CALIB_SIZE  = (960, 540)                   # (width, height) used in intrinsics
PATTERN_SZ  = (9, 6)                       # checkerboard interior corners
SQUARE_MM   = 30.0                         # printed square size (mm)
# ─────────────────────────────────────────────────────────────────────────


# ---------- helper functions --------------------------------------------
def chessboard_object_points():
    w, h = PATTERN_SZ
    objp = np.zeros((w * h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
    objp *= SQUARE_MM / 1000.0                 # mm → metres
    objp[:, 0] -= (w - 1) * (SQUARE_MM / 1000) / 2
    objp[:, 1] -= (h - 1) * (SQUARE_MM / 1000) / 2
    return objp

def load_yaml(path: Path):
    with open(path) as fh:
        return yaml.safe_load(fh)

def load_pointcloud(csv_path: Path):
    pts = np.loadtxt(csv_path, delimiter=",", usecols=(1,2,3))
    return pts

def scale_K(K, img_shape):
    """Return K scaled to current image resolution."""
    h, w = img_shape[:2]
    sw, sh = w / CALIB_SIZE[0], h / CALIB_SIZE[1]
    S = np.array([[sw, 0,  0],
                  [0,  sh, 0],
                  [0,  0,  1]])
    return S @ K

def project_points(pts_lidar, R, t, K, dist):
    pts_cam = (R @ pts_lidar.T + t).T
    z = pts_cam[:, 2]
    mask = z > 0.1                # keep points in front (>10 cm)
    pts_cam = pts_cam[mask]
    z = z[mask]
    uv, _ = cv2.projectPoints(pts_cam, np.zeros((3,1)), np.zeros((3,1)),
                              K, dist)
    return uv.reshape(-1, 2), z

# ---------- main ---------------------------------------------------------
def main():
    # paths
    yaml_path = ROOT / "extrinsics" / f"cam{CAM}_extrinsics.yaml"
    img_path  = ROOT / f"intrinsics/cam{CAM}" / IMG_FILE
    pc_path   = ROOT / f"cam{CAM}_csv" / LIDAR_FILE

    # load calib
    c = load_yaml(yaml_path)
    K0   = np.array(c["camera_matrix"], dtype=np.float64)
    dist = np.array(c["dist_coeff"], dtype=np.float64).flatten()
    R    = np.array(c["rotation_matrix"], dtype=np.float64)
    t    = np.array(c["translation_vec"], dtype=np.float64).reshape(3,1)

    # load image & undistort
    img_raw = cv2.imread(str(img_path))
    K = scale_K(K0, img_raw.shape)          # adapt intrinsics
    img_ud = cv2.undistort(img_raw, K, dist, None, K)

    # load point-cloud
    pts = load_pointcloud(pc_path)

    # project
    uv, depth = project_points(pts, R, t, K, np.zeros_like(dist))

    # colour-map depth
    cmap = cm.get_cmap("turbo")
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    colours = (cmap(d_norm)[:,:3] * 255).astype(np.uint8)

    for (u,v), col in zip(uv.astype(int), colours):
        if 0 <= u < img_ud.shape[1] and 0 <= v < img_ud.shape[0]:
            cv2.circle(img_ud, (u,v), 2, col.tolist(), -1)

    # ── RMS reprojection error on chessboard ——————————————
    gray = cv2.cvtColor(img_ud, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN_SZ,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ok:
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        objp = chessboard_object_points()
        objp_lidar = (R @ objp.T + t).T
        proj, _ = cv2.projectPoints(objp_lidar, np.zeros((3,1)), np.zeros((3,1)), K, np.zeros_like(dist))
        err = np.linalg.norm(corners.squeeze() - proj.squeeze(), axis=1)
        print(f"RMS reprojection error: {err.mean():.2f} px  (N={len(err)})")
        for p in proj.astype(int).reshape(-1,2):
            cv2.drawMarker(img_ud, p, (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)

    # -------- display / save ----------
    cv2.imshow(f"LiDAR → cam{CAM}", img_ud)
    cv2.waitKey(0)
    # cv2.imwrite(str(ROOT / f"overlay_cam{CAM}.png"), img_ud)

if __name__ == "__main__":
    main()

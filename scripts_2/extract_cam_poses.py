# extract_cam_poses.py
"""
Detect checkerboard in each image, estimate board->camera pose, save as JSON.
Usage:
    python extract_cam_poses.py calib_imgs_cam0
"""

import cv2, glob, numpy as np, json, sys, pathlib, tqdm

def main(folder):
    # --- Board config (match what you used for intrinsics) ---
    squaresX, squaresY = 7, 5
    square_size = 0.0376  # meters
    pattern_size = (squaresX - 1, squaresY - 1)  # (6,4)

    # Load intrinsics
    yaml_path = f"calib/{folder}_intrinsics.yaml"
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    K = fs.getNode("camera_matrix").mat()
    D = fs.getNode("dist_coeff").mat()
    fs.release()

    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2) * square_size

    poses = []
    img_files = sorted(glob.glob(f"{folder}/*.png"))  # your files have timestamps in name
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH |
             cv2.CALIB_CB_NORMALIZE_IMAGE |
             cv2.CALIB_CB_FILTER_QUADS)

    for fn in tqdm.tqdm(img_files, desc=folder):
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
        if not found:
            continue

        # refine
        cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                         (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Solve PnP (board coords → camera)
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok: 
            continue
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = tvec.ravel()

        poses.append({
            "image": pathlib.Path(fn).name,
            "timestamp": float(pathlib.Path(fn).stem),  # your filenames are epoch seconds
            "T_board_cam": T.tolist()
        })

    out = pathlib.Path("calib") / f"{folder}_board_poses.json"
    json.dump(poses, open(out, "w"), indent=2)
    print(f"✔ Saved {len(poses)} poses → {out}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_cam_poses.py <image_folder>")
    main(sys.argv[1])

# intrinsic_to_yaml.py
"""
Compute intrinsics from a plain checkerboard displayed full‑screen on your iPad Pro.

Usage:
    python intrinsic_to_yaml.py <image_folder>

Adjust the BOARD CONFIG below for your printed/displayed board:
  - squaresX, squaresY: total squares horizontally / vertically
  - square_size: edge length of each square in meters
"""

import cv2, glob, numpy as np, sys, pathlib

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python intrinsic_to_yaml.py <image_folder>")
    folder = pathlib.Path(sys.argv[1])
    imgs = glob.glob(str(folder / "*.png"))
    if not imgs:
        sys.exit(f"✗ No PNG images found in {folder}")

    # ─── BOARD CONFIG (edit these) ──────────────────────────────
    squaresX, squaresY = 7, 5       # e.g. 8×6 squares on the board
    square_size = 0.036             # meters (≈screen_width_mm/8 / 1000)
    # ─────────────────────────────────────────────────────────────

    pattern_size = (squaresX - 1, squaresY - 1)  # inner corners
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],
                          0:pattern_size[1]].T.reshape(-1,2) * square_size

    obj_points, img_points = [], []
    img_shape = None
    ann_dir = folder.parent / f"{folder.name}_annotated"
    ann_dir.mkdir(exist_ok=True)

    print(f"⏳ Calibrating from {len(imgs)} images in {folder}")
    for fn in imgs:
        img = cv2.imread(fn)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_shape is None:
            img_shape = gray.shape[::-1]  # (width, height)

        # find the chessboard corners
        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH
          | cv2.CALIB_CB_NORMALIZE_IMAGE
          | cv2.CALIB_CB_FAST_CHECK
        )
        if not found:
            print(f"⚠ Chessboard not found in {fn}")
            continue

        vis = img.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)
        else:
            cv2.putText(vis, "NOT FOUND", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imshow("debug", vis)
        if cv2.waitKey(30) & 0xFF == 27:   # press Esc to quit early
            break
        vis_name = ann_dir / f"annotated_{pathlib.Path(fn).name}"
        cv2.imwrite(str(vis_name), vis)

        print(f"✔ Saved {vis_name.name} with {len(corners)} corners")

        # refine corner locations
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                    30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1), criteria
        )

        obj_points.append(objp)
        img_points.append(corners_refined)
        print(f"✔ Detected {len(corners_refined)} corners in {pathlib.Path(fn).name}")

    if len(obj_points) < 5:
        sys.exit(f"✗ Only {len(obj_points)} valid images; need at least 5")

    # calibrate
    ret, K, D, _, _ = cv2.calibrateCamera(
        obj_points, img_points, img_shape, None, None
    )
    print(f"► RMS reprojection error: {ret:.3f} px")

    # write YAML
    calib_dir = pathlib.Path("calib")
    calib_dir.mkdir(exist_ok=True)
    out_path = calib_dir / f"{folder.name}_intrinsics.yaml"
    fs = cv2.FileStorage(str(out_path), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", K)
    fs.write("dist_coeff", D)
    fs.release()
    print(f"✔ Wrote {out_path}")
    print(f"✔ Annotated images in → {ann_dir}")

if __name__ == "__main__":
    main()

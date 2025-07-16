"""
camera_in_cali.py

Performs intrinsic calibration for a single USB camera using a set of checkerboard images.
- Detects checkerboard corners from calibration images
- Computes the camera matrix and distortion coefficients
- Saves the result as a `.npz` file for later use in undistortion and projection

Checkerboard should have known dimensions (e.g., 9x6 with 30mm squares).
Draws and optionally displays corners to visually confirm detection.
"""
import cv2
import numpy as np
import glob
import os

# ────────────────────────────────────────────────
# Checkerboard Configuration
# ────────────────────────────────────────────────

# Checkerboard dimensions (number of inner corners)
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 30  # mm

# ────────────────────────────────────────────────
# Input image directory
# ────────────────────────────────────────────────

# Update this path to point to the folder with captured images
image_dir = r"C:\Users\mehta\OneDrive\Desktop\Parth - Evitado\extrinsics\cam1"
images = glob.glob(os.path.join(image_dir, '*.png'))

# ────────────────────────────────────────────────
# Prepare 3D reference object points
# (same for all images)
# ────────────────────────────────────────────────

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE # Scale to real-world units (mm)

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Output directory for corner-drawn images
os.makedirs("intrinsics/cam1/corners", exist_ok=True)

# ────────────────────────────────────────────────
# Detect checkerboard corners in all images
# ────────────────────────────────────────────────

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    print(f"{fname}: Chessboard detected? {ret}")
    if ret:
        objpoints.append(objp)
        # Refine corner locations
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and save checkerboard detection image
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        output_path = os.path.join("intrinsics/cam1/corners", os.path.basename(fname))
        cv2.imwrite(output_path, img)

        # Optional: Show for visual check
        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()


# ────────────────────────────────────────────────
# Perform calibration if enough valid images
# ────────────────────────────────────────────────

if len(objpoints) == 0:
    print("❌ No valid checkerboard images found. Calibration aborted.")
else:
    # Calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("\n✅ Calibration Successful!")
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist.ravel())
    # Save calibration results as npz
    np.savez("intrinsics_cam1_test.npz", K=K, dist=dist)

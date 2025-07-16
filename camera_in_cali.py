import cv2
import numpy as np
import glob
import os

# Checkerboard dimensions (number of inner corners)
CHECKERBOARD = (9, 6)
SQUARE_SIZE = 30  # mm

# Set correct image directory
image_dir = r"C:\Users\mehta\OneDrive\Desktop\Parth - Evitado\extrinsics\cam1"
images = glob.glob(os.path.join(image_dir, '*.png'))

# Prepare object points (e.g., (0,0,0), (30,0,0), ..., (240,150,0))
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

os.makedirs("intrinsics/cam1/corners", exist_ok=True)

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    print(f"{fname}: Chessboard detected? {ret}")
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and save the image
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        output_path = os.path.join("intrinsics/cam1/corners", os.path.basename(fname))
        cv2.imwrite(output_path, img)

        # Optional: Show for visual check
        cv2.imshow("Corners", img)
        cv2.waitKey(100)

cv2.destroyAllWindows()


if len(objpoints) == 0:
    print("❌ No valid checkerboard images found. Calibration aborted.")
else:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    print("\n✅ Calibration Successful!")
    print("Camera matrix:\n", K)
    print("Distortion coefficients:\n", dist.ravel())

    np.savez("intrinsics_cam1_test.npz", K=K, dist=dist)

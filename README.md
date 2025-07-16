# Camera-integration
This repository documents my internship work involving multi-camera and LiDAR sensor fusion using 4x AR0234 USB cameras and the Ouster OS1 LiDAR. It includes scripts for data capturing, calibration, synchronization, and visualization.


---

## Camera Setup

### `camera/camera_view.py`
Displays a 2x2 live feed grid of 4 AR0234 USB cameras using OpenCV.

### `camera/camera_capture.py`
Capture images for intrinsic calibration for a selected camera (`cv2.VideoCapture(index)`).

### `camera/camera_in_cali.py`
Performs intrinsic calibration using a checkerboard pattern and saves camera matrix and distortion coefficients (`.npz` format).

### `camera/timestamp.py`
Extracts UNIX and ISO timestamps from saved images and stores in CSV for synchronization.

---

## Synchronization and Calibration

### `calibration/synchronize.py`
Matches camera image timestamps with LiDAR scan timestamps to generate frame-scan pairs.

### `calibration/ex_cali.py`
Extrinsic calibration of each camera relative to LiDAR using:
- Intrinsics (`.npz`)
- Matched timestamps
- LiDAR point cloud CSVs
- Checkerboard detection
- PnP solver to get extrinsic `.yaml` file

### `calibration/visualize.py`
Projects LiDAR points onto undistorted camera images using extrinsic and intrinsic parameters. Computes reprojection error to validate calibration.

---

## Data

Data required:
- Calibration images, corners, and `.npz`
- LiDAR scan CSV files
- Matched Ouster frame-image synchronized pairs as CSV files
- `.yaml` files from `ex_cali.py`

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt

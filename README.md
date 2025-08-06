# Camera-integration
This repository documents my internship work involving multi-camera and LiDAR sensor fusion using 4x AR0234 USB cameras and the Ouster OS1 LiDAR. It includes scripts for data capturing, calibration, synchronization, and visualization. 

Note: Initial attempts using OpenCV and ROS2 on both WSL and Windows encountered issues. I later pivoted to MATLAB and completed the calibration and synchronization using MATLAB's Lidar Camera Calibrator toolbox.

---
## Initial Approach (scripts/)
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

## Second Approach (scripts_2/)

## Enhanced Data Capture Scripts
### scripts_2/capture_cam.py
Advanced camera capture script with live preview and automatic timing control:
- Captures N still frames from specified camera index
- Shows live preview with countdown overlay for board alignment
- Saves timestamped images as <timestamp>.png
- Configurable capture intervals and frame counts
- Example: python capture_cam.py --idx 2 --outdir calib_imgs_cam2 --n 25

### scripts_2/capture_lidar.py
Legacy-style LiDAR recording using ouster.sdk.SensorPacketSource:
- Records LiDAR data to PCAP format with metadata JSON
- Configurable UDP ports for LiDAR and IMU data
- Time-limited recording with descriptive filenames
- Saves sensor metadata for later processing
- Example: python capture_lidar.py --ip 192.168.1.1 --outdir pcap


## Camera Intrinsic Calibration
### scripts_2/intrinsic_to_yaml.py
Robust intrinsic calibration from checkerboard patterns:
- Supports both printed and iPad-displayed checkerboards
- Configurable board geometry (squares and size)
- Automatic corner detection with subpixel refinement
- Saves annotated images showing detected corners
- Outputs OpenCV-compatible YAML format
- Calculates RMS reprojection error for quality assessment

### scripts_2/extract_cam_poses.py
Extracts board-to-camera poses for extrinsic calibration:
- Detects checkerboard corners in calibration images
- Solves Perspective-n-Point (PnP) problem for each image
- Saves 4x4 transformation matrices as JSON
- Links poses to image timestamps for synchronization
- Essential input for extrinsic calibration pipeline


## LiDAR Data Processing (tried different approaches but ran into issues)
### scripts_2/extract_lidar_planes_o3d.py
Extracts planar surfaces from LiDAR PCAP using Open3D (>= 0.19.0):
- RANSAC-based plane segmentation per frame
- Configurable distance threshold and minimum points
- Exports plane normal vectors and centroids
- Saves results with frame timestamps for matching

### scripts_2/extract_lidar_planes_osdk.py
Alternative plane extraction using ouster-sdk 0.15.x:
- Packet-based scan processing with automatic metadata detection
- XYZ point cloud conversion with NaN filtering
- Robust plane fitting with error handling
- Compatible with legacy Ouster SDK versions

### scripts_2/extract_lidar_planes_packets.py
Advanced packet-level processing with filtering:
- Range and signal filtering for better plane detection
- Support for vertical plane detection (iPad boards)
- Debug point cloud exports for verification
- Batch processing of complete scans
- Optimized for checkerboard plane detection


## Extrinsic Calibration Pipeline
### scripts_2/ex_cali.py
Core library for LiDAR-to-Camera extrinsic calibration:
- CSV loader with automatic column detection (x,y,z with unit conversion)
- Open3D RANSAC plane fitting with PCA-based orientation
- Checkerboard region cropping in board coordinates
- Hand-eye calibration solver (AX = XB problem)
- Robust point projection with distortion handling
- Depth-colored overlay generation for visualization

### scripts_2/csv_lidar_cam_calib_final.py
Complete extrinsic calibration pipeline from CSV data:
- Integrates camera intrinsics, board poses, and LiDAR CSVs
- Multi-frame robust calibration with outlier rejection
- Automatic axis-flip resolution for board frame ambiguity
- Generates calibration quality metrics and visualizations
Outputs:
<cam>_lidar_extrinsics.yaml: Transformation matrix
<cam>_overlay.png: Depth-colored projection overlay
<cam>_mask.png: Binary projection mask

### scripts_2/run_calibration.py
CLI wrapper for the complete calibration workflow:
- Single command calibration for any camera
- Configurable checkerboard parameters
- Hand-eye solver with pose averaging
- Automatic fallback strategies for poor data
- Comprehensive output generation and validation

### scripts_2/verify_poses.py
Quality assurance tool for board pose validation:
- Checks rotation matrix orthonormality
- Verifies proper determinant values
- Analyzes translation vector magnitudes
- Identifies problematic pose estimates


## MATLAB Integration (scripts_2/matlab/)
After initial difficulties with OpenCV and ROS2 implementations, MATLAB's Lidar Camera Calibrator provided the most robust solution.
### scripts_2/matlab/prep_cam.m
Prepares data for MATLAB Lidar Camera Calibrator:
- Converts OpenCV YAML intrinsics to MATLAB format
- Extracts point clouds from PCAP files matched to image count
- Creates organized folder structure for calibrator input
- Handles camera parameter conversion (focal length, distortion)
- Generates count-matched PCD files for temporal alignment

### scripts_2/matlab/mat2yaml.m
Converts MATLAB calibration results to standard formats:
- Extracts 4x4 transformation matrices from .mat files
- Handles both numeric matrices and rigidtform3d objects
- Exports to OpenCV-compatible YAML format
- Preserves precision for downstream applications

## Calibration Results (matlab_results/)
### matlab_results/cam0_tform.yaml
First calibration attempt for Camera 0:
- Contains 4x4 transformation matrix T_lidar_to_cam0
- Translation: [0.013, 0.150, 1.133] meters
- Rotation matrix with proper orthonormality

### matlab_results/cam0_tform_2.yaml
Refined calibration for Camera 0:
- Updated transformation T_lidar_to_cam0_2
- Improved translation: [0.023, 0.212, 1.112] meters
- Shows iterative calibration refinement process

## Data Requirements
### Input Data:
- Camera Images: Timestamped PNG files with checkerboard patterns
- Intrinsic Calibration: OpenCV YAML files with camera matrix and distortion coefficients
- Board Poses: JSON files with 4x4 transformation matrices from PnP solver
### LiDAR Data:
- PCAP files with sensor metadata JSON
- Exported CSV files with x,y,z point coordinates
- Synchronized frame-image pairs

### Output Data:
- Extrinsic Matrices: YAML files with LiDAR-to-camera transformations
- Visualization: Overlay images showing projected LiDAR points
- Validation: Binary masks and reprojection error metrics
- MATLAB Results: .mat files and converted YAML outputs


## Workflow Summary
1. Data Capture: Use capture_cam.py and capture_lidar.py for synchronized data collection
2. Intrinsic Calibration: Run intrinsic_to_yaml.py on checkerboard images
3. Pose Extraction: Use extract_cam_poses.py to get board-camera relationships
4. LiDAR Processing: Extract planes with extract_lidar_planes_*.py scripts
5. Extrinsic Calibration:
  a. Python approach: csv_lidar_cam_calib_final.py or run_calibration.py
  b. MATLAB approach: prep_cam.m → Lidar Camera Calibrator → mat2yaml.m
6. Validation: Use verify_poses.py and visualization outputs

## Requirements
### Install dependencies with:
scripts: pip install -r requirements.txt
scripts_2: pip install -r requirements_win.txt

### Key Dependencies:
- OpenCV for computer vision operations
- Open3D for 3D point cloud processing
- ouster-sdk for LiDAR data handling
- NumPy/SciPy for numerical computations
- pandas for CSV data processing
- PyYAML for configuration file handling

### MATLAB Requirements:
- MATLAB R2020b or later
- Computer Vision Toolbox
- Lidar Toolbox
- Navigation Toolbox (for rigidtform3d support)



# How to Use
## Prerequisites Setup

### Install Python dependencies:

pip install -r requirements.txt

### Create project directory structure:

project_root/
├── calib/                    # Output calibration files
├── pcap/                     # LiDAR PCAP files and metadata
├── calib_imgs_cam0/          # Camera 0 calibration images
├── calib_imgs_cam1/          # Camera 1 calibration images
├── csv_cam0/                 # Camera 0 synchronized CSV files
├── csv_cam1/                 # Camera 1 synchronized CSV files
└── scripts_2/                # All calibration scripts

### Step 1: Data Capture
#### Capture Camera Calibration Images
Capture 25 images from camera 0 with 2-second intervals
python scripts_2/capture_cam.py --idx 0 --outdir calib_imgs_cam0 --n 25 --pause 2.0

#### Repeat for other cameras
python scripts_2/capture_cam.py --idx 1 --outdir calib_imgs_cam1 --n 25 --pause 2.0
python scripts_2/capture_cam.py --idx 2 --outdir calib_imgs_cam2 --n 25 --pause 2.0
python scripts_2/capture_cam.py --idx 3 --outdir calib_imgs_cam3 --n 25 --pause 2.0

#### Capture LiDAR Data
Record LiDAR data (Ctrl+C to stop)
python scripts_2/capture_lidar.py --ip 192.168.1.1 --outdir pcap --n-seconds 60

### Step 2: Camera Intrinsic Calibration
Calibrate each camera's intrinsics
python scripts_2/intrinsic_to_yaml.py calib_imgs_cam0
python scripts_2/intrinsic_to_yaml.py calib_imgs_cam1
python scripts_2/intrinsic_to_yaml.py calib_imgs_cam2
python scripts_2/intrinsic_to_yaml.py calib_imgs_cam3
Output: calib/calib_imgs_cam0_intrinsics.yaml (and similar for other cameras)

### Step 3: Extract Board Poses
Extract board-to-camera poses for each camera
python scripts_2/extract_cam_poses.py calib_imgs_cam0
python scripts_2/extract_cam_poses.py calib_imgs_cam1
python scripts_2/extract_cam_poses.py calib_imgs_cam2
python scripts_2/extract_cam_poses.py calib_imgs_cam3
Output: calib/calib_imgs_cam0_board_poses.json (and similar for other cameras)

### Step 4: Export LiDAR Data to CSV
#### Method A: Using Ouster Studio (Recommended)

1. Open Ouster Studio
2. Load your PCAP file (pcap/sensor_data.pcap)
3. Export synchronized frames as CSV files
4. Save to csv_cam0/, csv_cam1/, etc. folders
5. Ensure CSV filenames match image timestamps

#### Method B: Using Python Scripts
Extract planes from PCAP (alternative approach)
python scripts_2/extract_lidar_planes_packets.py pcap/sensor_data.pcap --dist 0.02 --min_points 200

### Step 5: Extrinsic Calibration
#### Option A: Python Pipeline (Experimental)
Run complete calibration for camera 0
python scripts_2/csv_lidar_cam_calib_final.py \
  --intrinsics "calib/calib_imgs_cam0_intrinsics.yaml" \
  --board_json "calib/calib_imgs_cam0_board_poses.json" \
  --img_dir "calib_imgs_cam0" \
  --csv_dir "csv_cam0" \
  --out_dir "calib"

#### Option B: MATLAB Pipeline (Recommended)
##### Step 5B.1: Prepare Data for MATLAB
matlab% In MATLAB, navigate to scripts_2/matlab/ and run:
prep_cam(0)  % Prepare camera 0 data
prep_cam(1)  % Prepare camera 1 data
prep_cam(2)  % Prepare camera 2 data
prep_cam(3)  % Prepare camera 3 data

##### Step 5B.2: Use MATLAB Lidar Camera Calibrator

Open MATLAB and type: lidarCameraCalibrator
Load the prepared data:
Images: Select folder out/cam0/ (contains images)
Point Clouds: Select folder out/cam0/pointclouds/
Intrinsics: Load out/cam0/intrinsics/cam_intrinsics.mat

Follow the calibrator GUI:
1. Detect checkerboards in images
2. Segment planes in point clouds
3. Run calibration algorithm
4. Export calibration results to cam0_tform.mat

##### Step 5B.3: Convert MATLAB Results
matlab% Convert MATLAB results to YAML format
mat2yaml('cam0_tform.mat', 'cam0_tform.yaml', 'T_lidar_to_cam0')

### Step 6: Validation and Quality Check
Verify pose quality
python scripts_2/verify_poses.py calib/calib_imgs_cam0_board_poses.json

Check calibration results (if using Python pipeline)
View generated overlay images: calib/cam0_overlay.png
Check projection masks: calib/cam0_mask.png

### Step 7: Integration and Testing
Load calibration results in your sensor fusion application
Test projection accuracy using the overlay visualizations
Validate temporal synchronization between sensors
Refine calibration if reprojection errors are high

## Troubleshooting Tips
### Poor Checkerboard Detection
Ensure high contrast between squares
Use iPad display for consistent illumination
Capture images at various angles and distances
Check that squares_x and squares_y parameters match your board

### LiDAR Plane Segmentation Issues
Adjust --dist parameter (try 0.01-0.03)
Increase --min_points for noisy environments
Use --rng_min and --rng_max to filter distance ranges
Try --vertical_only flag for vertical checkerboards

### Calibration Quality Issues
Increase number of calibration image pairs
Ensure good spatial distribution of board poses
Check temporal synchronization between sensors
Validate intrinsic calibration quality first

### MATLAB vs Python Trade-offs
MATLAB: More robust, better GUI, proven algorithms
Python: Open source, more customizable, better for automation
Start with MATLAB for initial results, then adapt to Python if needed

## Example Complete Workflow
#### 1. Capture data (assuming setup is ready)
python scripts_2/capture_cam.py --idx 0 --outdir calib_imgs_cam0 --n 30
python scripts_2/capture_lidar.py --ip 192.168.1.1 --outdir pcap

#### 2. Process camera data
python scripts_2/intrinsic_to_yaml.py calib_imgs_cam0
python scripts_2/extract_cam_poses.py calib_imgs_cam0

#### 3. Export LiDAR CSVs using Ouster Studio
(Manual step - export synchronized frames)

#### 4. Run calibration
python scripts_2/csv_lidar_cam_calib_final.py \
  --intrinsics "calib/calib_imgs_cam0_intrinsics.yaml" \
  --board_json "calib/calib_imgs_cam0_board_poses.json" \
  --img_dir "calib_imgs_cam0" \
  --csv_dir "csv_cam0" \
  --out_dir "calib"

#### 5. Validate results
python scripts_2/verify_poses.py calib/calib_imgs_cam0_board_poses.json

## Notes
- The MATLAB approach proved most reliable for final extrinsic calibration
- CSV export from Ouster Studio provides the most consistent LiDAR data format
- Checkerboard detection works better with high-contrast patterns (iPad display recommended)
- Multiple calibration attempts help validate transformation accuracy
- Temporal synchronization is critical for accurate multi-sensor calibration



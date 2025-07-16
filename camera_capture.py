"""
camera_capture.py

Captures still images from a single USB camera (e.g., AR0234) for intrinsic calibration.
Allows the user to manually save images of a checkerboard by pressing the 'c' key.

Images are saved in a specified folder for each camera.
Press 'q' to quit the capture session.
"""

import cv2
import os

# ───────────────────────────────────────
# Configuration: Camera and output folder
# ───────────────────────────────────────

# Set the index of the camera to capture from (0, 1, 2, or 3)
cap = cv2.VideoCapture(1)  # Change index per camera

# Set the folder where images will be saved
output_dir = "intrinsics/cam1"
os.makedirs(output_dir, exist_ok=True)

# Image file counter
count = 0

# ───────────────────────────────────────
# Live capture loop
# ───────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break # Exit loop if frame is not captured

    # Display the live feed in a window
    cv2.imshow("Capture", frame)
    # Wait for user key input
    key = cv2.waitKey(1)

    # Press 'c' to capture and save the current frame
    if key == ord('c'):  # Press 'c' to capture
        filename = os.path.join(output_dir, f"frame_{count:02d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1
        
    # Press 'q' to quit the capture loop
    elif key == ord('q'):
        break
        
# ───────────────────────────────────────
# Cleanup resources
# ───────────────────────────────────────

cap.release()
cv2.destroyAllWindows()

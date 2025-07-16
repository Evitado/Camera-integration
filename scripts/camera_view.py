"""
camera_view.py

Streams live video feed from 4 AR0234 USB cameras and displays them in a 2x2 grid using OpenCV.
Each feed is resized and concatenated into a single window for real-time simultaneous visualization.

Press 'q' to quit the viewer.
"""

import cv2
import numpy as np

# ───────────────────────────────
# Camera configuration
# ───────────────────────────────

# List of USB camera device indices (varies based on system config)
camera_ids = [0, 1, 2, 3]

# Initialize VideoCapture for each camera
caps = [cv2.VideoCapture(i) for i in camera_ids]

# Set resolution to 1280x720 (can be changed)
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ───────────────────────────────
# Live streaming loop
# ───────────────────────────────
while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        # If frame capture fails, insert a blank image with "No Signal"
        if not ret:
            frame = cv2.putText(
                np.zeros((480, 640, 3), dtype=np.uint8),
                "No Signal", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        frames.append(frame)

    # Resize and stack 2x2 grid
    frames = [cv2.resize(f, (640, 360)) for f in frames]
    # Concatenate top and bottom rows to form 2x2 grid
    top = cv2.hconcat(frames[:2])
    bottom = cv2.hconcat(frames[2:])
    grid = cv2.vconcat([top, bottom])
    
    # Display the 2x2 camera feed window
    cv2.imshow("4x AR0234 Cameras", grid)
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ───────────────────────────────
# Cleanup
# ───────────────────────────────
for cap in caps:
    cap.release()
cv2.destroyAllWindows()

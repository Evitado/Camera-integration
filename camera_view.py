import cv2
import numpy as np

# List of USB camera indices – update if necessary
camera_ids = [0, 1, 2, 3]
caps = [cv2.VideoCapture(i) for i in camera_ids]

# Set resolution if needed
for cap in caps:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    frames = []
    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frame = cv2.putText(
                np.zeros((480, 640, 3), dtype=np.uint8),
                "No Signal", (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )
        frames.append(frame)

    # Resize and stack 2x2 grid
    frames = [cv2.resize(f, (640, 360)) for f in frames]
    top = cv2.hconcat(frames[:2])
    bottom = cv2.hconcat(frames[2:])
    grid = cv2.vconcat([top, bottom])

    cv2.imshow("4x AR0234 Cameras", grid)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()

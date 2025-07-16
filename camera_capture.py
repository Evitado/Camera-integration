import cv2
import os

cap = cv2.VideoCapture(1)  # Change index per camera

output_dir = "intrinsics/cam1"
os.makedirs(output_dir, exist_ok=True)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)

    if key == ord('c'):  # Press 'c' to capture
        filename = os.path.join(output_dir, f"frame_{count:02d}.png")
        cv2.imwrite(filename, frame)
        print(f"Saved {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

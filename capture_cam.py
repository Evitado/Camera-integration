"""
Grab N still frames from the specified camera index and save them as
<timestamp>.png in the desired output folder, while showing a live
preview so you can align the board.

Example:
    python capture_cam.py --idx 2 --outdir calib_imgs_cam2 --n 25
"""

import cv2, time, pathlib, argparse, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idx", type=int, required=True,
                    help="Camera index for cv2.VideoCapture")
    ap.add_argument("--outdir", required=True,
                    help="Folder to store PNGs (created if missing)")
    ap.add_argument("--n", type=int, default=25,
                    help="Number of frames to grab (default 25)")
    ap.add_argument("--pause", type=float, default=2.0,
                    help="Seconds to wait between auto‑captures")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"✗ Could not open camera {args.idx}")

    print(f"◆ Live preview started for cam{args.idx}. Auto‑capturing {args.n} frames every {args.pause}s into {out}")

    count = 0
    last_capture = time.time()

    while count < args.n:
        ok, frame = cap.read()
        if not ok:
            print("⚠ Frame grab failed, retrying...")
            time.sleep(0.1)
            continue

        # Show live preview with countdown overlay
        now = time.time()
        dt = now - last_capture
        secs_to_next = max(0, args.pause - dt)
        text = f"Frame {count+1}/{args.n} in {secs_to_next:.1f}s"
        cv2.putText(frame, text, (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow(f"Camera {args.idx} Preview", frame)

        # Auto‑capture
        if dt >= args.pause:
            ts = now
            fname = out / f"{ts:.6f}.png"
            cv2.imwrite(str(fname), frame)
            print(f"✔ Captured {fname.name}")
            count += 1
            last_capture = now

        # Handle keypress: ESC to quit early
        if cv2.waitKey(1) & 0xFF == 27:
            print("✗ Interrupted by user")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Done")

if __name__ == "__main__":
    main()

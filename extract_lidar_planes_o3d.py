# extract_lidar_planes_o3d.py
"""
Extract a plane (normal + centroid) per LiDAR 'frame' in an Ouster PCAP using Open3D.
Works with Open3D >= 0.19.0

Usage:
    python extract_lidar_planes_o3d.py pcap/board_round.pcap pcap/board_round.json
"""

import argparse, json
import numpy as np
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", help="Path to LiDAR pcap")
    ap.add_argument("meta", help="Path to sensor metadata json")
    ap.add_argument("--dist", type=float, default=0.01, 
                    help="RANSAC distance threshold (m) [default 0.01]")
    ap.add_argument("--min_points", type=int, default=2000,
                    help="Skip frames with fewer points than this")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Limit number of frames (0 = all)")
    args = ap.parse_args()

    # Open3D Ouster reader
    reader = o3d.io.OusterPcap(args.pcap, args.meta)
    planes = []
    count = 0

    while True:
        frame = reader.read_next_frame()
        if frame is None:
            break
        count += 1
        if args.max_frames and count > args.max_frames:
            break

        pts = np.asarray(frame.points)     # Nx3
        if pts.shape[0] < args.min_points:
            continue

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=args.dist, ransac_n=3, num_iterations=2000
            )
        except RuntimeError:
            continue

        n = plane_model[:3]
        plane_pts = pts[inliers]
        centroid = plane_pts.mean(axis=0)

        # frame.timestamps is Nx1 ns timestamps. Take median as this frame's ts.
        ts = int(np.median(frame.timestamps)) if hasattr(frame, "timestamps") else count

        planes.append({
            "timestamp": ts,
            "normal": n.tolist(),
            "centroid": centroid.tolist()
        })

    out = Path("calib/lidar_planes.json")
    json.dump(planes, open(out, "w"), indent=2)
    print(f"✔ Saved {len(planes)} planes → {out}")

if __name__ == "__main__":
    main()
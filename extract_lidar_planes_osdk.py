# extract_lidar_planes_osdk.py
"""
Extract a plane (normal + centroid) per LiDAR scan in a PCAP using
ouster-sdk 0.15.x open_source() + Open3D RANSAC.

Assumes the PCAP's metadata JSON is next to it with the same base name
(e.g., cam0.pcap + cam0.json).

Usage:
    python extract_lidar_planes_osdk.py pcap/cam0.pcap
"""

import argparse, json, numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

from ouster.sdk import open_source
# Deprecated but still present in 0.15.x
from ouster.sdk.client import Scans
# New core namespace for XYZLut
from ouster.sdk.core import XYZLut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", help="Path to LiDAR pcap file")
    ap.add_argument("--dist", type=float, default=0.01,
                    help="RANSAC plane distance threshold (m)")
    ap.add_argument("--min_points", type=int, default=1000,
                    help="Skip scans with fewer than this many valid points")
    ap.add_argument("--max_scans", type=int, default=0,
                    help="Process at most this many scans (0 = all)")
    args = ap.parse_args()

    pcap_abs = Path(args.pcap).resolve().as_posix()

    # Only pass the PCAP; SDK will locate the .json sidecar automatically
    source = open_source(pcap_abs, collate=False)

    # Get the correct SensorInfo object (first sensor if multi-sensor source)
    if hasattr(source, "sensor_info"):
        info = source.sensor_info[0]
    else:
        # Fallback for very old builds; 'metadata' returns a single SensorInfo
        info = source.metadata

    # Create XYZ LUT
    xyz_lut = XYZLut(info, use_extrinsics=False)

    planes = []
    scans_iter = Scans(source)
    if args.max_scans > 0:
        scans_iter = (s for i, s in enumerate(scans_iter) if i < args.max_scans)

    for scan in tqdm(scans_iter, desc="PCAP scans"):
        xyz = xyz_lut(scan)                    # (H*W, 3)
        pts = xyz[~np.isnan(xyz).any(axis=1)]  # drop NaNs
        if pts.shape[0] < args.min_points:
            continue

        # Plane fit
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=args.dist, ransac_n=3, num_iterations=2000
            )
        except RuntimeError:
            continue

        n = plane_model[:3]
        centroid = pts[inliers].mean(axis=0)

        # Pick timestamp attribute that exists
        ts = getattr(scan, "timestamp", 0)
        if ts == 0:
            ts = getattr(scan, "start_ts", getattr(scan, "end_ts", 0))

        planes.append({
            "timestamp": int(ts),
            "normal": n.tolist(),
            "centroid": centroid.tolist()
        })

    Path("calib").mkdir(exist_ok=True)
    out = Path("calib/lidar_planes.json")
    json.dump(planes, open(out, "w"), indent=2)
    print(f"✔ Saved {len(planes)} planes → {out}")

if __name__ == "__main__":
    main()

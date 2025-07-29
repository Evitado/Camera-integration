# extract_lidar_planes_packets.py
"""
Extract one plane (normal + centroid) per scan from an Ouster PCAP using
the packet→batch approach (works with ouster-sdk 0.15.x). Adds filters for
distance and signal so the iPad plane is detectable.

Usage:
    python extract_lidar_planes_packets.py pcap/cam0.pcap ^
        --dist 0.02 --min_points 200 --rng_min 0.3 --rng_max 1.5 --sig_min 50 ^
        --dump_debug 3
"""

import argparse, json, numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

from ouster.sdk import open_source
from ouster.sdk.client import LidarPacket, ScanBatcher, ChanField
from ouster.sdk.core import XYZLut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", help="Path to LiDAR pcap (json with same basename beside it)")
    ap.add_argument("--dist", type=float, default=0.01,
                    help="RANSAC distance threshold (m)")
    ap.add_argument("--min_points", type=int, default=500,
                    help="Minimum #points to attempt plane fit (after filtering)")
    ap.add_argument("--max_scans", type=int, default=0,
                    help="Process at most this many complete scans (0=all)")
    ap.add_argument("--rng_min", type=float, default=0.2,
                    help="Min range (meters) to keep points")
    ap.add_argument("--rng_max", type=float, default=2.0,
                    help="Max range (meters) to keep points")
    ap.add_argument("--sig_min", type=int, default=0,
                    help="Min SIGNAL value to keep points (0 disables)")
    ap.add_argument("--vertical_only", action="store_true",
                    help="Keep only planes whose normal z-component is small (approx vertical)")
    ap.add_argument("--dump_debug", type=int, default=0,
                    help="Dump first N debug PCDs (filtered cloud & inliers)")
    args = ap.parse_args()

    pcap_path = Path(args.pcap).resolve().as_posix()
    src = open_source(pcap_path, collate=False)

    # SensorInfo
    if hasattr(src, "sensor_info"):
        info = src.sensor_info[0]
    else:
        info = src.metadata

    xyz_lut   = XYZLut(info, use_extrinsics=False)
    batcher   = ScanBatcher(info)
    planes    = []

    dbg_dir = Path("debug_scans")
    if args.dump_debug > 0:
        dbg_dir.mkdir(exist_ok=True)

    scan_count = 0
    dumped = 0

    # iterate packets, batch into scans
    for pkt in tqdm(src, desc="Packets"):
        if not isinstance(pkt, LidarPacket):
            continue
        ls = batcher(pkt)
        if ls is None:
            continue  # not a full scan yet

        scan_count += 1
        if args.max_scans and scan_count > args.max_scans:
            break

        xyz = xyz_lut(ls)                        # (H*W, 3)
        rng = ls.field(ChanField.RANGE).reshape(-1) * info.range_unit
        sig = ls.field(ChanField.SIGNAL).reshape(-1)

        # Filter points (range & signal)
        m = (rng > args.rng_min) & (rng < args.rng_max)
        if args.sig_min > 0:
            m &= (sig >= args.sig_min)
        pts = xyz[m]
        if pts.shape[0] < args.min_points:
            continue

        # Fit plane
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=args.dist, ransac_n=3, num_iterations=2000)
        except RuntimeError:
            continue

        n = plane_model[:3]
        centroid = pts[inliers].mean(axis=0)

        # optional orientation gate (vertical plane: |n_z| small)
        if args.vertical_only and abs(n[2]) > 0.3:
            # skip planes tilted too much wrt vertical
            continue

        # Choose a timestamp
        ts = getattr(ls, "timestamp", None)
        if ts is None:
            ts = getattr(ls, "start_ts", getattr(ls, "end_ts", scan_count))
        ts = int(ts)

        planes.append({
            "timestamp": ts,
            "normal": n.tolist(),
            "centroid": centroid.tolist()
        })

        # Debug dumps
        if dumped < args.dump_debug:
            full_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            inlier_pcd = full_pcd.select_by_index(inliers)
            o3d.io.write_point_cloud(str(dbg_dir / f"scan_{scan_count:04d}_full.pcd"), full_pcd)
            o3d.io.write_point_cloud(str(dbg_dir / f"scan_{scan_count:04d}_plane.pcd"), inlier_pcd)
            dumped += 1
        print(f"scan {scan_count}: kept {pts.shape[0]} pts after filters "
        f"(rng {args.rng_min}-{args.rng_max}, sig >= {args.sig_min})")

    Path("calib").mkdir(exist_ok=True)
    out = Path("calib/lidar_planes.json")
    json.dump(planes, open(out, "w"), indent=2)
    print(f"✔ Saved {len(planes)} planes → {out}")


if __name__ == "__main__":
    main()

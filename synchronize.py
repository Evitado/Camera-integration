#!/usr/bin/env python3
"""
Synchronise camera images with Ouster OS-1 scans (all four cams).

Output
──────
matched_camX.csv  →  filename, unix_timestamp, lidar_csv, lidar_unix, delta_sec
"""

from __future__ import annotations
import csv, os, sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


# ── CONFIG ────────────────────────────────────────────────────────────────
ROOT_DIR          = Path("/Users/parthmehta/Downloads/mybox-selected")
CAMERAS           = [0, 1, 2, 3]         # cams to process
OUTPUT_DIR        = ROOT_DIR / "matched"
TOLERANCE_SEC     = 0.050                # max Δt allowed when matching

# If you logged the exact wall-clock boot time of the LiDAR, enter it here
# (same for every camera session).  Otherwise leave as None and the script
# will *estimate* it independently for each camera using the first frame.
LIDAR_BOOT_UNIX   = None
# ───────────────────────────────────────────────────────────────────────────


def load_scan_timestamps(csv_path: Path) -> np.ndarray:
    """Return all distinct non-zero TIMESTAMP ns values from one scan file."""
    stamps = []
    with csv_path.open(errors="ignore") as fh:
        rdr = csv.reader(fh)
        for row in rdr:
            if not row or row[0].startswith("#"):
                continue
            try:
                ns = int(row[0])
            except ValueError:
                continue
            if ns:
                stamps.append(ns)
    return np.unique(stamps)


def rep_ts_ns(stamps_ns: np.ndarray) -> int:
    """Representative time of a scan (median is robust)."""
    return int(np.median(stamps_ns))


def estimate_boot(first_img_unix: float, first_scan_ns: int) -> float:
    """boot ≈ earliest_cam_time  −  earliest_lidar_ns/1e9"""
    return first_img_unix - first_scan_ns / 1e9


def build_lidar_timeline(scan_files: List[Path], boot_unix: float
                         ) -> pd.DataFrame:
    """DataFrame: scan_csv | lidar_unix (one row per scan)."""
    rows = []
    for f in sorted(scan_files):
        ns = rep_ts_ns(load_scan_timestamps(f))
        rows.append((f.name, boot_unix + ns / 1e9))
    return pd.DataFrame(rows, columns=["lidar_csv", "lidar_unix"])


def sync_one_camera(cam: int) -> None:
    print(f"\n— Cam {cam} —")
    img_ts_path  = ROOT_DIR / "intrinsics" / f"cam{cam}" \
                               / f"cam{cam}_image_timestamps.csv"
    scan_dir     = ROOT_DIR / f"cam{cam}_csv"
    out_path     = OUTPUT_DIR / f"matched_cam{cam}.csv"

    if not img_ts_path.exists():
        print(f"⚠️  No image-timestamp CSV: {img_ts_path}")
        return
    if not scan_dir.exists():
        print(f"⚠️  No LiDAR CSV dir: {scan_dir}")
        return

    img_df = pd.read_csv(img_ts_path).sort_values("unix_timestamp")
    img_df["unix_timestamp"] = img_df["unix_timestamp"].astype("float64")
    scan_files = list(scan_dir.glob("cam*_scan_*.csv"))
    if not scan_files:
        print(f"⚠️  No LiDAR scans found for cam{cam}")
        return

    first_scan_ns   = rep_ts_ns(load_scan_timestamps(scan_files[0]))
    boot_unix = (LIDAR_BOOT_UNIX if LIDAR_BOOT_UNIX is not None
                 else estimate_boot(img_df["unix_timestamp"].min(),
                                     first_scan_ns))
    print(f"   LiDAR boot time (unix): {boot_unix:.6f}")

    lidar_df = build_lidar_timeline(scan_files, boot_unix) \
                   .sort_values("lidar_unix")
    lidar_df["lidar_unix"] = lidar_df["lidar_unix"].astype("float64")

    matched = pd.merge_asof(img_df, lidar_df,
                            left_on="unix_timestamp",
                            right_on="lidar_unix",
                            direction="nearest",
                            tolerance=TOLERANCE_SEC)

    matched["delta_sec"] = (matched["unix_timestamp"] - matched["lidar_unix"]
                             ).abs()

    OUTPUT_DIR.mkdir(exist_ok=True)
    matched.to_csv(out_path, index=False)
    print(f"   ✅  {len(matched)} rows →  {out_path.name}")


def main() -> None:
    for cam in CAMERAS:
        sync_one_camera(cam)


if __name__ == "__main__":
    main()

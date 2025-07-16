"""
timestamp.py

Extracts and saves the last modified timestamps of all image files in a specified folder.
- Converts timestamps to both ISO format and UNIX epoch time
- Saves output as a CSV with one row per image

This is useful for synchronizing image capture times with LiDAR data during calibration.
"""

import os
import csv
from datetime import datetime

# ─────────────────────────────────────────────
# Configuration: Path to image folder
# ─────────────────────────────────────────────

# Update this to point to the folder containing captured images
folder = r"C:\Users\mehta\OneDrive\Desktop\Parth - Evitado\intrinsics\cam3"
output_csv = os.path.join(folder, "cam3_image_timestamps.csv")

# File types to include in timestamp extraction
image_exts = (".png", ".jpg", ".jpeg", ".bmp")

# ─────────────────────────────────────────────
# Collect timestamps for all images
# ─────────────────────────────────────────────

rows = []
for fname in os.listdir(folder):
    if fname.lower().endswith(image_exts):
        fpath = os.path.join(folder, fname)
        # Get last modified time of the file
        modified_time = os.path.getmtime(fpath)
        iso_time = datetime.fromtimestamp(modified_time).isoformat()
        # Append filename, ISO time, and UNIX timestamp
        rows.append([fname, iso_time, int(modified_time)])

# Sort by timestamp if desired
rows.sort(key=lambda x: x[2]) # Sort by UNIX timestamp

# ─────────────────────────────────────────────
# Write results to CSV file
# ─────────────────────────────────────────────

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "modified_time_iso", "unix_timestamp"])
    writer.writerows(rows)

print(f"✅ Done! Timestamps saved to: {output_csv}")

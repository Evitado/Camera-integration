import os
import csv
from datetime import datetime

# Use a raw string (r"...") to prevent backslash escape issues
folder = r"C:\Users\mehta\OneDrive\Desktop\Parth - Evitado\intrinsics\cam3"
output_csv = os.path.join(folder, "cam3_image_timestamps.csv")

# Supported image extensions
image_exts = (".png", ".jpg", ".jpeg", ".bmp")

# Collect timestamps
rows = []
for fname in os.listdir(folder):
    if fname.lower().endswith(image_exts):
        fpath = os.path.join(folder, fname)
        modified_time = os.path.getmtime(fpath)  # <- Modified time
        iso_time = datetime.fromtimestamp(modified_time).isoformat()
        rows.append([fname, iso_time, int(modified_time)])

# Sort by timestamp if desired
rows.sort(key=lambda x: x[2])

# Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "modified_time_iso", "unix_timestamp"])
    writer.writerows(rows)

print(f"✅ Done! Timestamps saved to: {output_csv}")

# capture_lidar_legacy.py
"""
Legacy-style recording using ouster.sdk.SensorPacketSource.

Installs needed: pip install ouster-sdk more-itertools
Run:
    python capture_lidar_legacy.py --ip 192.168.1.1 --outdir pcap
"""

import argparse
import pathlib
from contextlib import closing
from datetime import datetime
from more_itertools import time_limited

from ouster.sdk import sensor
import ouster.sdk.pcap as pcap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", "--hostname", dest="hostname", default="192.168.1.1",
                    help="Sensor hostname or IP")
    ap.add_argument("--lidar-port", type=int, default=7502,
                    help="LiDAR UDP port (default 7502)")
    ap.add_argument("--imu-port",   type=int, default=7503,
                    help="IMU UDP port (default 7503)")
    ap.add_argument("--outdir", default="pcap",
                    help="Directory to save .pcap and .json (will be created)")
    # You can set n_seconds to a huge number so it effectively runs until Ctrl‑C
    ap.add_argument("--n-seconds", type=int, default=10**9,
                    help="Max seconds to record; default is very large (Ctrl-C to stop)")

    args = ap.parse_args()
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Open the sensor packet source
    with closing(sensor.SensorPacketSource(
            args.hostname,
            lidar_port=args.lidar_port,
            imu_port=args.imu_port,
            buffer_time_sec=1.0
        )) as source:

        # make a descriptive filename base
        time_part = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta = source.sensor_info[0]
        fname_base = f"{meta.prod_line}_{meta.sn}_{meta.config.lidar_mode}_{time_part}"
        pcap_path = outdir / f"{fname_base}.pcap"
        json_path = outdir / f"{fname_base}.json"

        # save metadata
        print(f"Saving sensor metadata → {json_path}")
        json_path.write_text(meta.to_json_string())

        # record packets
        print(f"Writing packets → {pcap_path} (Ctrl-C to stop)")
        # time_limited will stop after n_seconds, but default is huge
        source_it = time_limited(args.n_seconds, source)

        def to_packet():
            for idx, packet in source_it:
                yield packet

        n_packets = pcap.record(to_packet(), str(pcap_path))
        print(f"Captured {n_packets:,} packets")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""V4L2 D405 depth capture for PetaLinux boards (VMK180 / vp1502).

The pip pyrealsense2 wheel uses the V4L2 backend, which on stock PetaLinux
kernels dies at VIDIOC_QBUF because the kernel has no RealSense *metadata*
node support (Ubuntu/Jetson ship a patched kernel; PetaLinux does not).
But the plain depth video node streams fine via V4L2 — and Monty only needs
depth pixels (metadata is just timestamps). So this reads /dev/video1 (Z16)
directly, bypassing librealsense entirely.

Depth scale is the D405 constant 9.999e-05 m/unit (= 0.1mm/unit), confirmed
identical on Z8 and VMK180. raw uint16 * scale -> meters.

This is a standalone capture test; the numbers it prints (valid %, distance)
tell you the depth is usable before wiring it into the Monty pipeline.

Usage (on the board):
  sudo python3 v4l2_d405_test.py                 # 480x270 depth, 5 frames
  sudo python3 v4l2_d405_test.py --node /dev/video1 --width 480 --height 270
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEPTH_SCALE = 9.999999747378752e-05  # D405 m/unit (raw uint16 -> meters)
SATURATION = 65535                   # max Z16 value = invalid/too-far


def capture_raw(node, w, h, n, out):
    """Capture n Z16 frames to a raw file via v4l2-ctl."""
    cmd = [
        "v4l2-ctl", "-d", node,
        f"--set-fmt-video=width={w},height={h},pixelformat=Z16 ",
        "--stream-mmap", f"--stream-count={n}", f"--stream-to={out}",
    ]
    subprocess.run(cmd, check=True)


def to_meters(raw):
    m = raw.astype(np.float32) * DEPTH_SCALE
    m[raw == 0] = np.nan
    m[raw == SATURATION] = np.nan
    return m


def main():
    p = argparse.ArgumentParser(description="V4L2 D405 depth test (no librealsense)")
    p.add_argument("--node", default="/dev/video1", help="Z16 depth video node")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--out", default="/tmp/d405_depth.raw")
    args = p.parse_args()

    t0 = time.time()
    capture_raw(args.node, args.width, args.height, args.frames, args.out)
    dt = time.time() - t0

    frame_bytes = args.width * args.height * 2
    data = np.fromfile(args.out, np.uint16)
    got = data.size // (args.width * args.height)
    print(f"captured {got} frame(s) of {args.width}x{args.height} Z16 "
          f"in {dt:.2f}s ({frame_bytes} bytes/frame)")

    # analyse the last frame
    last = data[-(args.width * args.height):].reshape(args.height, args.width)
    m = to_meters(last)
    v = m[~np.isnan(m)]
    if v.size:
        cy, cx = args.height // 2, args.width // 2
        rh, rw = args.height // 6, args.width // 6
        centre = m[cy - rh:cy + rh, cx - rw:cx + rw]
        cv = centre[~np.isnan(centre)]
        print(f"  valid {100 * np.mean(~np.isnan(m)):.0f}%   "
              f"range {v.min():.3f}-{v.max():.3f}m   median {np.median(v):.3f}m")
        if cv.size:
            print(f"  centre ROI: {np.median(cv):.3f}m "
                  f"(put an object ~0.2-0.3m here)")
    else:
        print("  all invalid (no object / all saturated)")


if __name__ == "__main__":
    main()

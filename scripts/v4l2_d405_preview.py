#!/usr/bin/env python3
"""V4L2 D405 RGBD capture + MJPEG web preview for PetaLinux boards.

The board is headless and its pyrealsense2 dies at VIDIOC_QBUF (no kernel
metadata node support), so this reads the D405 depth (Z16) and colour (YUYV)
video nodes directly and serves an RGB | depth preview over HTTP — open
http://<board-ip>:8080 in a browser on another machine (same idea as the OAK
FPGA path in rgbd_capture_oak_fpga.py).

Streams are held open via v4l2py (linuxpy) and read continuously, which is
~20x faster than re-spawning v4l2-ctl per frame (measured on VMK180: 0.8 fps
subprocess -> depth 27 fps / depth+colour 16 fps streamed). Depth is the
D405 constant 9.999e-05 m/unit; colour is YUYV -> BGR.

Nodes on this D405/PetaLinux setup: video1 = Z16 depth, video5 = YUYV colour
(video0 is Multiplanar and fails single-plane capture, video3 = GREY IR).

Usage (on the board):
  sudo python3 v4l2_d405_preview.py
  sudo python3 v4l2_d405_preview.py --no-color            # depth only
  sudo python3 v4l2_d405_preview.py --port 8080
  # then browse to http://<board-ip>:8080
"""
import argparse
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

import numpy as np

warnings.filterwarnings("ignore")  # silence the v4l2py-deprecation notice

from v4l2py import Device
from v4l2py.device import VideoCapture

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

DEPTH_SCALE = 9.999999747378752e-05  # D405 m/unit
SATURATION = 65535
W, H = 640, 480


# --------------------------------------------------------------------------
# Streaming capture (held-open V4L2 streams, read continuously)
# --------------------------------------------------------------------------

def open_stream(node, pixfmt):
    dev = Device(node)
    dev.open()
    cap = VideoCapture(dev)
    cap.set_format(W, H, pixfmt)
    cap.open()
    return dev, cap, iter(cap)


def depth_from(frame):
    raw = np.frombuffer(bytes(frame), np.uint16)
    if raw.size < W * H:
        return None
    raw = raw[:W * H].reshape(H, W)
    m = raw.astype(np.float32) * DEPTH_SCALE
    m[(raw == 0) | (raw == SATURATION)] = np.nan
    return m


def color_from(frame):
    raw = np.frombuffer(bytes(frame), np.uint8)
    if raw.size < W * H * 2:
        return None
    yuyv = raw[:W * H * 2].reshape(H, W, 2)
    if HAVE_CV2:
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    y = yuyv[:, :, 0].astype(np.float32)
    u = yuyv[:, 0::2, 1].repeat(2, axis=1).astype(np.float32) - 128
    v = yuyv[:, 1::2, 1].repeat(2, axis=1).astype(np.float32) - 128
    r = np.clip(y + 1.402 * v, 0, 255)
    g = np.clip(y - 0.344 * u - 0.714 * v, 0, 255)
    b = np.clip(y + 1.772 * u, 0, 255)
    return np.stack([b, g, r], axis=-1).astype(np.uint8)


def draw_overlay(frame, has_color, valid, dist, fps, max_m):
    def txt(img, s, org, color=(240, 240, 240), scale=0.6, thick=2):
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                    color, thick, cv2.LINE_AA)

    if has_color:
        txt(frame, "RGB", (10, 26))
        txt(frame, f"depth 0-{max_m:.1f}m", (W + 10, 26))
        cv2.drawMarker(frame, (W // 2, H // 2), (0, 255, 255),
                       cv2.MARKER_CROSS, 24, 2)
    else:
        txt(frame, f"depth 0-{max_m:.1f}m", (10, 26))

    dist_s = f"{dist:.2f}m" if dist == dist else "?"
    dcol = (120, 255, 120) if valid >= 60 else (120, 200, 255)
    txt(frame, f"depth valid {valid:.0f}%   center {dist_s}   {fps:.1f} fps",
        (10, frame.shape[0] - 12), color=dcol)


def depth_colormap(depth_m, max_m=1.0):
    valid = ~np.isnan(depth_m)
    norm = np.clip(np.where(valid, depth_m, max_m) / max_m, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if HAVE_CV2:
        cm = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        cm[~valid] = (40, 40, 40)
        return cm
    g = np.stack([u8] * 3, axis=-1)
    g[~valid] = (40, 40, 40)
    return g


# --------------------------------------------------------------------------
# MJPEG server
# --------------------------------------------------------------------------

class Latest:
    def __init__(self):
        self.jpg = None
        self.lock = threading.Lock()

    def set(self, b):
        with self.lock:
            self.jpg = b

    def get(self):
        with self.lock:
            return self.jpg


def make_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%'></body></html>"
                )
                return
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                j = latest.get()
                if j is not None:
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(j)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                time.sleep(0.03)
    return Handler


def main():
    p = argparse.ArgumentParser(description="V4L2 D405 RGBD MJPEG preview")
    p.add_argument("--depth-node", default="/dev/video1")
    p.add_argument("--color-node", default="/dev/video5")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument("--no-color", action="store_true")
    args = p.parse_args()

    if not HAVE_CV2:
        print("WARNING: cv2 not found — JPEG/colormap need it; install "
              "opencv-python-headless")

    latest = Latest()
    ThreadingTCPServer.allow_reuse_address = True
    srv = ThreadingTCPServer(("0.0.0.0", args.port), make_handler(latest))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"serving on http://0.0.0.0:{args.port}  (cv2={'yes' if HAVE_CV2 else 'no'})")

    ddev, dcap, diter = open_stream(args.depth_node, "Z16 ")
    if not args.no_color:
        cdev, ccap, citer = open_stream(args.color_node, "YUYV")
    print("streams open. Ctrl+C to stop.")

    n = 0
    t0 = time.time()
    fps = 0.0
    try:
        while True:
            depth = depth_from(next(diter))
            color = color_from(next(citer)) if not args.no_color else None
            if depth is None:
                continue

            valid = 100 * np.mean(~np.isnan(depth))
            cy, cx = H // 2, W // 2
            roi = depth[cy - 40:cy + 40, cx - 60:cx + 60]
            rv = roi[~np.isnan(roi)]
            dist = float(np.median(rv)) if rv.size else float("nan")

            dvis = depth_colormap(depth, args.max_distance)
            frame = np.hstack([color, dvis]) if color is not None else dvis

            if HAVE_CV2:
                draw_overlay(frame, color is not None, valid, dist, fps,
                             args.max_distance)
                ok, enc = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    latest.set(enc.tobytes())

            n += 1
            fps = n / (time.time() - t0)
            if n % 30 == 0:
                print(f"frame {n}  {fps:.1f} fps  valid {valid:.0f}%  "
                      f"center {dist:.3f}m")
    finally:
        dcap.close(); ddev.close()
        if not args.no_color:
            ccap.close(); cdev.close()


if __name__ == "__main__":
    main()

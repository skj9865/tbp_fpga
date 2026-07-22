#!/usr/bin/env python3
"""RGBD capture for OAK-D Pro on the Versal FPGA board (USB2, headless).

Self-contained (copy this one file to the board). Same output format as the
world-image dataset, so SaccadeOnImageEnvironment loads it unchanged:
  - rgb_{v}.png   : 640x480 RGBA uint8 (alpha=255)
  - depth_{v}.data: 640x480 float32, meters, NaN for invalid

FPGA/USB2 specifics (see memory/oak-fpga-usb2-sequential.md):
  * The board is USB2-only. Running ColorCamera ISP + StereoDepth together
    draws more than USB2's 900mA budget (esp. with the IR projector) and
    brown-out-crashes the device on bus power alone. Give the OAK its own
    power (Y-adapter) — then the simultaneous pipeline runs fine at 30fps.
  * No X/GTK on the board (cv2.imshow can't open a window), and its xlnxdrmfb
    rejects /dev/fb0 writes (DRM-KMS only). So live preview is streamed as
    MJPEG and watched in a browser on the Z8; keys are read from the launching
    terminal (termios).

Modes:
  (default)     MJPEG stream (view in Z8 browser) + 'c'/'q' keys (ext. power)
  --headless    auto-capture N frames, no preview               (ext. power)
  --sequential  no-power fallback: depth & color in separate single-stream
                sessions, paired per version (2 device boots per pair)
  --fb-preview  blit to /dev/fb0 (only if a board's fbdev accepts writes)

Usage:
  # Live: run on the board, open http://<board-ip>:8080/ in a Z8 browser,
  # press 'c' in this terminal to capture
  python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0

  # Scripted auto-capture (external power)
  python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0 \\
      --headless --num-captures 4

  # No external power available
  python3 rgbd_capture_oak_fpga.py --object numenta_mug --index 0 \\
      --sequential --fps 5 --num-captures 4
"""

import argparse
import http.server
import os
import select
import socketserver
import sys
import termios
import threading
import time
import tty
from datetime import timedelta
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from PIL import Image


TARGET_W, TARGET_H = 640, 480

# OAK-D Pro RGB at native 640x480 preview has HFOV ~63.75 deg. The Monty
# environment (two_d_data.py) must run with MONTY_HFOV=63.75 to match.
OAKD_HFOV_NATIVE = 63.75
CAPTURE_W, CAPTURE_H = TARGET_W, TARGET_H


# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------

def build_pipeline(preset="DENSITY", max_distance_m=None, min_distance_m=None,
                   fps=30):
    """Simultaneous ColorCamera + StereoDepth (aligned to RGB). One device.

    Requires external power on USB2 (both engines + IR exceed bus current).
    A Sync node groups color+depth by timestamp (depth is produced later than
    color; syncing keeps preview and each saved pair the same instant).
    Returns (pipeline, sync_output) — the queue yields a MessageGroup with
    "rgb" and "depth" messages.
    """
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(CAPTURE_W, CAPTURE_H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setFps(fps)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(fps)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(
        getattr(dai.node.StereoDepth.PresetMode, preset.upper())
    )
    print(f"Preset: {preset.upper()}")
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # align to RGB
    stereo.setOutputSize(CAPTURE_W, CAPTURE_H)
    _apply_threshold(stereo, max_distance_m, min_distance_m)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # Sync color + depth by timestamp -> emits a MessageGroup {rgb, depth}.
    sync = pipeline.create(dai.node.Sync)
    sync.setSyncThreshold(timedelta(milliseconds=max(10, int(500 / fps))))
    cam_rgb.preview.link(sync.inputs["rgb"])
    stereo.depth.link(sync.inputs["depth"])
    return pipeline, sync.out


def build_depth_pipeline(preset="DENSITY", max_distance_m=None,
                         min_distance_m=None, fps=5):
    """Depth-only pipeline (single stream). Used by --sequential.

    setDepthAlign(CAM_A) works from calibration alone (no color node needed),
    so the depth stays registered to the RGB frame.
    Returns (pipeline, depth_output).
    """
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(fps)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(
        getattr(dai.node.StereoDepth.PresetMode, preset.upper())
    )
    print(f"[depth] Preset: {preset.upper()}")
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(CAPTURE_W, CAPTURE_H)
    _apply_threshold(stereo, max_distance_m, min_distance_m)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    return pipeline, stereo.depth


def build_color_pipeline(fps=5):
    """Color-only pipeline (single stream). Used by --sequential (IR off)."""
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(CAPTURE_W, CAPTURE_H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setFps(fps)
    return pipeline, cam_rgb.preview


def _apply_threshold(stereo, max_distance_m, min_distance_m):
    """thresholdFilter = OAK viewer 'max distance' slider (clip far noise)."""
    if max_distance_m is None and min_distance_m is None:
        return
    tf = stereo.initialConfig.postProcessing.thresholdFilter
    if max_distance_m is not None:
        tf.maxRange = int(max_distance_m * 1000)
    if min_distance_m is not None:
        tf.minRange = int(min_distance_m * 1000)
    print(f"Threshold: min={tf.minRange}mm max={tf.maxRange}mm")


def set_ir_projector(device, dot_mA, flood_mA):
    """Set OAK-D Pro IR dot projector / flood brightness (mA). Best-effort."""
    dot_intensity = dot_mA / 1200.0
    flood_intensity = flood_mA / 1500.0
    try:
        if device is None:
            return
        if hasattr(device, "setIrLaserDotProjectorIntensity"):
            device.setIrLaserDotProjectorIntensity(dot_intensity)
            if flood_intensity > 0:
                device.setIrFloodLightIntensity(flood_intensity)
            print(f"IR (v3): dot={dot_intensity:.2f} flood={flood_intensity:.2f}")
        elif hasattr(device, "setIrLaserDotProjectorBrightness"):
            device.setIrLaserDotProjectorBrightness(dot_mA)
            if flood_mA > 0:
                device.setIrFloodLightBrightness(flood_mA)
            print(f"IR (v2): dot={dot_mA}mA flood={flood_mA}mA")
    except Exception as e:
        print(f"IR control failed: {e}")


# -----------------------------------------------------------------------------
# Frame conversion / IO
# -----------------------------------------------------------------------------

def depth_to_colormap(depth_mm, max_m=4.0):
    """uint16 mm depth -> BGR colormap for preview. Invalid (0) = red."""
    invalid = depth_mm == 0
    norm = np.clip(depth_mm.astype(np.float32) / (max_m * 1000.0), 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap[invalid] = [0, 0, 255]
    return cmap


def convert_rgb(bgr_frame):
    """BGR (TARGET size) -> RGBA uint8."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def convert_depth(depth_mm):
    """uint16 mm depth -> float32 meters, NaN for invalid (0)."""
    depth_f = depth_mm.astype(np.float32) / 1000.0
    depth_f[depth_mm == 0] = np.nan
    return depth_f


def save_capture(output_dir, version, rgba, depth_m):
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"rgb_{version}.png"
    Image.fromarray(rgba, mode="RGBA").save(str(rgb_path))
    depth_path = output_dir / f"depth_{version}.data"
    depth_m.tofile(str(depth_path))
    print(f"  Saved: {rgb_path.name} + {depth_path.name}")


def detect_start_version(output_dir):
    if not output_dir.exists():
        return 0
    versions = []
    for p in sorted(output_dir.glob("rgb_*.png")):
        try:
            versions.append(int(p.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return max(versions) + 1 if versions else 0


def frame_stats(depth_m):
    nan_pct = np.isnan(depth_m).sum() / depth_m.size * 100
    valid = depth_m[~np.isnan(depth_m)]
    rng = f"{valid.min():.3f}-{valid.max():.3f}m" if valid.size else "all NaN"
    return f"depth: {rng}, NaN={nan_pct:.1f}%"


def wait_frame(queue, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        f = queue.tryGet()
        if f is not None:
            return f
        time.sleep(0.01)
    return None


def drain_queue(queue):
    while queue.tryGet() is not None:
        pass


# -----------------------------------------------------------------------------
# Framebuffer preview + terminal key input (no X/GTK on the board)
# -----------------------------------------------------------------------------

class FramebufferPreview:
    """Blit BGR frames straight to the Linux framebuffer (HDMI console)."""

    def __init__(self, dev="/dev/fb0", swap_rb=False):
        base = "/sys/class/graphics/fb0"
        w, h = open(f"{base}/virtual_size").read().strip().split(",")
        self.vw, self.vh = int(w), int(h)
        self.bpp = int(open(f"{base}/bits_per_pixel").read())
        self.stride = int(open(f"{base}/stride").read())
        self.bytespp = self.bpp // 8
        self.swap_rb = swap_rb
        # Use plain write() (os.pwrite), not mmap: this board's fbdev
        # (DRM emulation) rejects mmap on /dev/fb0 (ENODEV). Each frame is
        # composed into an (h, stride) buffer and written once at offset 0.
        self.fd = os.open(dev, os.O_RDWR)

    def show(self, bgr):
        img = bgr[:, :, ::-1] if self.swap_rb else bgr
        h = min(img.shape[0], self.vh)
        w = min(img.shape[1], self.vw)
        img = np.ascontiguousarray(img[:h, :w])
        if self.bytespp == 4:
            img = np.dstack([img, np.full((h, w), 255, np.uint8)])
        row_bytes = w * self.bytespp
        rows = img.reshape(h, row_bytes)
        # Row-by-row: a single huge write is rejected (EINVAL) by this fbdev,
        # but per-line writes at their stride offset are the standard blit.
        for r in range(h):
            os.pwrite(self.fd, rows[r].tobytes(), r * self.stride)

    def clear(self):
        zero = bytes(self.stride)
        for r in range(self.vh):
            os.pwrite(self.fd, zero, r * self.stride)

    def close(self):
        try:
            os.close(self.fd)
        except Exception:
            pass


class TermKeys:
    """Non-blocking single-key reads from stdin (cbreak). No echo, no Enter."""

    def __enter__(self):
        self.enabled = sys.stdin.isatty()
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def get(self):
        if self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        if self.enabled:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


# -----------------------------------------------------------------------------
# MJPEG stream (view live in a browser on Z8; this board's fbdev is write-only-
# emulated / DRM-KMS and rejects /dev/fb0 writes, so no local preview)
# -----------------------------------------------------------------------------

class LatestFrame:
    """Thread-safe holder for the most recent JPEG bytes."""

    def __init__(self):
        self._jpeg = None
        self._lock = threading.Lock()

    def set(self, jpeg):
        with self._lock:
            self._jpeg = jpeg

    def get(self):
        with self._lock:
            return self._jpeg


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    latest = None  # set on the class before serving

    def log_message(self, *a):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path not in ("/", "/stream", "/stream.mjpg"):
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.end_headers()
        try:
            while True:
                jpeg = self.latest.get()
                if jpeg is not None:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass


def start_mjpeg_server(port, latest):
    _MJPEGHandler.latest = latest
    srv = socketserver.ThreadingTCPServer(("0.0.0.0", port), _MJPEGHandler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# -----------------------------------------------------------------------------
# Capture modes
# -----------------------------------------------------------------------------

def run_stream_preview(args, scene_dir, version):
    """Live preview over MJPEG (watch in a Z8 browser) + 'c'/'q' keys.

    The board's xlnxdrmfb rejects /dev/fb0 writes, so we stream instead.
    Needs external power (simultaneous color+depth).
    """
    pipeline, sync_out = build_pipeline(
        preset=args.preset, max_distance_m=args.max_distance,
        min_distance_m=args.min_distance, fps=args.fps,
    )
    vis_max_m = args.max_distance if args.max_distance is not None else 4.0
    q = sync_out.createOutputQueue()
    pipeline.start()
    set_ir_projector(pipeline.getDefaultDevice(),
                     args.ir_brightness, args.flood_brightness)

    latest = LatestFrame()
    srv = start_mjpeg_server(args.port, latest)
    print(f"MJPEG: open  http://<board-ip>:{args.port}/  in a browser on Z8")
    print("Keys in THIS terminal:  c = capture   q = quit")

    bgr = depth_mm = None
    done = 0
    try:
        with TermKeys() as keys:
            while pipeline.isRunning():
                grp = q.tryGet()
                if grp is not None:
                    bgr = grp["rgb"].getCvFrame()
                    depth_mm = grp["depth"].getFrame()
                if bgr is not None and depth_mm is not None:
                    preview = np.hstack([bgr, depth_to_colormap(depth_mm, vis_max_m)])
                    ok, enc = cv2.imencode(".jpg", preview,
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        latest.set(enc.tobytes())

                k = keys.get()
                if k == "q":
                    print(f"\nDone. {done} frame(s).")
                    break
                if k == "c":
                    if bgr is None or depth_mm is None:
                        print("  [no frames yet]")
                        continue
                    depth_m = convert_depth(depth_mm)
                    print(f"  [v={version}] {frame_stats(depth_m)}")
                    save_capture(scene_dir, version, convert_rgb(bgr), depth_m)
                    version += 1
                    done += 1
                time.sleep(0.005)
    finally:
        pipeline.stop()
        srv.shutdown()

def run_fb_preview(args, scene_dir, version):
    """Live preview on /dev/fb0 + 'c'/'q' from the terminal. Ext. power."""
    pipeline, sync_out = build_pipeline(
        preset=args.preset, max_distance_m=args.max_distance,
        min_distance_m=args.min_distance, fps=args.fps,
    )
    vis_max_m = args.max_distance if args.max_distance is not None else 4.0
    q = sync_out.createOutputQueue()
    pipeline.start()
    set_ir_projector(pipeline.getDefaultDevice(),
                     args.ir_brightness, args.flood_brightness)

    fb = FramebufferPreview(swap_rb=args.fb_rgb)
    print(f"FB: {fb.vw}x{fb.vh} {fb.bpp}bpp -> live preview on HDMI monitor")
    print("Keys in THIS terminal:  c = capture   q = quit")

    bgr = depth_mm = None
    done = 0
    try:
        with TermKeys() as keys:
            while pipeline.isRunning():
                grp = q.tryGet()
                if grp is not None:
                    bgr = grp["rgb"].getCvFrame()
                    depth_mm = grp["depth"].getFrame()
                if bgr is not None and depth_mm is not None:
                    preview = np.hstack([bgr, depth_to_colormap(depth_mm, vis_max_m)])
                    if args.fb_scale != 1.0:
                        preview = cv2.resize(
                            preview, None, fx=args.fb_scale, fy=args.fb_scale,
                            interpolation=cv2.INTER_NEAREST)
                    fb.show(preview)

                k = keys.get()
                if k == "q":
                    print(f"\nDone. {done} frame(s).")
                    break
                if k == "c":
                    if bgr is None or depth_mm is None:
                        print("  [no frames yet]")
                        continue
                    depth_m = convert_depth(depth_mm)
                    print(f"  [v={version}] {frame_stats(depth_m)}")
                    save_capture(scene_dir, version, convert_rgb(bgr), depth_m)
                    version += 1
                    done += 1
                time.sleep(0.005)
    finally:
        pipeline.stop()
        fb.close()


def run_headless(args, scene_dir, version):
    """Auto-capture N frames, no display. Ext. power."""
    pipeline, sync_out = build_pipeline(
        preset=args.preset, max_distance_m=args.max_distance,
        min_distance_m=args.min_distance, fps=args.fps,
    )
    q = sync_out.createOutputQueue()
    pipeline.start()
    set_ir_projector(pipeline.getDefaultDevice(),
                     args.ir_brightness, args.flood_brightness)
    print(f"Warming up ({args.warmup}s)...")
    time.sleep(args.warmup)
    drain_queue(q)

    saved = 0
    try:
        for i in range(args.num_captures):
            print(f"Capturing {i + 1}/{args.num_captures}...")
            bgr = depth_mm = None
            end = time.time() + 5
            while time.time() < end:
                grp = q.tryGet()
                if grp is not None:
                    bgr = grp["rgb"].getCvFrame()
                    depth_mm = grp["depth"].getFrame()
                    break
                time.sleep(0.01)
            if bgr is None or depth_mm is None:
                print(f"  [v={version}] timeout")
                continue
            depth_m = convert_depth(depth_mm)
            print(f"  [v={version}] {frame_stats(depth_m)}")
            save_capture(scene_dir, version, convert_rgb(bgr), depth_m)
            version += 1
            saved += 1
            if i < args.num_captures - 1:
                time.sleep(args.interval)
    finally:
        pipeline.stop()
    print(f"\nDone. {saved} frame(s).")


def _capture_depth_once(args):
    pipeline, depth_out = build_depth_pipeline(
        preset=args.preset, max_distance_m=args.max_distance,
        min_distance_m=args.min_distance, fps=args.fps,
    )
    q = depth_out.createOutputQueue()
    pipeline.start()
    set_ir_projector(pipeline.getDefaultDevice(),
                     args.ir_brightness, args.flood_brightness)
    time.sleep(args.warmup)
    drain_queue(q)
    f = wait_frame(q)
    pipeline.stop()
    return f.getFrame().copy() if f is not None else None


def _capture_color_once(args):
    pipeline, rgb_out = build_color_pipeline(fps=args.fps)
    q = rgb_out.createOutputQueue()
    pipeline.start()
    set_ir_projector(pipeline.getDefaultDevice(), 0, 0)
    time.sleep(args.warmup)
    drain_queue(q)
    f = wait_frame(q)
    pipeline.stop()
    return f.getCvFrame().copy() if f is not None else None


def run_sequential(args, scene_dir, version):
    """No-power fallback: depth + color in separate single-stream sessions.

    Each session boots its own device (reusing one across sessions segfaults
    depthai 3.7.1 in tryGetCalibration after stop). Object must stay still for
    one depth->color cycle; re-pose between pairs (Enter unless --no-prompt).
    """
    saved = 0
    for i in range(args.num_captures):
        if not args.no_prompt:
            try:
                resp = input(f"\n[pair {i + 1}/{args.num_captures}] position "
                             f"object, Enter to capture ('q'+Enter = stop): ")
            except EOFError:
                resp = ""
            if resp.strip().lower() == "q":
                break

        print(f"[pair {i + 1}] depth session...")
        depth_mm = _capture_depth_once(args)
        print("  >>> KEEP STILL — color session...")
        bgr = _capture_color_once(args)

        if depth_mm is None or bgr is None:
            missing = "depth" if depth_mm is None else "color"
            print(f"  [v={version}] skipped (missing {missing} frame)")
            continue
        depth_m = convert_depth(depth_mm)
        print(f"  [v={version}] {frame_stats(depth_m)}")
        save_capture(scene_dir, version, convert_rgb(bgr), depth_m)
        version += 1
        saved += 1
    print(f"\nDone. {saved} pair(s) captured (sequential).")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="OAK-D Pro RGBD capture on the Versal FPGA board (USB2)."
    )
    p.add_argument("--object", required=True, help="Object name (e.g. numenta_mug)")
    p.add_argument("--index", required=True, type=int, help="Scene index")
    p.add_argument("--output-dir",
                   default=str(Path.home() / "tbp/data/worldimages/captured_scenes_oak"),
                   help="Output base directory")
    p.add_argument("--start-version", type=int, default=None,
                   help="Start version number (default: auto-detect)")
    p.add_argument("--num-captures", type=int, default=4,
                   help="Frames/pairs to capture in headless/sequential")

    # Modes
    p.add_argument("--headless", action="store_true",
                   help="Auto-capture N frames, no preview (needs ext. power)")
    p.add_argument("--sequential", action="store_true",
                   help="No-power fallback: depth+color in separate sessions")
    p.add_argument("--no-prompt", action="store_true",
                   help="Sequential: don't wait for Enter between pairs")

    # Timing
    p.add_argument("--fps", type=int, default=30,
                   help="Camera fps (30 with ext. power; ~5 for --sequential)")
    p.add_argument("--warmup", type=float, default=2.0,
                   help="Seconds to settle before auto/sequential capture")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Headless: seconds between captures")

    # Depth / IR
    p.add_argument("--preset", default="DENSITY",
                   help="StereoDepth preset (DEFAULT/ACCURACY/DENSITY/...)")
    p.add_argument("--max-distance", type=float, default=None,
                   help="Clip depth beyond this many meters")
    p.add_argument("--min-distance", type=float, default=None,
                   help="Clip depth closer than this many meters")
    p.add_argument("--ir-brightness", type=int, default=1200,
                   help="IR dot projector mA (0-1200, 0=off)")
    p.add_argument("--flood-brightness", type=int, default=0,
                   help="IR flood mA (0-1500, 0=off)")

    # Live preview: MJPEG stream (default; view in a Z8 browser) — this board's
    # xlnxdrmfb rejects /dev/fb0 writes, so there is no working local preview.
    p.add_argument("--port", type=int, default=8080,
                   help="MJPEG stream port (default 8080)")
    # Framebuffer preview (only if a board's fbdev accepts writes; not this one)
    p.add_argument("--fb-preview", action="store_true",
                   help="Blit preview to /dev/fb0 instead of streaming "
                        "(only works if the fbdev accepts writes)")
    p.add_argument("--fb-scale", type=float, default=2.0,
                   help="Upscale factor for the /dev/fb0 preview (default 2.0)")
    p.add_argument("--fb-rgb", action="store_true",
                   help="Swap R/B if framebuffer colors look wrong")
    args = p.parse_args()

    scene_name = f"{args.index}_{args.object}"
    scene_dir = Path(args.output_dir) / scene_name
    version = (args.start_version if args.start_version is not None
               else detect_start_version(scene_dir))

    mode = ("SEQUENTIAL" if args.sequential
            else "HEADLESS" if args.headless
            else "FB-PREVIEW" if args.fb_preview else "STREAM")
    print(f"Scene: {scene_name} | Mode: {mode}")
    print(f"Output: {scene_dir}")
    print(f"Capture: native {TARGET_W}x{TARGET_H} (HFOV {OAKD_HFOV_NATIVE:.2f} deg)")
    print()

    if args.sequential:
        run_sequential(args, scene_dir, version)
    elif args.headless:
        run_headless(args, scene_dir, version)
    elif args.fb_preview:
        run_fb_preview(args, scene_dir, version)
    else:
        run_stream_preview(args, scene_dir, version)


if __name__ == "__main__":
    main()

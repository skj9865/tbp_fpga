#!/usr/bin/env python3
"""RGBD capture script for OAK-D Pro camera.

Captures RGB+Depth frames in the same format as the world image dataset,
so SaccadeOnImageEnvironment can load them without modification.

Output format:
  - rgb_{v}.png   : 640x480 RGBA uint8 (alpha=255)
  - depth_{v}.data: 640x480 float32, meters, NaN for invalid

Usage:
    python scripts/rgbd_capture.py --object numenta_mug --index 0
    python scripts/rgbd_capture.py --object numenta_mug --index 0 --preview-only
    python scripts/rgbd_capture.py --object numenta_mug --index 0 --headless
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from PIL import Image


TARGET_W, TARGET_H = 640, 480

# OAK-D Pro RGB at native 640x480 preview has HFOV ~63.75 deg
# (from calibration intrinsics: fx=514.6 -> 2*atan(W/(2*fx))=63.75).
# Non-standard preview sizes trigger ISP cropping (e.g. 864x642 gives ~50.4 deg),
# so we capture at standard 640x480 directly. The downstream Monty environment
# (tbp.monty/.../two_d_data.py) must be configured for this HFOV (default 54.201
# is for iPad — patch it to 63.75 to match OAK-D Pro output).
OAKD_HFOV_NATIVE = 63.75
CAPTURE_W, CAPTURE_H = TARGET_W, TARGET_H


def build_pipeline(preset="DENSITY", max_distance_m=None, min_distance_m=None,
                   fps=30, extended_disparity=False,
                   color_resolution="THE_4_K"):
    """Build depthai v3 pipeline: ColorCamera + StereoDepth aligned to RGB.

    Captures at CAPTURE_W x CAPTURE_H (larger than target) so that center-crop
    to TARGET_W x TARGET_H matches iPad's 54.201 deg hfov without upscaling.

    fps caps the RGB + mono frame rate. Keep 30 on USB3 hosts; drop low
    (e.g. 5) on USB2 hosts (FPGA board) — two 640x480 streams at 30fps
    exceed USB2's ~350Mbit ceiling and crash the VPU (X_LINK_ERROR).

    Returns (pipeline, rgb_output, depth_output) where outputs support
    createOutputQueue() for the v3 API.
    """
    pipeline = dai.Pipeline()

    # Color camera (OAK-D Pro: IMX378, 12MP)
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(CAPTURE_W, CAPTURE_H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # IMX378 native is 4056x3040 = 4:3. THE_4_K (3840x2160) is a 16:9 crop of
    # that, so a 4:3 preview taken from it loses vertical FOV — while depth,
    # aligned to CAM_A's native field, keeps it. Measured on captures: depth
    # must be scaled ~1.22-1.28x vertically to match RGB edges. A 4:3 sensor
    # mode (e.g. THE_1352X1012) keeps both on the same field.
    cam_rgb.setResolution(
        getattr(dai.ColorCameraProperties.SensorResolution, color_resolution)
    )
    print(f"Color sensor mode: {color_resolution}")
    cam_rgb.setFps(fps)

    # Mono cameras for stereo depth (OAK-D Pro: OV9282, 1MP -> 800P available)
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_left.setFps(fps)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_right.setFps(fps)

    # Stereo depth — depthai library DEFAULT settings. No manual post-processing
    # tuning (decimation / temporal / spatial / speckle / threshold, subpixel,
    # extended disparity all removed). Only the two structurally required calls
    # are kept: align depth to the RGB frame, and output at capture size.
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(
        getattr(dai.node.StereoDepth.PresetMode, preset.upper())
    )
    print(f"Preset: {preset.upper()}")
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
    stereo.setOutputSize(CAPTURE_W, CAPTURE_H)

    # Extended disparity doubles the disparity search range, which halves the
    # minimum measurable distance (~35cm -> ~17cm at 400P with the 7.5cm
    # baseline). Needed to shoot objects at the Robot Lab training distance
    # (0.17-0.26m); below the default minimum the object returns no depth at
    # all and only background pixels remain valid.
    if extended_disparity:
        stereo.setExtendedDisparity(True)
        print("Extended disparity: ON (min distance ~35cm -> ~17cm)")

    # Threshold filter = the OAK viewer's "max distance" slider. Clips pixels
    # outside [min, max] (mm), which removes noisy far-background points and
    # cleans the depth map. Only touched when the user passes --max/min-distance.
    if max_distance_m is not None or min_distance_m is not None:
        tf = stereo.initialConfig.postProcessing.thresholdFilter
        if max_distance_m is not None:
            tf.maxRange = int(max_distance_m * 1000)
        if min_distance_m is not None:
            tf.minRange = int(min_distance_m * 1000)
        print(f"Threshold: min={tf.minRange}mm max={tf.maxRange}mm")

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # In depthai v3, output queues are created directly from node outputs
    # (no XLinkOut nodes needed)
    return pipeline, cam_rgb.preview, stereo.depth


def depth_to_colormap(depth_mm, max_m=4.0):
    """Convert uint16 mm depth to a colormap for visualization.

    Invalid (0) pixels shown in red. Normalization range = max_m (meters) so
    the preview color scale matches the capture's max-distance clip.
    """
    invalid_mask = depth_mm == 0
    # Normalize to 0-255 for colormap (clip at max_m)
    depth_norm = np.clip(depth_mm.astype(np.float32) / (max_m * 1000.0), 0, 1)
    depth_u8 = (depth_norm * 255).astype(np.uint8)
    colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
    # Mark invalid pixels red
    colormap[invalid_mask] = [0, 0, 255]  # BGR red
    return colormap


def crop_center(frame, target_w, target_h):
    """Crop center region of exact target size from a larger frame."""
    h, w = frame.shape[:2]
    x_start = (w - target_w) // 2
    y_start = (h - target_h) // 2
    return frame[y_start:y_start + target_h, x_start:x_start + target_w]


def convert_rgb(bgr_frame):
    """BGR frame (already TARGET size) → RGBA."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def convert_depth(depth_mm):
    """uint16 mm depth (already TARGET size) → float32 meters, NaN for invalid."""
    depth_f = depth_mm.astype(np.float32) / 1000.0
    depth_f[depth_mm == 0] = np.nan
    return depth_f


def save_capture(output_dir, version, rgba, depth_m):
    """Save RGBA as PNG and depth as raw float32 .data file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_path = output_dir / f"rgb_{version}.png"
    Image.fromarray(rgba, mode="RGBA").save(str(rgb_path))

    depth_path = output_dir / f"depth_{version}.data"
    depth_m.tofile(str(depth_path))

    print(f"  Saved: {rgb_path.name} + {depth_path.name}")
    return rgb_path, depth_path


def detect_start_version(output_dir):
    """Auto-detect next available version number."""
    if not output_dir.exists():
        return 0
    existing = sorted(output_dir.glob("rgb_*.png"))
    if not existing:
        return 0
    versions = []
    for p in existing:
        try:
            v = int(p.stem.split("_")[1])
            versions.append(v)
        except (ValueError, IndexError):
            continue
    return max(versions) + 1 if versions else 0


def capture_one(q_rgb, q_depth, scene_dir, version):
    """Wait for both RGB+depth frames and save. Returns True on success."""
    bgr_frame = None
    depth_mm = None

    # Wait up to 5 seconds for both frames
    timeout = time.time() + 5
    while time.time() < timeout:
        in_rgb = q_rgb.tryGet()
        in_depth = q_depth.tryGet()
        if in_rgb is not None:
            bgr_frame = in_rgb.getCvFrame()
        if in_depth is not None:
            depth_mm = in_depth.getFrame()
        if bgr_frame is not None and depth_mm is not None:
            break
        time.sleep(0.01)

    if bgr_frame is None or depth_mm is None:
        print(f"  [v={version}] timeout waiting for frames")
        return False

    rgba = convert_rgb(bgr_frame)
    depth_m = convert_depth(depth_mm)

    nan_pct = np.isnan(depth_m).sum() / depth_m.size * 100
    valid = depth_m[~np.isnan(depth_m)]
    if valid.size > 0:
        print(f"  [v={version}] depth: {valid.min():.3f}-{valid.max():.3f}m, "
              f"NaN={nan_pct:.1f}%")
    else:
        print(f"  [v={version}] depth: all NaN")

    save_capture(scene_dir, version, rgba, depth_m)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Capture RGBD frames from OAK-D Pro for Monty inference"
    )
    parser.add_argument("--object", required=True, help="Object name (e.g. numenta_mug)")
    parser.add_argument("--index", required=True, type=int, help="Scene index")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "tbp/data/worldimages/captured_scenes"),
        help="Output base directory",
    )
    parser.add_argument(
        "--start-version", type=int, default=None, help="Start version number (auto)"
    )
    parser.add_argument(
        "--num-captures", type=int, default=4, help="Max number of captures"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="Preview only, no saving"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No display mode: auto-capture after warmup, no GUI"
    )
    parser.add_argument(
        "--warmup", type=float, default=2.0,
        help="Headless mode: seconds to wait before capturing (default: 2.0)"
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Headless mode: seconds between captures (default: 1.0)"
    )
    parser.add_argument(
        "--ir-brightness", type=int, default=1200,
        help="OAK-D Pro IR dot projector brightness in mA (0-1200, 0=off, default: 1200=max)"
    )
    parser.add_argument(
        "--flood-brightness", type=int, default=0,
        help="OAK-D Pro IR flood light brightness in mA (0-1500, default: 0=off)"
    )
    parser.add_argument(
        "--preset", default="DENSITY",
        help="StereoDepth preset: DEFAULT, ACCURACY, DENSITY, FAST_ACCURACY, "
             "FAST_DENSITY, FACE, HIGH_DETAIL, ROBOTICS (default: DENSITY)"
    )
    parser.add_argument(
        "--max-distance", type=float, default=None,
        help="Clip depth beyond this many meters (= OAK viewer 'max distance' "
             "slider; removes far-background noise). Default: preset default."
    )
    parser.add_argument(
        "--min-distance", type=float, default=None,
        help="Clip depth closer than this many meters. Default: preset default."
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Camera frame rate (default: 30)"
    )
    args = parser.parse_args()

    scene_name = f"{args.index}_{args.object}"
    scene_dir = Path(args.output_dir) / scene_name

    if args.start_version is not None:
        version = args.start_version
    else:
        version = detect_start_version(scene_dir)

    mode = ("HEADLESS" if args.headless
            else "PREVIEW ONLY" if args.preview_only
            else f"CAPTURE (start v={version})")
    print(f"Scene: {scene_name} | Mode: {mode}")
    print(f"Output: {scene_dir}")
    print(f"Capture: native {TARGET_W}x{TARGET_H} (HFOV {OAKD_HFOV_NATIVE:.2f} deg)")
    if not args.headless:
        print("Controls: 'c' = capture, 'q' = quit")
    print()

    pipeline, rgb_out, depth_out = build_pipeline(
        preset=args.preset,
        max_distance_m=args.max_distance,
        min_distance_m=args.min_distance,
        fps=args.fps,
    )
    # Preview color scale matches the max-distance clip (falls back to 4m).
    vis_max_m = args.max_distance if args.max_distance is not None else 4.0

    # depthai v3: create output queues directly from node outputs
    q_rgb = rgb_out.createOutputQueue()
    q_depth = depth_out.createOutputQueue()

    pipeline.start()

    # Enable OAK-D Pro IR dot projector (and optional flood light) for
    # active stereo. depthai v3 uses Intensity (0.0-1.0); v2 used Brightness (mA).
    dot_intensity = args.ir_brightness / 1200.0  # mA -> normalized
    flood_intensity = args.flood_brightness / 1500.0
    try:
        device = pipeline.getDefaultDevice()
        if device is not None:
            ok = False
            # Try v3 API first
            if hasattr(device, "setIrLaserDotProjectorIntensity"):
                device.setIrLaserDotProjectorIntensity(dot_intensity)
                if flood_intensity > 0:
                    device.setIrFloodLightIntensity(flood_intensity)
                ok = True
                print(f"IR projector (v3): dot={dot_intensity:.2f} "
                      f"flood={flood_intensity:.2f}")
            elif hasattr(device, "setIrLaserDotProjectorBrightness"):
                device.setIrLaserDotProjectorBrightness(args.ir_brightness)
                if args.flood_brightness > 0:
                    device.setIrFloodLightBrightness(args.flood_brightness)
                ok = True
                print(f"IR projector (v2): dot={args.ir_brightness}mA "
                      f"flood={args.flood_brightness}mA")
            if not ok:
                print("IR control: no compatible API found")
    except Exception as e:
        print(f"IR control failed: {e}")

    try:
        if args.headless:
            # Headless mode: warmup then auto-capture
            print(f"Warming up ({args.warmup}s)...")
            time.sleep(args.warmup)
            # Drain old frames
            while q_rgb.tryGet() is not None:
                pass
            while q_depth.tryGet() is not None:
                pass

            for i in range(args.num_captures):
                print(f"Capturing {i+1}/{args.num_captures}...")
                if capture_one(q_rgb, q_depth, scene_dir, version):
                    version += 1
                if i < args.num_captures - 1:
                    time.sleep(args.interval)

            print(f"\nDone. {args.num_captures} frame(s) captured.")
        else:
            # GUI mode: interactive with preview
            bgr_frame = None
            depth_mm = None
            captures_done = 0

            win_name = f"OAK-D Pro | {scene_name}"
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

            while pipeline.isRunning():
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is not None:
                    bgr_frame = in_rgb.getCvFrame()
                if in_depth is not None:
                    depth_mm = in_depth.getFrame()

                if bgr_frame is not None and depth_mm is not None:
                    depth_vis = depth_to_colormap(depth_mm, max_m=vis_max_m)
                    preview = np.hstack([bgr_frame, depth_vis])
                    cv2.imshow(win_name, preview)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print(f"\nDone. {captures_done} frame(s) captured.")
                    break

                if key == ord("c"):
                    if args.preview_only:
                        print("  [preview-only mode, not saving]")
                        continue

                    if bgr_frame is None or depth_mm is None:
                        print("  [no frames yet, wait...]")
                        continue

                    if captures_done >= args.num_captures:
                        print(f"  Max captures ({args.num_captures}) reached. Press 'q'.")
                        continue

                    rgba = convert_rgb(bgr_frame)
                    depth_m = convert_depth(depth_mm)

                    nan_pct = np.isnan(depth_m).sum() / depth_m.size * 100
                    valid = depth_m[~np.isnan(depth_m)]
                    if valid.size > 0:
                        print(f"  [v={version}] depth: {valid.min():.3f}-{valid.max():.3f}m, "
                              f"NaN={nan_pct:.1f}%")
                    else:
                        print(f"  [v={version}] depth: all NaN")

                    save_capture(scene_dir, version, rgba, depth_m)
                    version += 1
                    captures_done += 1

            cv2.destroyAllWindows()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

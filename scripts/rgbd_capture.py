#!/usr/bin/env python3
"""RGBD capture script for OAK-D Lite camera.

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

# FOV matching: crop OAK-D's wider FOV to match iPad's hfov used by Monty
OAKD_HFOV = 69.4    # OAK-D Lite preview actual hfov (from calibration)
IPAD_HFOV = 54.201  # iPad front camera hfov hardcoded in Monty

# Capture at larger size so that after center-crop we still get 640x480
_CROP_RATIO = np.tan(np.radians(IPAD_HFOV / 2)) / np.tan(np.radians(OAKD_HFOV / 2))
CAPTURE_W = int(np.ceil(TARGET_W / _CROP_RATIO / 16) * 16)   # ~880, multiple of 16
CAPTURE_H = int(np.ceil(TARGET_H / _CROP_RATIO / 2) * 2)     # ~654, keep even


def build_pipeline():
    """Build depthai v3 pipeline: ColorCamera + StereoDepth aligned to RGB.

    Captures at CAPTURE_W x CAPTURE_H (larger than target) so that center-crop
    to TARGET_W x TARGET_H matches iPad's 54.201 deg hfov without upscaling.

    Returns (pipeline, rgb_output, depth_output) where outputs support
    createOutputQueue() for the v3 API.
    """
    pipeline = dai.Pipeline()

    # Color camera
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(CAPTURE_W, CAPTURE_H)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    # OAK-D Lite native sensor is 4208x3120; 1080P causes mismatch.
    # Use 4K and let preview handle the downscale.
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setFps(30)

    # Mono cameras for stereo depth
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)

    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

    # Stereo depth
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.FAST_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # Align to RGB
    stereo.setSubpixel(True)
    stereo.setExtendedDisparity(True)
    stereo.setOutputSize(CAPTURE_W, CAPTURE_H)

    # Post-processing filters to reduce stereo noise
    # Note: median filter disabled — subpixel+extended disparity exceeds max (1024)
    pp = stereo.initialConfig.postProcessing
    pp.speckleFilter.enable = True
    pp.speckleFilter.speckleRange = 28  # max supported by default memory
    pp.temporalFilter.enable = True
    pp.spatialFilter.enable = True
    pp.spatialFilter.holeFillingRadius = 2
    pp.spatialFilter.numIterations = 1
    pp.thresholdFilter.minRange = 200
    pp.thresholdFilter.maxRange = 4000

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    # In depthai v3, output queues are created directly from node outputs
    # (no XLinkOut nodes needed)
    return pipeline, cam_rgb.preview, stereo.depth


def depth_to_colormap(depth_mm):
    """Convert uint16 mm depth to a colormap for visualization.

    Invalid (0) pixels shown in red.
    """
    invalid_mask = depth_mm == 0
    # Normalize to 0-255 for colormap (clip at 4m)
    depth_norm = np.clip(depth_mm.astype(np.float32) / 4000.0, 0, 1)
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
    """BGR frame (CAPTURE size) → center crop to TARGET → RGBA (no upscale)."""
    cropped = crop_center(bgr_frame, TARGET_W, TARGET_H)
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def convert_depth(depth_mm):
    """uint16 mm depth (CAPTURE size) → center crop to TARGET → float32 meters."""
    cropped = crop_center(depth_mm, TARGET_W, TARGET_H)
    depth_f = cropped.astype(np.float32) / 1000.0
    depth_f[cropped == 0] = np.nan
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
        description="Capture RGBD frames from OAK-D Lite for Monty inference"
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
    args = parser.parse_args()

    scene_name = f"{args.index}_{args.object}"
    scene_dir = Path(args.output_dir) / scene_name

    if args.start_version is not None:
        version = args.start_version
    else:
        version = detect_start_version(scene_dir)

    mode = "HEADLESS" if args.headless else ("PREVIEW ONLY" if args.preview_only else f"CAPTURE (start v={version})")
    print(f"Scene: {scene_name} | Mode: {mode}")
    print(f"Output: {scene_dir}")
    print(f"Capture: {CAPTURE_W}x{CAPTURE_H} -> crop center {TARGET_W}x{TARGET_H} "
          f"(FOV {OAKD_HFOV:.1f} -> {IPAD_HFOV:.1f})")
    if not args.headless:
        print("Controls: 'c' = capture, 'q' = quit")
    print()

    pipeline, rgb_out, depth_out = build_pipeline()

    # depthai v3: create output queues directly from node outputs
    q_rgb = rgb_out.createOutputQueue()
    q_depth = depth_out.createOutputQueue()

    pipeline.start()
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

            win_name = f"OAK-D Lite | {scene_name}"
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

            while pipeline.isRunning():
                in_rgb = q_rgb.tryGet()
                in_depth = q_depth.tryGet()

                if in_rgb is not None:
                    bgr_frame = in_rgb.getCvFrame()
                if in_depth is not None:
                    depth_mm = in_depth.getFrame()

                if bgr_frame is not None and depth_mm is not None:
                    preview_bgr = crop_center(bgr_frame, TARGET_W, TARGET_H)
                    depth_vis = depth_to_colormap(crop_center(depth_mm, TARGET_W, TARGET_H))
                    preview = np.hstack([preview_bgr, depth_vis])
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

#!/usr/bin/env python3
"""RGBD capture script for Intel RealSense D405.

Captures RGB+Depth frames matching iPad TrueDepth FOV (54.201 deg) by
center-cropping D405's wider native FOV (~89.3 deg at 1280x720).
Output format identical to rgbd_capture.py so SaccadeOnImageEnvironment
can load files without modification.

Output:
  - rgb_{v}.png   : 640x480 RGBA uint8 (alpha=255)
  - depth_{v}.data: 640x480 float32, meters, NaN for invalid

Usage:
    python scripts/rgbd_capture_d405.py --object numenta_mug --index 0
    python scripts/rgbd_capture_d405.py --object numenta_mug --index 0 --headless
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from PIL import Image


# Target output (matches OAK / iPad pipeline)
TARGET_W, TARGET_H = 640, 480
TARGET_HFOV = 54.201  # iPad TrueDepth HFOV — matches Monty pretrained models

# D405 native streaming resolution
NATIVE_W, NATIVE_H = 1280, 720
D405_HFOV = 89.277  # from calibration (color stream)

# Compute center-crop size for target HFOV on a 4:3 output aspect ratio.
# crop_ratio = tan(target/2) / tan(native/2)
_CROP_RATIO = np.tan(np.radians(TARGET_HFOV / 2)) / np.tan(np.radians(D405_HFOV / 2))
CROP_W = int(round(NATIVE_W * _CROP_RATIO))             # ~663
# Force 4:3 aspect ratio matching the target
CROP_H = int(round(CROP_W * TARGET_H / TARGET_W))        # ~497
assert CROP_H <= NATIVE_H, (
    f"crop height {CROP_H} exceeds native height {NATIVE_H}"
)


def build_pipeline():
    """Build RealSense pipeline streaming color + depth at native resolution."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, NATIVE_W, NATIVE_H, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, NATIVE_W, NATIVE_H, rs.format.z16, 30)

    profile = pipeline.start(config)

    # Align depth to color (D405 is near-coaxial but align ensures pixel match)
    align = rs.align(rs.stream.color)

    # Depth scale (units -> meters)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"Depth scale: {depth_scale} m/unit")

    return pipeline, align, depth_scale


# ---------------------------------------------------------------------------
# Depth post-processing (edge-preserving smoothing)
# ---------------------------------------------------------------------------

SMOOTH_PRESETS = {
    # spatial: magnitude (iterations), alpha (smooth), delta (edge threshold)
    # temporal: alpha (frame weight 0-1), delta (edge threshold)
    # hole:     mode 0=fill_left, 1=farthest_around, 2=nearest_around
    "light":  dict(sp_mag=1, sp_alpha=0.6, sp_delta=20,
                   tp_alpha=0.4, tp_delta=20, hole_mode=1),
    "medium": dict(sp_mag=2, sp_alpha=0.5, sp_delta=20,
                   tp_alpha=0.5, tp_delta=20, hole_mode=1),
    "heavy":  dict(sp_mag=3, sp_alpha=0.4, sp_delta=15,
                   tp_alpha=0.65, tp_delta=15, hole_mode=2),
}


def build_filters(preset="medium"):
    """Edge-preserving depth filter chain (Intel-recommended order).

    depth -> disparity_xform -> spatial -> temporal -> depth_xform -> hole_fill

    Decimation is skipped so we keep full 1280x720 (cropped later for FOV match).
    spatial.delta and temporal.delta are edge thresholds: differences larger
    than `delta` are treated as boundaries and NOT smoothed across — this
    protects the mug handle edge from being blurred away.
    """
    p = SMOOTH_PRESETS[preset]

    to_disparity = rs.disparity_transform(True)
    to_depth = rs.disparity_transform(False)

    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, p["sp_mag"])
    spatial.set_option(rs.option.filter_smooth_alpha, p["sp_alpha"])
    spatial.set_option(rs.option.filter_smooth_delta, p["sp_delta"])
    spatial.set_option(rs.option.holes_fill, 1)  # tiny holes during spatial

    temporal = rs.temporal_filter()
    temporal.set_option(rs.option.filter_smooth_alpha, p["tp_alpha"])
    temporal.set_option(rs.option.filter_smooth_delta, p["tp_delta"])

    hole_filling = rs.hole_filling_filter()
    hole_filling.set_option(rs.option.holes_fill, p["hole_mode"])

    return [to_disparity, spatial, temporal, to_depth, hole_filling]


def apply_depth_filters(depth_frame, filters):
    """Apply filter chain to a depth frame; returns the processed frame."""
    for f in filters:
        depth_frame = f.process(depth_frame)
    return depth_frame


def crop_and_downscale_rgb(bgr_frame):
    """Center-crop BGR frame to target HFOV, downscale to 640x480, return RGBA."""
    h, w = bgr_frame.shape[:2]
    x0 = (w - CROP_W) // 2
    y0 = (h - CROP_H) // 2
    cropped = bgr_frame[y0:y0 + CROP_H, x0:x0 + CROP_W]
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def crop_and_downscale_depth(depth_units, depth_scale):
    """Center-crop depth, downscale, convert units→meters, NaN for invalid."""
    h, w = depth_units.shape
    x0 = (w - CROP_W) // 2
    y0 = (h - CROP_H) // 2
    cropped = depth_units[y0:y0 + CROP_H, x0:x0 + CROP_W]
    # Resize with INTER_NEAREST to preserve invalid (0) pixels.
    resized = cv2.resize(
        cropped, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST
    )
    depth_m = resized.astype(np.float32) * depth_scale
    depth_m[resized == 0] = np.nan
    return depth_m


def depth_to_colormap(depth_units, max_units=10000):
    """uint16 depth (D405 units) -> colormap for preview. Invalid (0) shown red."""
    invalid_mask = depth_units == 0
    norm = np.clip(depth_units.astype(np.float32) / max_units, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cmap[invalid_mask] = [0, 0, 255]  # BGR red
    return cmap


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
    existing = sorted(output_dir.glob("rgb_*.png"))
    versions = []
    for p in existing:
        try:
            versions.append(int(p.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue
    return max(versions) + 1 if versions else 0


def capture_one(pipeline, align, depth_scale, scene_dir, version,
                filters=None, timeout_s=5.0):
    """Wait for synced RGB+depth frames and save. Returns True on success."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            continue
        aligned = align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        if not color or not depth:
            continue

        if filters is not None:
            depth = apply_depth_filters(depth, filters)

        bgr = np.asanyarray(color.get_data())
        depth_units = np.asanyarray(depth.get_data())

        rgba = crop_and_downscale_rgb(bgr)
        depth_m = crop_and_downscale_depth(depth_units, depth_scale)

        nan_pct = np.isnan(depth_m).sum() / depth_m.size * 100
        valid = depth_m[~np.isnan(depth_m)]
        if valid.size > 0:
            print(f"  [v={version}] depth: {valid.min():.3f}-{valid.max():.3f}m, "
                  f"NaN={nan_pct:.1f}%")
        else:
            print(f"  [v={version}] depth: all NaN")

        save_capture(scene_dir, version, rgba, depth_m)
        return True

    print(f"  [v={version}] timeout waiting for frames")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Capture RGBD frames from Intel RealSense D405 for Monty"
    )
    parser.add_argument("--object", required=True, help="Object name (e.g. numenta_mug)")
    parser.add_argument("--index", required=True, type=int, help="Scene index")
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "tbp/data/worldimages/captured_scenes"),
        help="Output base directory",
    )
    parser.add_argument(
        "--start-version", type=int, default=None,
        help="Start version number (default: auto-detect next)",
    )
    parser.add_argument(
        "--num-captures", type=int, default=4, help="Max number of captures"
    )
    parser.add_argument(
        "--preview-only", action="store_true", help="Preview only, no saving"
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No display mode: auto-capture after warmup",
    )
    parser.add_argument(
        "--warmup", type=float, default=2.0,
        help="Headless mode: seconds before first capture (default: 2.0)",
    )
    parser.add_argument(
        "--interval", type=float, default=1.0,
        help="Headless mode: seconds between captures (default: 1.0)",
    )
    parser.add_argument(
        "--laser-power", type=float, default=None,
        help="D405 IR laser power 0-360 mW (default: device default ~150)",
    )
    parser.add_argument(
        "--smooth", action="store_true",
        help="Enable RealSense SDK depth filters during capture (default: raw). "
             "For testing, prefer raw capture + separate scripts/smooth_depth.py.",
    )
    parser.add_argument(
        "--smooth-preset", choices=["light", "medium", "heavy"], default="medium",
        help="Smoothing intensity if --smooth is on (default: medium)",
    )
    args = parser.parse_args()

    scene_name = f"{args.index}_{args.object}"
    scene_dir = Path(args.output_dir) / scene_name
    version = args.start_version if args.start_version is not None \
        else detect_start_version(scene_dir)

    mode = "HEADLESS" if args.headless else (
        "PREVIEW ONLY" if args.preview_only else f"CAPTURE (start v={version})"
    )
    print(f"Scene: {scene_name} | Mode: {mode}")
    print(f"Output: {scene_dir}")
    print(f"Capture: native {NATIVE_W}x{NATIVE_H} (HFOV {D405_HFOV:.2f} deg) "
          f"-> center crop {CROP_W}x{CROP_H} -> downscale "
          f"{TARGET_W}x{TARGET_H} (HFOV {TARGET_HFOV:.2f} deg)")
    if not args.headless:
        print("Controls: 'c' = capture, 'q' = quit")

    pipeline, align, depth_scale = build_pipeline()

    filters = build_filters(args.smooth_preset) if args.smooth else None
    print(f"Depth smoothing: "
          f"{args.smooth_preset if filters is not None else 'OFF (raw)'}")

    # Optional laser power adjustment
    if args.laser_power is not None:
        try:
            depth_sensor = pipeline.get_active_profile() \
                .get_device().first_depth_sensor()
            depth_sensor.set_option(rs.option.laser_power, args.laser_power)
            print(f"Laser power set to {args.laser_power} mW")
        except Exception as e:
            print(f"Laser power set failed: {e}")

    try:
        # Warm-up: drain initial frames so auto-exposure stabilizes AND
        # temporal filter accumulates history before the first capture.
        for _ in range(15):
            try:
                frames = pipeline.wait_for_frames(timeout_ms=500)
                if filters is not None:
                    aligned = align.process(frames)
                    d = aligned.get_depth_frame()
                    if d:
                        apply_depth_filters(d, filters)
            except RuntimeError:
                pass

        if args.headless:
            print(f"Warming up ({args.warmup}s)...")
            t_end = time.time() + args.warmup
            while time.time() < t_end:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=500)
                    if filters is not None:
                        aligned = align.process(frames)
                        d = aligned.get_depth_frame()
                        if d:
                            apply_depth_filters(d, filters)
                except RuntimeError:
                    pass
            for i in range(args.num_captures):
                print(f"Capturing {i+1}/{args.num_captures}...")
                if capture_one(pipeline, align, depth_scale, scene_dir, version,
                               filters=filters):
                    version += 1
                if i < args.num_captures - 1:
                    # Keep filter state warm during inter-capture wait
                    t_next = time.time() + args.interval
                    while time.time() < t_next:
                        try:
                            frames = pipeline.wait_for_frames(timeout_ms=200)
                            if filters is not None:
                                aligned = align.process(frames)
                                d = aligned.get_depth_frame()
                                if d:
                                    apply_depth_filters(d, filters)
                        except RuntimeError:
                            pass
            print(f"\nDone. {args.num_captures} frame(s) captured.")
        else:
            captures_done = 0
            win_name = f"D405 | {scene_name}"
            cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)

            while True:
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1000)
                except RuntimeError:
                    continue
                aligned = align.process(frames)
                color = aligned.get_color_frame()
                depth = aligned.get_depth_frame()
                if not color or not depth:
                    continue

                if filters is not None:
                    depth = apply_depth_filters(depth, filters)

                bgr = np.asanyarray(color.get_data())
                depth_units = np.asanyarray(depth.get_data())

                # Preview: show the cropped & resized region
                rgba_preview = crop_and_downscale_rgb(bgr)
                preview_bgr = cv2.cvtColor(rgba_preview[:, :, :3], cv2.COLOR_RGB2BGR)
                # Depth preview: also crop & resize, then colormap
                h, w = depth_units.shape
                x0 = (w - CROP_W) // 2
                y0 = (h - CROP_H) // 2
                d_crop = depth_units[y0:y0 + CROP_H, x0:x0 + CROP_W]
                d_resized = cv2.resize(d_crop, (TARGET_W, TARGET_H),
                                       interpolation=cv2.INTER_NEAREST)
                depth_vis = depth_to_colormap(d_resized)
                preview = np.hstack([preview_bgr, depth_vis])
                cv2.imshow(win_name, preview)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print(f"\nDone. {captures_done} frame(s) captured.")
                    break
                if key == ord("c"):
                    if args.preview_only:
                        print("  [preview-only mode]")
                        continue
                    if captures_done >= args.num_captures:
                        print(f"  Max captures ({args.num_captures}) reached. Press 'q'.")
                        continue
                    rgba = crop_and_downscale_rgb(bgr)
                    depth_m = crop_and_downscale_depth(depth_units, depth_scale)
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

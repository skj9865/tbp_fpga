#!/usr/bin/env python3
"""Print Orbbec Femto Bolt calibration for FOV/intrinsic verification.

Runs in the `orbbec` conda env (Python 3.11 + pyorbbecsdk). Mirrors the
purpose of check_d405_calibration.py — confirms device detection, prints
intrinsics for both color and depth streams, and reports HFOV / depth
scale so we can crop to Monty's 54.201 deg target.

Usage (must be run with camera permission granted to Terminal):

    /opt/homebrew/Caskroom/miniforge/base/envs/orbbec/bin/python \\
        scripts/check_femto_calibration.py
"""

import numpy as np
import pyorbbecsdk as ob


def hfov_from(width, fx):
    return 2 * np.degrees(np.arctan(width / (2 * fx)))


def vfov_from(height, fy):
    return 2 * np.degrees(np.arctan(height / (2 * fy)))


def print_intrinsics(label, intr):
    w, h = intr.width, intr.height
    fx, fy = intr.fx, intr.fy
    cx, cy = intr.cx, intr.cy
    print(f"\n--- {label} @ {w}x{h} ---")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  HFOV={hfov_from(w, fx):.3f} deg   VFOV={vfov_from(h, fy):.3f} deg")


def list_stream_modes(profile_list, kind):
    """List available video stream profiles for a sensor."""
    try:
        n = profile_list.get_count()
    except Exception:
        n = 0
    print(f"\n=== Available {kind} stream modes ({n}) ===")
    seen = set()
    for i in range(n):
        try:
            p = profile_list.get_stream_profile_by_index(i)
            vp = p.as_video_stream_profile()
            key = (vp.get_width(), vp.get_height(),
                   vp.get_fps(), str(vp.get_format()))
            if key in seen:
                continue
            seen.add(key)
            print(f"  {vp.get_width()}x{vp.get_height()} "
                  f"@ {vp.get_fps()}fps  fmt={vp.get_format()}")
        except Exception as e:
            print(f"  [{i}] (cannot inspect: {e})")


def main():
    ctx = ob.Context()
    ctx.set_logger_level(ob.OBLogLevel.ERROR)

    devs = ctx.query_devices()
    if devs.get_count() == 0:
        print("ERROR: no Orbbec device found")
        return

    d = devs.get_device_by_index(0)
    info = d.get_device_info()
    print(f"Device:   {info.get_name()}")
    print(f"PID:      0x{info.get_pid():04x}")
    print(f"SN:       {info.get_serial_number()}")
    print(f"Firmware: {info.get_firmware_version()}")
    print(f"USB:      {info.get_connection_type()}")

    pipeline = ob.Pipeline(d)

    # Enumerate stream modes
    color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    depth_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    list_stream_modes(color_list, "color")
    list_stream_modes(depth_list, "depth")

    # Configure WFOV close-range — start a brief pipeline to grab camera params
    config = ob.Config()
    # Try to find depth + color profiles that work together.
    # Default profiles (highest-priority) are usually safe.
    try:
        color_profile = color_list.get_default_video_stream_profile()
    except Exception:
        color_profile = color_list.get_stream_profile_by_index(0)
    try:
        depth_profile = depth_list.get_default_video_stream_profile()
    except Exception:
        depth_profile = depth_list.get_stream_profile_by_index(0)

    print(f"\nDefault color: {color_profile.as_video_stream_profile().get_width()}"
          f"x{color_profile.as_video_stream_profile().get_height()} "
          f"@ {color_profile.as_video_stream_profile().get_fps()}fps")
    print(f"Default depth: {depth_profile.as_video_stream_profile().get_width()}"
          f"x{depth_profile.as_video_stream_profile().get_height()} "
          f"@ {depth_profile.as_video_stream_profile().get_fps()}fps")

    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    pipeline.start(config)
    try:
        # Wait for a few frames so calibration is populated
        for _ in range(5):
            frames = pipeline.wait_for_frames(2000)
            if frames is not None:
                break

        # Camera params (intrinsics + extrinsics)
        cam = pipeline.get_camera_param()
        print_intrinsics(
            "Color intrinsics", cam.rgb_intrinsic,
        )
        print_intrinsics(
            "Depth intrinsics", cam.depth_intrinsic,
        )

        # Depth scale (units -> meters)
        depth_frame = None
        if frames is not None:
            depth_frame = frames.get_depth_frame()
        if depth_frame is not None:
            scale = depth_frame.get_depth_scale()
            print(f"\nDepth scale: {scale} (mm/unit "
                  f"= {scale / 1000.0} m/unit)")

        # Equivalent HFOV when downscaled to 640 wide
        d_intr = cam.depth_intrinsic
        c_intr = cam.rgb_intrinsic
        print(f"\n--- Equivalent at 640x480 (downscaled) ---")
        c_fx_640 = c_intr.fx * (640.0 / c_intr.width)
        d_fx_640 = d_intr.fx * (640.0 / d_intr.width)
        print(f"  color fx@640={c_fx_640:.2f}  "
              f"HFOV={hfov_from(640, c_fx_640):.3f} deg")
        print(f"  depth fx@640={d_fx_640:.2f}  "
              f"HFOV={hfov_from(640, d_fx_640):.3f} deg")

        target_hfov = 54.201
        # crop_ratio = tan(target/2) / tan(native/2) — use depth HFOV
        native_hfov = hfov_from(d_intr.width, d_intr.fx)
        crop_ratio = (
            np.tan(np.radians(target_hfov / 2))
            / np.tan(np.radians(native_hfov / 2))
        )
        crop_w = int(round(d_intr.width * crop_ratio))
        crop_h = int(round(crop_w * 480 / 640))
        print(f"\n--- For Monty 54.201 deg target (matches iPad) ---")
        print(f"  native depth HFOV = {native_hfov:.3f} deg")
        print(f"  crop_ratio = {crop_ratio:.4f}")
        print(f"  center crop: {crop_w} x {crop_h} -> resize to 640 x 480")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

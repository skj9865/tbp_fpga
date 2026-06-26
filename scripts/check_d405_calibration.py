#!/usr/bin/env python3
"""Print Intel RealSense D405 calibration for FOV/intrinsic verification.

Run this first to confirm D405 detection and obtain the actual HFOV to use
in the Monty environment configuration.
"""
import numpy as np
import pyrealsense2 as rs


def main():
    print("Connecting to RealSense D405...")
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        print("ERROR: no RealSense device found")
        return
    for d in devices:
        print(f"Device: {d.get_info(rs.camera_info.name)} "
              f"S/N {d.get_info(rs.camera_info.serial_number)}")

    pipeline = rs.pipeline()
    config = rs.config()

    # D405 supports up to 1280x720 for both
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    profile = pipeline.start(config)

    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()

    color_intr = color_stream.get_intrinsics()
    depth_intr = depth_stream.get_intrinsics()

    print(f"\n--- Color (RGB) intrinsics @ {color_intr.width}x{color_intr.height} ---")
    print(f"  fx={color_intr.fx:.2f}  fy={color_intr.fy:.2f}")
    print(f"  cx={color_intr.ppx:.2f}  cy={color_intr.ppy:.2f}")
    hfov_c = 2 * np.degrees(np.arctan(color_intr.width / (2 * color_intr.fx)))
    vfov_c = 2 * np.degrees(np.arctan(color_intr.height / (2 * color_intr.fy)))
    print(f"  HFOV={hfov_c:.3f} deg   VFOV={vfov_c:.3f} deg")
    print(f"  Distortion model: {color_intr.model}  coeffs={color_intr.coeffs}")

    print(f"\n--- Depth intrinsics @ {depth_intr.width}x{depth_intr.height} ---")
    print(f"  fx={depth_intr.fx:.2f}  fy={depth_intr.fy:.2f}")
    print(f"  cx={depth_intr.ppx:.2f}  cy={depth_intr.ppy:.2f}")
    hfov_d = 2 * np.degrees(np.arctan(depth_intr.width / (2 * depth_intr.fx)))
    print(f"  HFOV={hfov_d:.3f} deg")

    # Effective HFOV if cropped to 640x480
    fx_at_640 = color_intr.fx * (640.0 / color_intr.width)
    hfov_at_640 = 2 * np.degrees(np.arctan(640 / (2 * fx_at_640)))
    print(f"\n--- Equivalent at 640x480 (downscaled) ---")
    print(f"  fx_at_640={fx_at_640:.2f}  HFOV={hfov_at_640:.3f} deg")

    # Depth scale (units per meter)
    depth_sensor = profile.get_device().first_depth_sensor()
    scale = depth_sensor.get_depth_scale()
    print(f"\nDepth scale: {scale} m/unit  (so depth uint16 value * {scale} = meters)")

    pipeline.stop()
    print(f"\nUse HFOV ~{hfov_c:.2f} deg in Monty env (two_d_data.py hfov param).")


if __name__ == "__main__":
    main()

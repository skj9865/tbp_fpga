#!/usr/bin/env python3
"""Print OAK-D Pro RGB camera calibration for FOV/intrinsic verification.

Use this to verify rgbd_capture.py's OAKD_HFOV constant matches reality.
"""
import depthai as dai
import numpy as np


def main():
    print("Connecting to OAK-D Pro...")
    with dai.Device() as device:
        try:
            print(f"Device name: {device.getDeviceName()}")
        except Exception:
            pass
        try:
            print(f"Connected cameras: {device.getConnectedCameraFeatures()}")
        except Exception as e:
            print(f"(getConnectedCameraFeatures unavailable: {e})")

        calib = device.readCalibration()

        # RGB camera (CAM_A)
        cam = dai.CameraBoardSocket.CAM_A
        try:
            hfov = calib.getFov(cam)
            print(f"\n--- RGB camera (CAM_A) ---")
            print(f"HFOV (calibration): {hfov:.3f} deg")
        except Exception as e:
            print(f"getFov failed: {e}")

        # Try common capture sizes (incl. current and updated capture sizes)
        for w, h in [(1920, 1080), (3840, 2160), (864, 642), (784, 584), (640, 480)]:
            try:
                K = calib.getCameraIntrinsics(cam, w, h)
                K = np.array(K)
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                # Effective hfov from fx
                hfov_from_fx = 2 * np.degrees(np.arctan(w / (2 * fx)))
                print(f"\n  At {w}x{h}: fx={fx:.1f} fy={fy:.1f} "
                      f"cx={cx:.1f} cy={cy:.1f}  hfov_from_fx={hfov_from_fx:.2f}deg")
            except Exception as e:
                print(f"  Intrinsics for {w}x{h} unavailable: {e}")

        # Stereo (depth alignment)
        print(f"\n--- Stereo baseline ---")
        try:
            baseline = calib.getBaselineDistance(useSpecTranslation=False)
            print(f"Baseline (CAM_B<->CAM_C): {baseline:.3f} cm")
        except Exception as e:
            print(f"Baseline unavailable: {e}")

    print("\nCompare HFOV above to rgbd_capture.py OAKD_HFOV (currently 68.7).")


if __name__ == "__main__":
    main()

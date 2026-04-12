import depthai as dai                                                                                                                                                                                                                             
with dai.Device() as device:
    calib = device.readCalibration()
    # RGB camera (CAM_A) FOV at different resolutions
    print('RGB full sensor hfov:', calib.getFov(dai.CameraBoardSocket.CAM_A))

    # Get intrinsics for 640x480 (what preview actually uses)
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, 640, 480)
    fx = intrinsics[0][0]
    import numpy as np
    actual_hfov = 2 * np.degrees(np.arctan(320 / fx))
    print(f'Preview 640x480 fx={fx:.1f}, actual hfov={actual_hfov:.1f}°')
    print(f'iPad hfov: 54.201°')
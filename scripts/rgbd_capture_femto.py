#!/usr/bin/env python3
"""RGBD capture script for Orbbec Femto Bolt (ToF).

Captures RGB + depth frames matching iPad TrueDepth FOV (54.201 deg) by
center-cropping Femto Bolt's wider native depth FOV (~90.78 deg in WFOV
1024x1024 mode). Output format identical to rgbd_capture_d405.py so
SaccadeOnImageEnvironment can load files without modification.

Output:
  - rgb_{v}.png   : 640x480 RGBA uint8 (alpha=255)
  - depth_{v}.data: 640x480 float32, meters, NaN for invalid

Run in the orbbec conda env (Python 3.11 + pyorbbecsdk):

    /opt/homebrew/Caskroom/miniforge/base/envs/orbbec/bin/python \\
        scripts/rgbd_capture_femto.py --object numenta_mug --index 0

Usage:
    --headless  : auto-capture after warmup (no GUI required)
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyorbbecsdk as ob
from PIL import Image


# Target output (matches OAK / D405 / iPad pipeline)
TARGET_W, TARGET_H = 640, 480
TARGET_HFOV = 54.201

# Femto Bolt streaming (WFOV unbinned: 0.25-2.21m, square FOV)
NATIVE_DEPTH_W, NATIVE_DEPTH_H = 1024, 1024
NATIVE_DEPTH_HFOV = 90.783            # measured from calibration

# Color stream (16:9, 80.86 deg HFOV)
COLOR_W, COLOR_H = 1280, 720

# Center-crop for 54.201 deg HFOV on the DEPTH stream
_CROP_RATIO = (
    np.tan(np.radians(TARGET_HFOV / 2))
    / np.tan(np.radians(NATIVE_DEPTH_HFOV / 2))
)
CROP_W = int(round(NATIVE_DEPTH_W * _CROP_RATIO))            # ~517
CROP_H = int(round(CROP_W * TARGET_H / TARGET_W))             # ~388
assert CROP_H <= NATIVE_DEPTH_H


def build_pipeline(align_mode="none"):
    """Open Femto Bolt and start WFOV depth + 1280x720 color streams.

    align_mode:
      'none' — no alignment; depth stays 1024x1024, color 1280x720 (independent)
      'sw'   — SW AlignFilter (depth -> color frame at runtime)
      'hw'   — HW alignment (some setups suppress depth stream — try only if
               'none' works first)
    """
    ctx = ob.Context()
    ctx.set_logger_level(ob.OBLogLevel.ERROR)
    devs = ctx.query_devices()
    if devs.get_count() == 0:
        raise RuntimeError("no Orbbec device found")
    device = devs.get_device_by_index(0)

    pipeline = ob.Pipeline(device)
    depth_list = pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
    color_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)

    depth_profile = depth_list.get_video_stream_profile(
        NATIVE_DEPTH_W, NATIVE_DEPTH_H, ob.OBFormat.Y16, 15,
    )
    color_profile = color_list.get_video_stream_profile(
        COLOR_W, COLOR_H, ob.OBFormat.YUYV, 30,
    )

    config = ob.Config()
    config.enable_stream(depth_profile)
    config.enable_stream(color_profile)

    # Bolt delivers depth/color in separate framesets by default. AlignFilter
    # (sw) and HW align both need depth+color in ONE frameset, so turn on
    # frame sync whenever any alignment is requested.
    if align_mode in ("sw", "hw"):
        try:
            pipeline.enable_frame_sync()
        except Exception as e:
            print(f"enable_frame_sync failed ({e}) — align may starve")

    hw_aligned = False
    if align_mode == "hw":
        try:
            config.set_align_mode(ob.OBAlignMode.HW_MODE)
            hw_aligned = True
            print("Align: HW (depth aligned to color)")
        except Exception as e:
            print(f"HW align failed ({e}); falling back to no align")

    pipeline.start(config)

    sw_align = None
    if align_mode == "sw":
        try:
            sw_align = ob.AlignFilter(
                align_to_stream=ob.OBStreamType.COLOR_STREAM,
            )
            print("Align: SW filter (depth aligned to color at runtime)")
        except Exception as e:
            print(f"SW align filter unavailable: {e}")
    if align_mode == "none":
        print("Align: none (depth 1024x1024 / color 1280x720 independent)")

    return pipeline, sw_align, hw_aligned


def yuyv_to_bgr(yuyv_bytes, w, h):
    """Decode YUYV 4:2:2 to BGR using OpenCV."""
    arr = np.frombuffer(yuyv_bytes, dtype=np.uint8).reshape(h, w, 2)
    bgr = cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_YUYV)
    return bgr


def grab_synced(pipeline, sw_align, timeout_ms=2000, max_retries=20):
    """Wait until we get a frameset with both depth and color."""
    for _ in range(max_retries):
        fs = pipeline.wait_for_frames(timeout_ms)
        if fs is None:
            continue
        if sw_align is not None:
            try:
                fs = sw_align.process(fs)
            except Exception:
                fs = None
        if fs is None:
            continue
        df = fs.get_depth_frame()
        cf = fs.get_color_frame()
        if df is not None and cf is not None:
            return df, cf
    return None, None


def crop_and_downscale_rgb(bgr_frame):
    """Center-crop BGR to the depth's view + 4:3 aspect, downscale to 640x480."""
    h, w = bgr_frame.shape[:2]
    # After HW align, color frame IS the canvas — crop to depth FOV proxy
    # Use ratio computed from depth HFOV vs color HFOV — but if HW aligned,
    # the depth and color share intrinsics, so crop ratio applied to color
    # frame is meaningful as the depth FOV.
    # We use depth-derived crop dims (in native depth pixels) then map to
    # the color resolution.
    crop_w = int(round(w * (CROP_W / NATIVE_DEPTH_W)))
    crop_h = int(round(crop_w * TARGET_H / TARGET_W))
    if crop_h > h:
        crop_h = h
        crop_w = int(round(crop_h * TARGET_W / TARGET_H))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = bgr_frame[y0:y0 + crop_h, x0:x0 + crop_w]
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H),
                         interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255
    return rgba


def crop_and_downscale_depth(depth_units, depth_scale_mm):
    """Center-crop depth, downscale, convert units->meters, NaN for invalid."""
    h, w = depth_units.shape
    # Use proportional crop relative to whatever depth came in (HW-aligned
    # depth has color resolution; non-aligned depth is native 1024x1024).
    crop_w = int(round(w * (CROP_W / NATIVE_DEPTH_W)))
    crop_h = int(round(crop_w * TARGET_H / TARGET_W))
    if crop_h > h:
        crop_h = h
        crop_w = int(round(crop_h * TARGET_W / TARGET_H))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = depth_units[y0:y0 + crop_h, x0:x0 + crop_w]
    resized = cv2.resize(cropped, (TARGET_W, TARGET_H),
                         interpolation=cv2.INTER_NEAREST)
    # Femto Bolt depth_scale is mm/unit -> meters
    depth_m = resized.astype(np.float32) * (depth_scale_mm / 1000.0)
    depth_m[resized == 0] = np.nan
    return depth_m


def depth_to_colormap(depth_units, max_units=2000):
    """uint16 depth -> colormap for preview. Invalid (0) shown red."""
    invalid = depth_units == 0
    norm = np.clip(depth_units.astype(np.float32) / max_units, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cmap[invalid] = [0, 0, 255]
    return cmap


def _center_place(src, out_h, out_w):
    """Place src into a black out_h x out_w canvas, centered (crop or pad)."""
    out = np.zeros((out_h, out_w, src.shape[2]), dtype=src.dtype)
    sh, sw = src.shape[:2]
    ch, cw = min(sh, out_h), min(sw, out_w)
    oy, ox = (out_h - ch) // 2, (out_w - cw) // 2
    sy, sx = (sh - ch) // 2, (sw - cw) // 2
    out[oy:oy + ch, ox:ox + cw] = src[sy:sy + ch, sx:sx + cw]
    return out


def overlay_rgb_depth(rgb_bgr, depth_vis, shift_y=0, shift_x=0, scale=1.0,
                      alpha=0.45):
    """Blend depth colormap onto RGB, depth scaled then shifted.

    Preview-only aid to eyeball RGB<->depth registration without aligning the
    saved depth (saved data stays raw / --align none). The Femto Bolt RGB and
    ToF are separate optics with different native FOV + a baseline, so the raw
    streams differ in position AND scale. Tune live in GUI: i/k = up/down,
    j/l = left/right, u/o = depth bigger/smaller. shift_y > 0 moves depth DOWN,
    shift_x > 0 RIGHT, scale > 1 enlarges depth.
    """
    h, w = rgb_bgr.shape[:2]
    dv = depth_vis
    if abs(scale - 1.0) > 1e-3:
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        dv = _center_place(
            cv2.resize(depth_vis, (sw, sh), interpolation=cv2.INTER_NEAREST),
            h, w)
    shifted = np.zeros_like(dv)
    sy, sx = int(shift_y), int(shift_x)
    # source / destination ranges (clamped so out-of-canvas stays black)
    dy0, dy1 = max(sy, 0), min(h + sy, h)
    sy0, sy1 = max(-sy, 0), min(h - sy, h)
    dx0, dx1 = max(sx, 0), min(w + sx, w)
    sx0, sx1 = max(-sx, 0), min(w - sx, w)
    if dy1 > dy0 and dx1 > dx0:
        shifted[dy0:dy1, dx0:dx1] = dv[sy0:sy1, sx0:sx1]
    return cv2.addWeighted(rgb_bgr, 1.0 - alpha, shifted, alpha, 0)


def save_capture(scene_dir, version, rgba, depth_m):
    scene_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = scene_dir / f"rgb_{version}.png"
    Image.fromarray(rgba, mode="RGBA").save(str(rgb_path))
    depth_path = scene_dir / f"depth_{version}.data"
    depth_m.tofile(str(depth_path))
    print(f"  saved {rgb_path.name} + {depth_path.name}")


def detect_start_version(scene_dir):
    if not scene_dir.exists():
        return 0
    vs = []
    for p in scene_dir.glob("rgb_*.png"):
        try:
            vs.append(int(p.stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return max(vs) + 1 if vs else 0


def main():
    parser = argparse.ArgumentParser(
        description="Capture RGBD frames from Orbbec Femto Bolt for Monty"
    )
    parser.add_argument("--object", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument(
        "--output-dir",
        default=str(Path.home() / "tbp/data/worldimages/captured_scenes"),
    )
    parser.add_argument("--start-version", type=int, default=None)
    parser.add_argument("--num-captures", type=int, default=4)
    parser.add_argument("--headless", action="store_true",
                        help="Auto-capture N frames after warmup (no input)")
    parser.add_argument("--stdin", action="store_true",
                        help="Interactive without GUI: Enter=capture, q+Enter=quit. "
                             "Preview PNG updated to --preview-path. "
                             "Use when GUI is unavailable (sudo, SSH).")
    parser.add_argument("--preview-path", default="/tmp/femto_preview.png",
                        help="Where to write the live preview PNG in --stdin mode")
    parser.add_argument("--align", choices=["none", "sw", "hw"], default="none",
                        help="Depth alignment: none (default), sw (filter), "
                             "hw (HW mode — known to suppress depth on some setups)")
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--preview-only", action="store_true")
    parser.add_argument("--preview-shift-y", type=int, default=0,
                        help="Vertical px shift of depth over RGB in the "
                             "preview overlay (depth>0 = down). Preview only; "
                             "saved depth stays raw. Tune live with i/k in GUI.")
    parser.add_argument("--preview-shift-x", type=int, default=0,
                        help="Horizontal px shift of depth over RGB in the "
                             "preview overlay (depth>0 = right). Preview only. "
                             "Tune live with j/l in GUI.")
    parser.add_argument("--preview-scale", type=float, default=1.0,
                        help="Scale of depth over RGB in the preview overlay "
                             "(>1 enlarges depth). Preview only; saved depth "
                             "stays raw. Tune live with u/o in GUI.")
    args = parser.parse_args()

    scene_name = f"{args.index}_{args.object}"
    scene_dir = Path(args.output_dir) / scene_name
    version = (args.start_version if args.start_version is not None
               else detect_start_version(scene_dir))

    print(f"Scene: {scene_name}  start_version={version}")
    print(f"Output: {scene_dir}")
    print(f"Capture: depth WFOV {NATIVE_DEPTH_W}x{NATIVE_DEPTH_H} "
          f"(HFOV {NATIVE_DEPTH_HFOV:.2f}deg) -> "
          f"crop ratio {CROP_W}/{NATIVE_DEPTH_W} -> 640x480 "
          f"(HFOV {TARGET_HFOV:.2f}deg)")

    pipeline, sw_align, hw_aligned = build_pipeline(align_mode=args.align)

    try:
        # Warm-up — drain frames without requiring sync (Bolt depth/color
        # arrive in separate framesets, so grab_synced would starve here).
        print("Warming up...")
        warm_end = time.time() + args.warmup
        depth_scale_mm = None
        depth_seen = color_seen = 0
        while time.time() < warm_end:
            fs = pipeline.wait_for_frames(500)
            if fs is None:
                continue
            if sw_align is not None:
                try:
                    fs = sw_align.process(fs)
                except Exception:
                    fs = None
            if fs is None:
                continue
            df = fs.get_depth_frame()
            cf = fs.get_color_frame()
            if df is not None:
                depth_seen += 1
                if depth_scale_mm is None:
                    depth_scale_mm = df.get_depth_scale()
            if cf is not None:
                color_seen += 1
        print(f"  warmup: depth frames={depth_seen}, color frames={color_seen}")
        if depth_scale_mm is None:
            depth_scale_mm = 1.0
            print("  WARN: no depth frame in warmup — assuming 1.0 mm/unit")
        else:
            print(f"  depth scale: {depth_scale_mm} mm/unit")

        def grab_and_save(v):
            df, cf = grab_synced(pipeline, sw_align)
            if df is None or cf is None:
                print(f"  [v={v}] timeout — no synced frames")
                return False
            dh, dw = df.get_height(), df.get_width()
            cw, ch = cf.get_width(), cf.get_height()
            depth_arr = np.frombuffer(df.get_data(), dtype=np.uint16) \
                .reshape(dh, dw)
            color_bgr = yuyv_to_bgr(bytes(cf.get_data()), cw, ch)
            rgba = crop_and_downscale_rgb(color_bgr)
            depth_m = crop_and_downscale_depth(depth_arr, depth_scale_mm)
            nan_pct = float(np.isnan(depth_m).mean() * 100)
            valid = depth_m[~np.isnan(depth_m)]
            if valid.size:
                print(f"  [v={v}] depth {valid.min():.3f}-{valid.max():.3f}m  "
                      f"NaN={nan_pct:.1f}%")
            else:
                print(f"  [v={v}] depth all NaN")
            save_capture(scene_dir, v, rgba, depth_m)
            return True

        def make_preview_image(depth_arr, color_bgr):
            """Build side-by-side RGB + depth colormap (640x480 each)."""
            rgba = crop_and_downscale_rgb(color_bgr)
            preview_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
            dh, dw = depth_arr.shape
            crop_w = int(round(dw * (CROP_W / NATIVE_DEPTH_W)))
            crop_h = int(round(crop_w * TARGET_H / TARGET_W))
            if crop_h > dh:
                crop_h = dh
                crop_w = int(round(crop_h * TARGET_W / TARGET_H))
            x0 = (dw - crop_w) // 2
            y0 = (dh - crop_h) // 2
            d_crop = depth_arr[y0:y0 + crop_h, x0:x0 + crop_w]
            d_resized = cv2.resize(d_crop, (TARGET_W, TARGET_H),
                                   interpolation=cv2.INTER_NEAREST)
            depth_vis = depth_to_colormap(d_resized)
            overlay = overlay_rgb_depth(preview_bgr, depth_vis,
                                        args.preview_shift_y,
                                        args.preview_shift_x,
                                        args.preview_scale)
            cv2.putText(
                overlay,
                f"shift=({args.preview_shift_x},{args.preview_shift_y}) "
                f"scale={args.preview_scale:.2f}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return np.hstack([preview_bgr, depth_vis, overlay])

        if args.headless:
            for i in range(args.num_captures):
                print(f"Capturing {i+1}/{args.num_captures}...")
                if grab_and_save(version):
                    version += 1
                if i < args.num_captures - 1:
                    time.sleep(args.interval)
        elif args.stdin:
            # Background thread continuously grabs frames + updates preview PNG.
            # Main thread waits for stdin: Enter=capture, 'q'+Enter=quit.
            import threading

            latest = {"depth": None, "color": None}
            stop = threading.Event()
            preview_path = args.preview_path

            def grabber():
                """Update latest depth/color independently as they arrive.

                Femto Bolt depth (15fps) and color (30fps) come in different
                framesets, so requiring a synced frameset starves the queue.
                We use whatever arrived most recently — depth and color are
                temporally close enough for static-scene mug capture.
                """
                last_write = 0.0
                depth_count = color_count = 0
                last_stat = time.time()
                while not stop.is_set():
                    fs = pipeline.wait_for_frames(500)
                    if fs is None:
                        continue
                    if sw_align is not None:
                        try:
                            fs = sw_align.process(fs)
                        except Exception:
                            fs = None
                    if fs is None:
                        continue
                    df = fs.get_depth_frame()
                    cf = fs.get_color_frame()

                    if df is not None:
                        dh, dw = df.get_height(), df.get_width()
                        depth_arr = np.frombuffer(
                            df.get_data(), dtype=np.uint16,
                        ).reshape(dh, dw)
                        latest["depth"] = depth_arr
                        depth_count += 1
                    if cf is not None:
                        cw, ch = cf.get_width(), cf.get_height()
                        color_bgr = yuyv_to_bgr(
                            bytes(cf.get_data()), cw, ch,
                        )
                        latest["color"] = color_bgr
                        color_count += 1

                    # Stat line every ~2 s
                    now = time.time()
                    if now - last_stat > 2.0:
                        sys.stderr.write(
                            f"  [grabber] last 2s: depth={depth_count} "
                            f"color={color_count}\n"
                        )
                        sys.stderr.flush()
                        depth_count = color_count = 0
                        last_stat = now

                    # Throttle preview PNG write to ~3 fps
                    if (latest["depth"] is not None
                            and latest["color"] is not None
                            and now - last_write > 0.33):
                        try:
                            cv2.imwrite(
                                preview_path,
                                make_preview_image(
                                    latest["depth"], latest["color"],
                                ),
                            )
                        except Exception:
                            pass
                        last_write = now

            t = threading.Thread(target=grabber, daemon=True)
            t.start()

            print("\nStdin mode active.")
            print(f"  Preview PNG: {preview_path}  "
                  "(open in Preview.app, Cmd+R to refresh)")
            print("  Enter        = capture")
            print("  c <RET>      = capture (same as Enter)")
            print("  q <RET>      = quit\n")

            captures = 0
            while captures < args.num_captures:
                try:
                    line = input(
                        f"[{captures+1}/{args.num_captures}, "
                        f"next v={version}] >>> "
                    )
                except EOFError:
                    break
                cmd = line.strip().lower()
                if cmd in ("q", "quit", "exit"):
                    break

                d, c = latest["depth"], latest["color"]
                if d is None or c is None:
                    print("  no frame available yet — try again in a moment")
                    continue
                rgba = crop_and_downscale_rgb(c)
                depth_m = crop_and_downscale_depth(d, depth_scale_mm)
                nan_pct = float(np.isnan(depth_m).mean() * 100)
                valid = depth_m[~np.isnan(depth_m)]
                if valid.size:
                    print(f"  [v={version}] depth "
                          f"{valid.min():.3f}-{valid.max():.3f}m  "
                          f"NaN={nan_pct:.1f}%")
                else:
                    print(f"  [v={version}] depth all NaN")
                save_capture(scene_dir, version, rgba, depth_m)
                version += 1
                captures += 1

            stop.set()
            t.join(timeout=1.5)
            print(f"\nDone. {captures} captured.")
        else:
            captures = 0
            shift_y = args.preview_shift_y
            shift_x = args.preview_shift_x
            scale = args.preview_scale
            win = f"Femto Bolt | {scene_name}"
            cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
            print("Controls: 'c' = capture, 'q' = quit, "
                  "'i'/'k' = up/down, 'j'/'l' = left/right, "
                  "'u'/'o' = depth bigger/smaller")
            while True:
                df, cf = grab_synced(pipeline, sw_align, timeout_ms=500,
                                     max_retries=3)
                if df is None or cf is None:
                    continue
                dh, dw = df.get_height(), df.get_width()
                cw, ch = cf.get_width(), cf.get_height()
                depth_arr = np.frombuffer(df.get_data(), dtype=np.uint16) \
                    .reshape(dh, dw)
                color_bgr = yuyv_to_bgr(bytes(cf.get_data()), cw, ch)
                # Preview using cropped 640x480
                rgba = crop_and_downscale_rgb(color_bgr)
                preview_bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
                # Depth preview at native then crop
                crop_w = int(round(dw * (CROP_W / NATIVE_DEPTH_W)))
                crop_h = int(round(crop_w * TARGET_H / TARGET_W))
                if crop_h > dh:
                    crop_h = dh
                    crop_w = int(round(crop_h * TARGET_W / TARGET_H))
                x0 = (dw - crop_w) // 2
                y0 = (dh - crop_h) // 2
                d_crop = depth_arr[y0:y0 + crop_h, x0:x0 + crop_w]
                d_resized = cv2.resize(d_crop, (TARGET_W, TARGET_H),
                                       interpolation=cv2.INTER_NEAREST)
                depth_vis = depth_to_colormap(d_resized)
                overlay = overlay_rgb_depth(preview_bgr, depth_vis,
                                            shift_y, shift_x, scale)
                cv2.putText(overlay,
                            f"shift=({shift_x},{shift_y}) scale={scale:.2f}",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                preview = np.hstack([preview_bgr, depth_vis, overlay])
                cv2.imshow(win, preview)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("k"):
                    shift_y += 1
                    continue
                if key == ord("i"):
                    shift_y -= 1
                    continue
                if key == ord("l"):
                    shift_x += 1
                    continue
                if key == ord("j"):
                    shift_x -= 1
                    continue
                if key == ord("u"):
                    scale += 0.02
                    continue
                if key == ord("o"):
                    scale = max(0.1, scale - 0.02)
                    continue
                if key == ord("q"):
                    print(f"Done. {captures} captured "
                          f"(shift=({shift_x},{shift_y}) scale={scale:.2f}).")
                    break
                if key == ord("c"):
                    if args.preview_only:
                        continue
                    if captures >= args.num_captures:
                        print(f"Max ({args.num_captures}) reached. 'q' to quit.")
                        continue
                    if grab_and_save(version):
                        version += 1
                        captures += 1
            cv2.destroyAllWindows()
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()

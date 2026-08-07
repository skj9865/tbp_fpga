#!/usr/bin/env python3
"""Manual capture-then-infer on the board, shown in a cv2 GUI window.

Unlike v4l2_d405_live_infer_gui.py (which auto-captures and infers whenever the
isolate coverage gate passes — so it sometimes infers on a poorly-framed
object), this version only runs recognition when you press 'c'. The live
preview (RGB | depth | isolated-> Monty) updates continuously with the current
coverage so you can frame the object first, then capture on demand.

This is the V4L2/board counterpart of capture_and_infer.py (which uses
pyrealsense/depthai on the PC). All heavy lifting — V4L2 capture, warm model,
isolate, inference, rendering — is imported from v4l2_d405_live_infer so the
two stay in sync; this file only owns the window + key handling.

Requires an HDMI display + working X on the board (run from a LOCAL session,
not SSH; over SSH cv2.imshow raises "cannot initialize GTK backend", or
export DISPLAY=:0).

Controls:
  'c'      = capture current frame -> isolate -> infer -> show result
  'q'/ESC  = quit

Usage (on the board, monitor attached):
  python3 v4l2_d405_capture_infer_gui.py --objects montys_brain
  MONTY_UPDATER=default python3 v4l2_d405_capture_infer_gui.py --objects montys_brain
"""
import argparse
import tempfile
import time

import cv2

import v4l2_d405_live_infer as base
from isolate_object import isolate_depth


def main():
    p = argparse.ArgumentParser(
        description="Manual capture + Monty inference on the board (cv2 window)")
    p.add_argument("--depth-node", default="/dev/video0")
    p.add_argument("--color-node", default="/dev/video4")
    p.add_argument("--objects", default=None,
                   help="comma-separated objects to keep (fewer = faster)")
    p.add_argument("--object", default="live")
    p.add_argument("--model-path", default=None)
    p.add_argument("--hfov", type=float, default=54.201)
    p.add_argument("--depth-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-eval-steps", type=int, default=300)
    p.add_argument("--max-total-steps", type=int, default=600)
    p.add_argument("--min-cov", type=float, default=5.0,
                   help="coverage below this only warns; capture still allowed")
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument("--no-isolate", action="store_true")
    p.add_argument("--cooldown", type=float, default=1.5,
                   help="seconds to keep the result on screen after inferring")
    p.add_argument("--drain", type=float, default=0.4)
    args = p.parse_args()

    import monty_inference as mi
    if args.model_path is None:
        args.model_path = mi.default_model_path()
    mi.MAX_TOTAL_STEPS = args.max_total_steps

    keep = [o.strip() for o in args.objects.split(",")] if args.objects else None
    print("loading Monty model (warm)...")
    t0 = time.time()
    model, lm = base.build_warm_model(args.model_path, args.hfov,
                                      args.depth_clip, keep)
    print(f"model ready in {time.time()-t0:.1f}s  "
          f"x_percent={lm.x_percent_threshold}")

    ddev, dcap, diter = base.open_stream(args.depth_node, "Z16 ")
    cdev, ccap, citer = base.open_stream(args.color_node, "YUYV")
    tmp = tempfile.mkdtemp(prefix="cap_infer_gui_")

    win = "D405 capture+infer  ('c' = capture,  q/ESC = quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("window open on the board's display. 'c' = capture+infer, 'q' = quit.")

    result = None
    dt = 0.0
    n_inf = 0
    try:
        while True:
            depth = base.depth_from(next(diter))
            bgr = base.color_from(next(citer))
            if depth is None or bgr is None:
                continue

            iso, info = isolate_depth(depth)
            # Continuous live preview — shows current coverage + last result,
            # but does NOT infer until 'c' is pressed.
            frame = base.render(bgr, depth, iso, info, args.max_distance,
                                "live", result, dt, n_inf)
            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("c"):
                continue

            # --- manual capture: use a fresh current frame ---
            base.drain([diter, citer], 0.15)
            d2 = base.depth_from(next(diter))
            c2 = base.color_from(next(citer))
            if d2 is not None and c2 is not None:
                depth, bgr = d2, c2
                iso, info = isolate_depth(depth)

            cov = info.get("coverage_pct", 0.0)
            if not info.get("ok"):
                print(f"  [isolate FAILED: {info.get('reason')}] "
                      f"-> raw depth; result unreliable. Frame the object in "
                      f"the yellow box.")
            elif cov < args.min_cov:
                print(f"  [coverage {cov:.1f}% < {args.min_cov:.0f}%] object "
                      f"barely captured; result may be unreliable.")

            # freeze on the frame being recognised
            cv2.imshow(win, base.render(bgr, depth, iso, info,
                                        args.max_distance, "inferring",
                                        None, 0.0, n_inf))
            cv2.waitKey(1)

            fed = depth if args.no_isolate else iso
            dp = base.write_scene(tmp, args.object, bgr, fed)
            t = time.time()
            result = base.infer_scene(model, lm, dp, args.seed,
                                      args.max_eval_steps)
            dt = time.time() - t
            n_inf += 1
            print(f"[{n_inf}] -> {result['detected_object']}  "
                  f"{result['terminal_state']}  {result['steps']} steps  "
                  f"{dt:.1f}s  (cov={cov:.0f}%)")

            # keep the result up for cooldown, staying responsive to q/ESC
            end = time.time() + args.cooldown
            while time.time() < end:
                cv2.imshow(win, base.render(bgr, depth, iso, info,
                                            args.max_distance, "done",
                                            result, dt, n_inf))
                if (cv2.waitKey(30) & 0xFF) in (ord("q"), 27):
                    raise KeyboardInterrupt
            base.drain([diter, citer], args.drain)
    except KeyboardInterrupt:
        pass
    finally:
        dcap.close(); ddev.close()
        ccap.close(); cdev.close()
        cv2.destroyAllWindows()
        print(f"\nstopped. {n_inf} inference(s).")


if __name__ == "__main__":
    main()

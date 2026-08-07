#!/usr/bin/env python3
"""Live D405 capture + Monty inference shown in a cv2 GUI window.

GUI counterpart of v4l2_d405_live_infer.py (which serves the preview over
HTTP for a headless board). This one opens a cv2 window on a monitor wired
directly to the board — requires an HDMI display + working X (cv2 built with
GUI support). On a headless board cv2.imshow raises "cannot initialize GTK
backend"; use the web version there instead.

All the heavy lifting (V4L2 capture, warm model, isolate, inference, frame
rendering) is imported from v4l2_d405_live_infer so the two stay in sync;
this file only owns the window + key handling.

Same behaviour as the web version: sequential (A-1), freezes the window on
the frame being recognised, drains the V4L2 queue so each cycle uses the
current scene, and --max-total-steps bounds the no-observation case.

Usage (on the board, with a monitor attached):
  python3 v4l2_d405_live_infer_gui.py --objects montys_brain
  # 'q' or ESC in the window to quit
"""
import argparse
import tempfile
import time

import cv2

import v4l2_d405_live_infer as base
from isolate_object import isolate_depth


def main():
    p = argparse.ArgumentParser(description="Live D405 + Monty in a cv2 window")
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
    p.add_argument("--min-cov", type=float, default=5.0)
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument("--no-isolate", action="store_true")
    p.add_argument("--cooldown", type=float, default=1.0)
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
    tmp = tempfile.mkdtemp(prefix="live_infer_gui_")

    win = "D405 live inference  (q / ESC = quit)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("window open on the board's display. 'q' or ESC to quit.")

    def show(frame):
        cv2.imshow(win, frame)
        return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)

    result = None
    dt = 0.0
    n_inf = 0
    fps = 0.0
    prev_t = None
    try:
        while True:
            depth = base.depth_from(next(diter))
            bgr = base.color_from(next(citer))
            if depth is None or bgr is None:
                continue

            # Rolling preview FPS (EMA over the live grab->render loop).
            now = time.time()
            if prev_t is not None:
                inst = 1.0 / max(now - prev_t, 1e-6)
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst
            prev_t = now

            iso, info = isolate_depth(depth)
            live = base.render(bgr, depth, iso, info, args.max_distance,
                               "live", result, dt, n_inf)
            cv2.putText(live, f"{fps:.1f} fps", (8, live.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)
            if show(live):
                break

            if info.get("ok") and info.get("coverage_pct", 0) >= args.min_cov:
                base.drain([diter, citer], 0.15)
                d2 = base.depth_from(next(diter))
                c2 = base.color_from(next(citer))
                if d2 is not None and c2 is not None:
                    depth, bgr = d2, c2
                    iso, info = isolate_depth(depth)
                    if not (info.get("ok")
                            and info.get("coverage_pct", 0) >= args.min_cov):
                        continue

                if show(base.render(bgr, depth, iso, info, args.max_distance,
                                    "inferring", None, 0.0, n_inf)):
                    break
                fed = depth if args.no_isolate else iso
                dp = base.write_scene(tmp, args.object, bgr, fed)
                t = time.time()
                result = base.infer_scene(model, lm, dp, args.seed,
                                          args.max_eval_steps)
                dt = time.time() - t
                n_inf += 1
                print(f"[{n_inf}] -> {result['detected_object']}  "
                      f"{result['terminal_state']}  {result['steps']} steps  "
                      f"{dt:.1f}s  (cov={info.get('coverage_pct',0):.0f}%)")
                if show(base.render(bgr, depth, iso, info, args.max_distance,
                                    "done", result, dt, n_inf)):
                    break
                # keep the result up for cooldown seconds, staying responsive
                end = time.time() + args.cooldown
                while time.time() < end:
                    if show(base.render(bgr, depth, iso, info,
                                        args.max_distance, "done", result, dt,
                                        n_inf)):
                        raise KeyboardInterrupt
                base.drain([diter, citer], args.drain)
                prev_t = None  # reset FPS timer so the inference pause isn't counted
    except KeyboardInterrupt:
        pass
    finally:
        dcap.close(); ddev.close()
        ccap.close(); cdev.close()
        cv2.destroyAllWindows()
        print(f"\nstopped. {n_inf} inference(s).")


if __name__ == "__main__":
    main()

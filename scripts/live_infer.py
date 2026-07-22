#!/usr/bin/env python3
"""Continuous live inference (A-1, sequential) for the demo.

Unlike capture_and_infer.py (press 'c' to capture one frame, then infer),
this runs the capture -> isolate -> infer cycle automatically in a loop and
keeps showing what it is currently processing. Meant for testing: the point
is to *see* which frame Monty is working on and what it concludes.

Each cycle:
  1. grab the latest RGBD frame from the camera
  2. isolate the centred object
  3. run Monty on that single frame (still a full saccade episode internally)
  4. overlay the result and show RGB | depth | isolated-being-processed

Inference takes a few seconds per frame, so the preview updates once per
cycle (sequential). For a smooth 30fps preview with background inference,
that's the A-2 threaded variant (not this file).

Usage:
  conda activate tbp_fpga
  scripts/ci.sh -m live_infer --camera d405            # via wrapper (DISPLAY=:1)
  python scripts/live_infer.py --camera d405
  python scripts/live_infer.py --camera femto --align c2d
"""
import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import cv2
import numpy as np

import capture_and_infer as ci
from isolate_object import isolate_depth

TARGET_W, TARGET_H = 640, 480


def draw_result(bgr, depth_full, depth_iso, info, res, clip_m, vis_max_m,
                cycle, dt):
    """Compose the RGB | depth | isolated(-> Monty) panel with result overlay."""
    cov = info.get("coverage_pct", 0.0)
    dist = info.get("obj_dist")
    ok = info.get("ok")

    # Result banner colour: green if a confident match, yellow if MLH-only,
    # grey if nothing yet.
    perf = res["primary_performance"] if res else ""
    if perf.startswith("correct") or res and res["terminal_state"] == "match":
        rc = (90, 230, 90)
    elif perf:
        rc = (90, 220, 230)
    else:
        rc = (180, 180, 180)

    guide = bgr.copy()
    cv2.rectangle(guide, (ci.CENTER_X0, ci.CENTER_Y0),
                  (ci.CENTER_X1, ci.CENTER_Y1), (0, 255, 255), 1)
    panels = [
        ci.label_panel(guide, "RGB (put object in yellow box)"),
        ci.label_panel(ci.raw_depth_colormap(depth_full, vis_max_m),
                       f"depth 0-{vis_max_m:.1f}m (magenta=NO depth)"),
        ci.label_panel(
            ci.iso_to_colormap(depth_iso),
            f"ISOLATED -> Monty  cov={cov:.0f}%"
            + (f" dist={dist:.2f}m" if dist else "")
            + ("" if ok else f"  {info.get('reason', '')}"),
            color=(120, 255, 120) if ok else (120, 120, 255)),
    ]
    frame = np.hstack(panels)

    # Bottom result banner across the full width.
    banner = np.full((60, frame.shape[1], 3), 25, dtype=np.uint8)
    if res:
        det = str(res["detected_object"])
        txt = f"-> {det}   ({perf}, {res['steps']} steps, {dt:.1f}s)"
    else:
        txt = "inferring..."
    cv2.putText(banner, txt, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rc, 2,
                cv2.LINE_AA)
    cv2.putText(banner, f"cycle {cycle}", (frame.shape[1] - 130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    return np.vstack([frame, banner])


def run(args):
    print("Loading Monty model (warm)...")
    keep = ([o.strip() for o in args.objects.split(",")] if args.objects
            else None)
    model, lm = ci.build_warm_model(args.model_path, args.hfov, args.depth_clip,
                                    keep_objects=keep)
    clip_m = args.depth_clip if args.depth_clip is not None else 0.4
    vis_max_m = (args.max_distance if args.max_distance is not None
                 else clip_m * 2.0)
    print(f"Model ready. camera={args.camera} hfov={args.hfov} "
          f"x_percent={lm.x_percent_threshold}")
    print("Live inference running. 'q' or ESC in the window / terminal to quit.\n")

    cam = ci.CAMERAS[args.camera](args)
    tmp_base = tempfile.mkdtemp(prefix="live_infer_")
    win = "live inference | q=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    last_res = None
    last_dt = 0.0
    cycle = 0

    # Warm up the stream so the first frame isn't stale/empty.
    t_end = time.time() + args.warmup
    while time.time() < t_end:
        cam.read()
        cv2.waitKey(1)

    try:
        with ci.TermKeys() as keys:
            while cam.running():
                bgr, depth_full = cam.read()
                if bgr is None or depth_full is None:
                    if _should_quit(keys):
                        break
                    continue

                depth_iso, info = isolate_depth(depth_full)

                # Show the frame we're about to process BEFORE blocking on
                # inference, so the operator sees the live scene.
                pre = draw_result(bgr, depth_full, depth_iso, info, last_res,
                                  clip_m, vis_max_m, cycle, last_dt)
                cv2.imshow(win, pre)
                if _should_quit(keys):
                    break

                # Only spend inference time on a usable frame.
                if not args.infer_all and (not info.get("ok")
                                           or info.get("coverage_pct", 0) < args.min_cov):
                    continue

                cycle += 1
                t = time.time()
                fed = depth_full if args.no_isolate else depth_iso
                data_path = write_scene_live(tmp_base, args.object, bgr, fed)
                res = ci.infer_scene(model, lm, data_path, args.seed,
                                     args.max_eval_steps)
                last_dt = time.time() - t
                last_res = res
                print(f"[cycle {cycle}] -> {res['detected_object']}  "
                      f"{res['primary_performance']}  {res['steps']} steps  "
                      f"{last_dt:.1f}s  (cov={info.get('coverage_pct', 0):.0f}%)")

                out = draw_result(bgr, depth_full, depth_iso, info, res,
                                  clip_m, vis_max_m, cycle, last_dt)
                cv2.imshow(win, out)
                if _should_quit(keys):
                    break
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"\nDone. {cycle} inference cycle(s).")


def write_scene_live(base_dir, object_name, bgr, depth_m):
    from PIL import Image
    scene = Path(base_dir) / f"0_{object_name}"
    scene.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, np.full((TARGET_H, TARGET_W), 255, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(str(scene / "rgb_0.png"))
    depth_m.astype(np.float32).tofile(str(scene / "depth_0.data"))
    return Path(base_dir)


def _should_quit(keys):
    k = cv2.waitKey(1) & 0xFF
    tk = keys.get()
    if tk:
        k = ord(tk)
    return k in (ord("q"), 27)


def main():
    p = argparse.ArgumentParser(description="Continuous live Monty inference.")
    p.add_argument("--camera", default="d405", choices=sorted(ci.CAMERAS))
    p.add_argument("--align", default="c2d", choices=["c2d", "none", "sw", "hw"],
                   help="Femto only (default c2d).")
    p.add_argument("--rs-preset", default="none", help="D405 depth filter preset.")
    p.add_argument("--object", default="live",
                   help="Scene label; a real class name also scores correct/wrong.")
    p.add_argument("--objects", default=None,
                   help="Comma-separated object ids to keep in memory (e.g. "
                        "montys_brain,tomato_soup_can). Inference is ~linear in "
                        "object count, so limiting to the demo objects is the "
                        "biggest speedup (10->3 objs ~= 2.2x). Default: all 10.")
    p.add_argument("--model-path", default=None)
    p.add_argument("--hfov", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-eval-steps", type=int, default=300)
    p.add_argument("--depth-clip", type=float, default=None)
    p.add_argument("--no-isolate", action="store_true")
    p.add_argument("--min-cov", type=float, default=5.0,
                   help="Skip inference when isolate coverage%% is below this "
                        "(no object framed). Default 5.")
    p.add_argument("--infer-all", action="store_true",
                   help="Infer every frame regardless of coverage/quality.")
    p.add_argument("--warmup", type=float, default=1.5)
    # OAK-only pass-throughs (ignored by other cameras)
    p.add_argument("--extended-disparity", action="store_true")
    p.add_argument("--color-res", default="THE_4_K")
    p.add_argument("--preset", default="DENSITY")
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--min-distance", type=float, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--ir-brightness", type=int, default=1200)
    args = p.parse_args()

    if args.model_path is None:
        args.model_path = ci.default_model_path()
    if args.hfov is None:
        args.hfov = ci.CAMERAS[args.camera].default_hfov
    run(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Live D405 capture + Monty inference on a PetaLinux board, with web preview.

Board version of live_infer.py: the Z8 script uses depthai/pyrealsense2 and a
cv2 window, neither of which works here (pyrealsense2 streams die on
VIDIOC_QBUF, and the board is headless). So this reads the D405 V4L2 nodes
directly via v4l2py and serves the preview over HTTP.

Sequential (A-1) by design, same as the Z8 script: when an object is framed
(isolate coverage above --min-cov) it runs a full Monty episode, and the
preview *freezes on the frame being processed* so you can see exactly what
the recogniser is looking at. Result is overlaid when the episode finishes.

Only needs monty_inference.py + isolate_object.py next to it (no camera SDK).

Node numbers differ per board — vp1502: video0=depth, video4=colour;
VMK180: video1=depth, video5=colour. Defaults here are vp1502.

Usage (on the board):
  python3 v4l2_d405_live_infer.py
  python3 v4l2_d405_live_infer.py --objects montys_brain            # 1 obj = fast
  python3 v4l2_d405_live_infer.py --objects montys_brain,hot_sauce
  # then browse to http://<board-ip>:8080
"""
import argparse
import os
import sys
import tempfile
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer

os.environ.setdefault("MPLBACKEND", "Agg")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

warnings.filterwarnings("ignore")

from v4l2py import Device
from v4l2py.device import VideoCapture

import cv2
from PIL import Image
from isolate_object import isolate_depth

DEPTH_SCALE = 9.999999747378752e-05  # D405 m/unit
SATURATION = 65535
W, H = 640, 480

# isolate_depth geometry, mirrored for the framing guide
CX0, CX1 = int(W * 0.35), int(W * 0.65)
CY0, CY1 = int(H * 0.35), int(H * 0.65)


# --------------------------------------------------------------------------
# Camera (held-open V4L2 streams)
# --------------------------------------------------------------------------

def open_stream(node, pixfmt):
    dev = Device(node)
    dev.open()
    cap = VideoCapture(dev)
    cap.set_format(W, H, pixfmt)
    cap.open()
    return dev, cap, iter(cap)


def depth_from(frame):
    raw = np.frombuffer(bytes(frame), np.uint16)
    if raw.size < W * H:
        return None
    raw = raw[:W * H].reshape(H, W)
    m = raw.astype(np.float32) * DEPTH_SCALE
    m[(raw == 0) | (raw == SATURATION)] = np.nan
    return m


def color_from(frame):
    raw = np.frombuffer(bytes(frame), np.uint8)
    if raw.size < W * H * 2:
        return None
    return cv2.cvtColor(raw[:W * H * 2].reshape(H, W, 2), cv2.COLOR_YUV2BGR_YUYV)


def drain(iterators, seconds=0.4):
    """Discard frames queued while we were busy, so the next cycle sees NOW.

    Inference takes tens of seconds; the V4L2 driver keeps filling its buffer
    the whole time, so without this the next iteration would process a frame
    from when the episode *started*. Pulling for a short window empties the
    queued ones and leaves us on a fresh frame.
    """
    end = time.time() + seconds
    while time.time() < end:
        for it in iterators:
            next(it)


# --------------------------------------------------------------------------
# Monty (warm model, self-contained so no camera-SDK import is needed)
# --------------------------------------------------------------------------

def build_warm_model(model_path, hfov, depth_clip, keep_objects):
    os.environ["MONTY_HFOV"] = repr(float(hfov))
    if depth_clip is not None:
        os.environ["MONTY_DEPTH_CLIP"] = repr(float(depth_clip))

    import monty_inference as mi
    from tbp.monty.frameworks.experiments.mode import ExperimentMode

    model = mi.create_model(mi.create_sensor_modules(),
                            [mi.create_learning_module()],
                            mi.create_motor_system())
    model.set_experiment_mode(ExperimentMode.EVAL)
    mi.load_pretrained_model(model, model_path)
    lm = model.learning_modules[0]

    if keep_objects:
        gm = lm.graph_memory
        avail = set(gm.models_in_memory.keys())
        keep = set(keep_objects)
        missing = keep - avail
        if missing:
            raise ValueError(f"unknown objects {sorted(missing)}; "
                             f"available: {sorted(avail)}")
        for gid in list(gm.models_in_memory.keys()):
            if gid not in keep:
                del gm.models_in_memory[gid]
        m = getattr(lm, "graph_id_to_target", None)
        if isinstance(m, dict):
            for gid in list(m.keys()):
                if gid not in keep:
                    del m[gid]
        print(f"graph memory pruned to {len(keep)}: {sorted(keep)}")
    return model, lm


def infer_scene(model, lm, data_path, seed, max_eval_steps):
    import monty_inference as mi
    from tbp.monty.context import RuntimeContext
    from tbp.monty.frameworks.environments.embodied_data import (
        SaccadeOnImageEnvironmentInterface,
    )
    from tbp.monty.frameworks.environments.two_d_data import (
        SaccadeOnImageEnvironment,
    )
    from tbp.monty.frameworks.experiments.mode import ExperimentMode
    from tbp.monty.frameworks.experiments.seed import episode_seed

    env = SaccadeOnImageEnvironment(data_path=str(data_path))
    ei = SaccadeOnImageEnvironmentInterface(
        scenes=[0], versions=[0], env=env, motor_system=model.motor_system,
        rng=np.random.RandomState(seed), transform=None,
        experiment_mode=ExperimentMode.EVAL, seed=seed)
    ei.pre_epoch()
    rng = np.random.RandomState(episode_seed(seed, ExperimentMode.EVAL, 0))
    target = ei.primary_target
    model.pre_episode(target)
    ei.pre_episode(rng)

    ctx = RuntimeContext(rng=rng)
    step = 0
    while True:
        obs = ei.step(ctx, first=(step == 0))
        # NOTE: check_reached_max_matching_steps counts *matching* steps only.
        # When nothing lands on the object the LM never matches, so that
        # never fires and the episode runs to MAX_TOTAL_STEPS instead — this
        # is why "no_observations_yet" costs 6000 steps / ~27s while a
        # successful match finishes in ~50 steps / ~2s. Cap total steps to
        # bound the failure case.
        if model.check_reached_max_matching_steps(max_eval_steps):
            break
        if step >= mi.MAX_TOTAL_STEPS:
            model.deal_with_time_out()
            break
        if model.is_motor_only_step:
            model.pass_features_directly_to_motor_system(ctx, obs)
        else:
            model.step(ctx, obs)
        if model.is_done:
            break
        step += 1
    model.post_episode()

    detected = lm.detected_object
    terminal = lm.terminal_state
    if terminal == "time_out":
        mlh = lm.get_current_mlh()
        detected = mlh.get("graph_id")
    return {"detected_object": detected, "terminal_state": terminal,
            "steps": step}


def write_scene(base_dir, name, bgr, depth_m):
    scene = Path(base_dir) / f"0_{name}"
    scene.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, np.full((H, W), 255, np.uint8)])
    Image.fromarray(rgba, "RGBA").save(str(scene / "rgb_0.png"))
    depth_m.astype(np.float32).tofile(str(scene / "depth_0.data"))
    return Path(base_dir)


# --------------------------------------------------------------------------
# Preview rendering
# --------------------------------------------------------------------------

def depth_colormap(depth_m, max_m):
    valid = ~np.isnan(depth_m)
    norm = np.clip(np.where(valid, depth_m, max_m) / max_m, 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm[~valid] = (40, 40, 40)
    return cm


def iso_colormap(iso):
    valid = ~np.isnan(iso)
    if valid.any():
        lo = float(np.nanmin(iso)); hi = float(np.nanmax(iso))
    else:
        lo, hi = 0.0, 1.0
    if hi <= lo:
        hi = lo + 1e-3
    norm = np.clip((np.where(valid, iso, lo) - lo) / (hi - lo), 0, 1)
    cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cm[~valid] = (35, 35, 35)
    return cm


def txt(img, s, org, color=(240, 240, 240), scale=0.6, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def render(bgr, depth, iso, info, max_m, status, result, dt, n_inf):
    guide = bgr.copy()
    cv2.rectangle(guide, (CX0, CY0), (CX1, CY1), (0, 255, 255), 1)
    frame = np.hstack([guide, depth_colormap(depth, max_m), iso_colormap(iso)])

    txt(frame, "RGB (object in box)", (10, 26))
    txt(frame, f"depth 0-{max_m:.1f}m", (W + 10, 26))
    cov = info.get("coverage_pct", 0.0)
    dist = info.get("obj_dist")
    ok = info.get("ok")
    txt(frame, f"ISOLATED -> Monty  cov={cov:.0f}%"
               + (f" dist={dist:.2f}m" if dist else ""),
        (2 * W + 10, 26), color=(120, 255, 120) if ok else (120, 120, 255))

    bar = np.full((70, frame.shape[1], 3), 25, np.uint8)
    if status == "inferring":
        txt(bar, "INFERRING...  (preview frozen on this frame)", (12, 30),
            color=(90, 220, 255), scale=0.8)
    elif result:
        txt(bar, f"-> {result['detected_object']}   "
                 f"({result['terminal_state']}, {result['steps']} steps, "
                 f"{dt:.1f}s)", (12, 30), color=(90, 230, 90), scale=0.8)
    else:
        txt(bar, "waiting for an object in the yellow box...", (12, 30),
            color=(180, 180, 180), scale=0.7)
    txt(bar, f"inferences: {n_inf}", (frame.shape[1] - 220, 58),
        color=(150, 150, 150), scale=0.55, thick=1)
    return np.vstack([frame, bar])


# --------------------------------------------------------------------------
# MJPEG server
# --------------------------------------------------------------------------

class Latest:
    def __init__(self):
        self.jpg = None
        self.lock = threading.Lock()

    def set(self, b):
        with self.lock:
            self.jpg = b

    def get(self):
        with self.lock:
            return self.jpg


def make_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='margin:0;background:#111'>"
                    b"<img src='/stream' style='width:100%'></body></html>")
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                j = latest.get()
                if j is not None:
                    try:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(j)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        return
                time.sleep(0.03)
    return Handler


def main():
    p = argparse.ArgumentParser(description="Live D405 capture + Monty (board)")
    p.add_argument("--depth-node", default="/dev/video0")
    p.add_argument("--color-node", default="/dev/video4")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--objects", default=None,
                   help="comma-separated objects to keep in memory "
                        "(fewer = much faster; inference cost is ~linear)")
    p.add_argument("--object", default="live",
                   help="scene label; a real class name also scores the result")
    p.add_argument("--model-path", default=None)
    p.add_argument("--hfov", type=float, default=54.201)
    p.add_argument("--depth-clip", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-eval-steps", type=int, default=300,
                   help="cap on MATCHING steps (only fires once the LM is "
                        "actually matching; does not bound the no-observation "
                        "case — use --max-total-steps for that)")
    p.add_argument("--max-total-steps", type=int, default=600,
                   help="hard cap on episode steps. monty_inference's default "
                        "is 6000, which makes a failed 'no_observations_yet' "
                        "episode take ~27s vs ~2s for a match. 600 bounds it "
                        "to a few seconds; successes need only ~50.")
    p.add_argument("--min-cov", type=float, default=5.0,
                   help="skip inference below this isolate coverage %%")
    p.add_argument("--max-distance", type=float, default=1.0)
    p.add_argument("--no-isolate", action="store_true")
    p.add_argument("--cooldown", type=float, default=1.0,
                   help="seconds to keep showing a result before resuming")
    p.add_argument("--drain", type=float, default=0.4,
                   help="seconds spent flushing frames queued during "
                        "inference, so the next cycle uses a current frame")
    args = p.parse_args()

    import monty_inference as mi
    if args.model_path is None:
        args.model_path = mi.default_model_path()
    # infer_scene reads mi.MAX_TOTAL_STEPS; override it so a failed episode
    # doesn't burn 6000 steps (~27s) before giving up.
    mi.MAX_TOTAL_STEPS = args.max_total_steps

    keep = [o.strip() for o in args.objects.split(",")] if args.objects else None
    print("loading Monty model (warm)...")
    t0 = time.time()
    model, lm = build_warm_model(args.model_path, args.hfov, args.depth_clip,
                                 keep)
    print(f"model ready in {time.time()-t0:.1f}s  "
          f"x_percent={lm.x_percent_threshold}")

    latest = Latest()
    ThreadingTCPServer.allow_reuse_address = True
    srv = ThreadingTCPServer(("0.0.0.0", args.port), make_handler(latest))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"preview: http://0.0.0.0:{args.port}")

    ddev, dcap, diter = open_stream(args.depth_node, "Z16 ")
    cdev, ccap, citer = open_stream(args.color_node, "YUYV")
    tmp = tempfile.mkdtemp(prefix="live_infer_")
    print("streaming. Ctrl+C to stop.")

    def publish(frame):
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            latest.set(enc.tobytes())

    result = None
    dt = 0.0
    n_inf = 0
    try:
        while True:
            depth = depth_from(next(diter))
            bgr = color_from(next(citer))
            if depth is None or bgr is None:
                continue

            iso, info = isolate_depth(depth)
            publish(render(bgr, depth, iso, info, args.max_distance,
                           "live", result, dt, n_inf))

            if info.get("ok") and info.get("coverage_pct", 0) >= args.min_cov:
                # Re-grab so we infer on the scene as it is NOW, not on a
                # frame that sat in the queue while we rendered/gated.
                drain([diter, citer], 0.15)
                d2 = depth_from(next(diter))
                c2 = color_from(next(citer))
                if d2 is not None and c2 is not None:
                    depth, bgr = d2, c2
                    iso, info = isolate_depth(depth)
                    if not (info.get("ok")
                            and info.get("coverage_pct", 0) >= args.min_cov):
                        continue   # object left the frame in the meantime

                # freeze the preview on this frame while Monty runs
                publish(render(bgr, depth, iso, info, args.max_distance,
                               "inferring", None, 0.0, n_inf))
                fed = depth if args.no_isolate else iso
                dp = write_scene(tmp, args.object, bgr, fed)
                t = time.time()
                result = infer_scene(model, lm, dp, args.seed,
                                     args.max_eval_steps)
                dt = time.time() - t
                n_inf += 1
                print(f"[{n_inf}] -> {result['detected_object']}  "
                      f"{result['terminal_state']}  {result['steps']} steps  "
                      f"{dt:.1f}s  (cov={info.get('coverage_pct',0):.0f}%)")
                publish(render(bgr, depth, iso, info, args.max_distance,
                               "done", result, dt, n_inf))
                time.sleep(args.cooldown)
                # Drop everything the driver queued during the episode so the
                # next cycle starts from the current scene, not a stale frame.
                drain([diter, citer], args.drain)
    finally:
        dcap.close(); ddev.close()
        ccap.close(); cdev.close()
        print(f"\nstopped. {n_inf} inference(s).")


if __name__ == "__main__":
    main()

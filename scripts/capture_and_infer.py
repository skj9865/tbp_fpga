#!/usr/bin/env python3
"""Live capture + inference in one process (OAK-D Pro, Z8).

Captures RGBD from the OAK-D Pro, isolates the centered object in-memory,
and runs Monty recognition immediately — no separate capture / isolate /
inference steps, no disk round-trip between programs. The Monty model is
loaded once and kept warm, so each capture returns a result in a few
seconds instead of re-loading the pretrained model every time.

Requires depthai in the same env as tbp.monty (installed into `tbp_fpga`;
OAK needs only depthai, not pyorbbecsdk/pyrealsense).

Controls (interactive window):
  'c' = capture current frame -> isolate -> infer -> print result
  'q' = quit

Usage:
  conda activate tbp_fpga
  python scripts/capture_and_infer.py                     # object unknown
  python scripts/capture_and_infer.py --object numenta_mug  # also score correct/wrong
  python scripts/capture_and_infer.py --save-dir ~/tbp/data/worldimages/captured_scenes_oak
"""
import argparse
import os
import select
import sys
import tempfile
import termios
import time
import tty
from pathlib import Path

# Must be set before tbp.monty / matplotlib import (two_d_data imports plt).
os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import cv2
import depthai as dai
import numpy as np
from PIL import Image

# OAK capture helpers (pipeline + conversions) reused verbatim.
from rgbd_capture import (
    build_pipeline,
    convert_depth,
    convert_rgb,
    depth_to_colormap,
)
from isolate_object import isolate_depth, make_debug_image

# Monty inference is imported lazily inside build_warm_model() so that a
# --help / arg error doesn't pay the multi-second torch+monty import cost.


TARGET_W, TARGET_H = 640, 480
OAKD_HFOV = 63.75  # OAK-D Pro at 640x480 preview (see rgbd_capture.py).


# isolate_depth's geometry, mirrored here to draw framing guides.
# center_roi_frac=0.30 (object-distance estimate), bbox_frac=0.80 (hard clip).
CENTER_X0, CENTER_X1 = int(TARGET_W * 0.35), int(TARGET_W * 0.65)
CENTER_Y0, CENTER_Y1 = int(TARGET_H * 0.35), int(TARGET_H * 0.65)
BBOX_X0, BBOX_X1 = int(TARGET_W * 0.10), int(TARGET_W * 0.90)
BBOX_Y0, BBOX_Y1 = int(TARGET_H * 0.10), int(TARGET_H * 0.90)

# Isolate coverage (% of frame kept) observed in the training-matched iPad
# data is 13-22%. OAK captures of glossy objects land at 0.6-2.4% because the
# object surface itself loses ~40% of its depth, so isolate can't grow a
# connected blob. Treat coverage as the live capture-quality gauge.
COV_GOOD = 10.0
COV_WEAK = 3.0


def iso_to_colormap(depth_m, max_m=1.0):
    """Color-map isolated depth (float32 meters, NaN outside object).

    NaN -> dark, so the kept object stands out as the only colored region.
    This is what Monty actually receives, so it's the panel to watch when
    tuning recognition.
    """
    valid = ~np.isnan(depth_m)
    if valid.any():
        vmin = float(np.nanmin(depth_m))
        vmax = min(float(np.nanmax(depth_m)), max_m)
    else:
        vmin, vmax = 0.0, max_m
    if vmax <= vmin:
        vmax = vmin + 1e-3
    norm = np.clip((np.where(valid, depth_m, vmin) - vmin) / (vmax - vmin), 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cmap[~valid] = (35, 35, 35)  # masked-out = dark grey
    return cmap


def raw_depth_colormap(depth_m, max_m):
    """Color-map raw depth for the preview.

    rgbd_capture.depth_to_colormap paints invalid pixels pure red, which is
    also where JET's far end lands — so "no data" and "far away" become
    indistinguishable and the background reads as one red mass. Here invalid
    is a distinct magenta and the range is tied to the working distance, so
    the depth panel actually resembles the RGB scene.
    """
    invalid = np.isnan(depth_m)
    norm = np.clip(np.where(invalid, max_m, depth_m) / max_m, 0, 1)
    cmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cmap[invalid] = (200, 0, 200)  # magenta = no depth (distinct from far)
    return cmap


class TermKeys:
    """Non-blocking single-key reader on the controlling terminal.

    cv2.waitKey only receives keys when the preview window has focus, which is
    on whichever display the window opened on. Running over SSH with
    DISPLAY=:1 puts the window on the machine's physical monitor, so without
    this the operator would have to walk over to press 'c'. Reading stdin too
    lets the same keys be typed in the SSH session.
    """

    def __enter__(self):
        self.enabled = sys.stdin.isatty()
        if self.enabled:
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        return self

    def get(self):
        if self.enabled and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def __exit__(self, *exc):
        if self.enabled:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def label_panel(img, text, color=(220, 220, 220)):
    """Prepend a small caption band above a panel."""
    band = np.full((26, img.shape[1], 3), 30, dtype=np.uint8)
    cv2.putText(band, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA)
    return np.vstack([band, img])


# ---------------------------------------------------------------------------
# Camera backends — each yields (bgr 640x480, depth_m 640x480 float32/NaN)
# ---------------------------------------------------------------------------

class OakCamera:
    """OAK-D Pro via depthai. Depth is HW-aligned to the colour camera."""

    name = "oak"
    default_hfov = OAKD_HFOV

    def __init__(self, args):
        self.pipeline, rgb_out, depth_out = build_pipeline(
            preset=args.preset,
            max_distance_m=args.max_distance,
            min_distance_m=args.min_distance,
            fps=args.fps,
            extended_disparity=args.extended_disparity,
            color_resolution=args.color_res,
        )
        self.q_rgb = rgb_out.createOutputQueue()
        self.q_depth = depth_out.createOutputQueue()
        self.pipeline.start()
        try:
            dev = self.pipeline.getDefaultDevice()
            if dev is not None and hasattr(dev, "setIrLaserDotProjectorIntensity"):
                dev.setIrLaserDotProjectorIntensity(args.ir_brightness / 1200.0)
        except Exception as e:
            print(f"IR control failed: {e}")
        self._bgr = None
        self._depth_m = None

    def running(self):
        return self.pipeline.isRunning()

    def read(self):
        in_rgb = self.q_rgb.tryGet()
        in_depth = self.q_depth.tryGet()
        if in_rgb is not None:
            self._bgr = in_rgb.getCvFrame()
        if in_depth is not None:
            self._depth_m = convert_depth(in_depth.getFrame())
        return self._bgr, self._depth_m

    def close(self):
        self.pipeline.stop()


class FemtoCamera:
    """Femto Bolt (ToF) via pyorbbecsdk.

    Depth and colour are separate streams here (no HW align), so the capture
    script's centre-crop maps the native 1024x1024 / 90.78 deg depth onto the
    54.201 deg training FOV — the same value the iPad data used, which is why
    this path keeps the default --hfov.
    """

    name = "femto"
    default_hfov = 54.201

    def __init__(self, args):
        import rgbd_capture_femto as fem
        import pyorbbecsdk as ob
        self._fem = fem
        self._ob = ob
        self.mode = args.align
        self.depth_scale_mm = None

        if self.mode != "c2d":
            self.pipeline, self.sw_align, self.hw_aligned = fem.build_pipeline(
                align_mode=self.mode
            )
            self.align = None
            return

        # Colour-to-depth: warp COLOUR into the depth frame, leaving depth
        # untouched. The opposite direction (--align sw, depth->colour) was
        # measured worse because it resamples the depth that carries pose,
        # curvature and 3D location — 72% of the feature weight plus the
        # 0.015m location tolerance. hsv is only a single centre pixel but
        # carries 29% of the weight, and with the sensors unaligned it reads
        # ~100px off the object, i.e. systematically wrong at every step.
        # C2D needs an RGB (not YUYV) colour stream: YUYV raises
        # "Unsupported format for C2D conversion".
        ctx = ob.Context()
        dev = ctx.query_devices().get_device_by_index(0)
        self.pipeline = ob.Pipeline(dev)
        dl = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
        cl = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        dprof = next(
            p for p in (dl.get_stream_profile_by_index(i).as_video_stream_profile()
                        for i in range(dl.get_count()))
            if (p.get_width(), p.get_height()) == (1024, 1024)
        )
        cprof = next(
            p for p in (cl.get_stream_profile_by_index(i).as_video_stream_profile()
                        for i in range(cl.get_count()))
            if str(p.get_format()) == "OBFormat.RGB"
            and (p.get_width(), p.get_height()) == (1280, 960)
        )
        cfg = ob.Config()
        cfg.enable_stream(dprof)
        cfg.enable_stream(cprof)
        self.pipeline.start(cfg)
        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            print(f"enable_frame_sync failed ({e})")
        self.align = ob.AlignFilter(align_to_stream=ob.OBStreamType.DEPTH_STREAM)
        self.sw_align = None
        print("Align: C2D (colour warped into the depth frame; depth untouched)")

    def running(self):
        return True

    def read(self):
        if self.mode == "c2d":
            fs = self.pipeline.wait_for_frames(500)
            if fs is None:
                return None, None
            fs = self.align.process(fs)
            if fs is None:
                return None, None
            df, cf = fs.get_depth_frame(), fs.get_color_frame()
            if df is None or cf is None:
                return None, None
            if self.depth_scale_mm is None:
                self.depth_scale_mm = df.get_depth_scale()
            dh, dw = df.get_height(), df.get_width()
            depth_arr = np.frombuffer(df.get_data(), np.uint16).reshape(dh, dw)
            colour = np.frombuffer(cf.get_data(), np.uint8).reshape(
                cf.get_height(), cf.get_width(), 3
            )[:, :, ::-1]
            depth_m = self._fem.crop_and_downscale_depth(depth_arr,
                                                         self.depth_scale_mm)
            # Colour now shares the depth frame, so crop it with the same
            # geometry rather than the colour-FOV crop used when unaligned.
            fem = self._fem
            cw2 = int(round(dw * (fem.CROP_W / fem.NATIVE_DEPTH_W)))
            ch2 = int(round(cw2 * TARGET_H / TARGET_W))
            x0, y0 = (dw - cw2) // 2, (dh - ch2) // 2
            bgr = cv2.resize(colour[y0:y0 + ch2, x0:x0 + cw2],
                             (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(bgr), depth_m

        df, cf = self._fem.grab_synced(self.pipeline, self.sw_align,
                                       timeout_ms=500, max_retries=2)
        if df is None or cf is None:
            return None, None
        if self.depth_scale_mm is None:
            self.depth_scale_mm = df.get_depth_scale()
        dh, dw = df.get_height(), df.get_width()
        cw, ch = cf.get_width(), cf.get_height()
        depth_arr = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
        color_bgr = self._fem.yuyv_to_bgr(bytes(cf.get_data()), cw, ch)
        rgba = self._fem.crop_and_downscale_rgb(color_bgr)
        depth_m = self._fem.crop_and_downscale_depth(depth_arr,
                                                     self.depth_scale_mm)
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        return bgr, depth_m

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


class D405Camera:
    """Intel RealSense D405 via pyrealsense2.

    Passive stereo (no IR projector), and its colour is derived from the left
    depth imager, so RGB and depth are natively co-registered — the alignment
    problem that needed C2D on the Femto does not arise here. Weak on
    untextured/glossy surfaces for the same reason OAK was: no projected
    texture to match on.
    """

    name = "d405"
    default_hfov = 54.201  # centre-cropped from the native 89.277 deg

    def __init__(self, args):
        import rgbd_capture_d405 as d405
        self._d = d405
        self.pipeline, self.align, self.depth_scale = d405.build_pipeline()
        self.filters = (d405.build_filters(args.rs_preset)
                        if args.rs_preset != "none" else None)
        if self.filters is not None:
            print(f"RealSense depth filters: {args.rs_preset}")

    def running(self):
        return True

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except Exception:
            return None, None
        frames = self.align.process(frames)
        df = frames.get_depth_frame()
        cf = frames.get_color_frame()
        if not df or not cf:
            return None, None
        if self.filters is not None:
            df = self._d.apply_depth_filters(df, self.filters)
        depth_units = np.asanyarray(df.get_data())
        bgr_full = np.asanyarray(cf.get_data())
        depth_m = self._d.crop_and_downscale_depth(depth_units, self.depth_scale)
        rgba = self._d.crop_and_downscale_rgb(bgr_full)
        bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(bgr), depth_m

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


CAMERAS = {"oak": OakCamera, "femto": FemtoCamera, "d405": D405Camera}


# ---------------------------------------------------------------------------
# Warm Monty model
# ---------------------------------------------------------------------------

def build_warm_model(model_path, hfov, depth_clip=None, keep_objects=None):
    """Build the Monty model once and load pretrained weights (kept warm).

    Sets MONTY_HFOV before any environment is created so two_d_data's
    DepthTo3DLocations projects with the OAK-D Pro FOV, and optionally
    MONTY_DEPTH_CLIP, the threshold beyond which process_depth_data marks
    pixels off-object (default 0.4m).

    keep_objects: if given, prune the pretrained graph memory down to these
    object ids. Inference cost is ~linear in the number of objects (measured:
    10 objs 13.9s/frame, 4 objs 7.3s, 3 objs 6.2s, 1 obj 0.65s), so loading
    only the demo objects is the biggest speed lever with no accuracy loss.
    """
    os.environ["MONTY_HFOV"] = repr(float(hfov))
    if depth_clip is not None:
        os.environ["MONTY_DEPTH_CLIP"] = repr(float(depth_clip))

    import monty_inference as mi
    from tbp.monty.frameworks.experiments.mode import ExperimentMode

    sensor_modules = mi.create_sensor_modules()
    lm = mi.create_learning_module()
    motor_system = mi.create_motor_system()
    model = mi.create_model(sensor_modules, [lm], motor_system)
    model.set_experiment_mode(ExperimentMode.EVAL)
    mi.load_pretrained_model(model, model_path)
    if keep_objects:
        prune_graph_memory(lm, keep_objects)
    return model, lm


def prune_graph_memory(lm, keep_objects):
    """Drop all but keep_objects from the LM's graph memory (in place).

    Speeds up inference (cost is ~linear in object count). Raises if any
    requested object isn't in the pretrained model.
    """
    gm = lm.graph_memory
    available = set(gm.models_in_memory.keys())
    keep = set(keep_objects)
    missing = keep - available
    if missing:
        raise ValueError(
            f"objects not in model: {sorted(missing)}. "
            f"available: {sorted(available)}"
        )
    for gid in list(gm.models_in_memory.keys()):
        if gid not in keep:
            del gm.models_in_memory[gid]
    # Keep the graph_id -> target-label map consistent with what remains.
    m = getattr(lm, "graph_id_to_target", None)
    if isinstance(m, dict):
        for gid in list(m.keys()):
            if gid not in keep:
                del m[gid]
    print(f"Graph memory pruned to {len(keep)} object(s): {sorted(keep)}")


def infer_scene(model, lm, data_path, seed, max_eval_steps):
    """Run one episode over the single scene at data_path (scene 0, version 0).

    Mirrors monty_inference.run_inference's per-episode loop and scoring,
    but reuses the already-loaded (warm) model.
    """
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
    rng = np.random.RandomState(seed)
    env_interface = SaccadeOnImageEnvironmentInterface(
        scenes=[0],
        versions=[0],
        env=env,
        motor_system=model.motor_system,
        rng=rng,
        transform=None,
        experiment_mode=ExperimentMode.EVAL,
        seed=seed,
    )
    env_interface.pre_epoch()

    ep_seed = episode_seed(seed, ExperimentMode.EVAL, 0)
    rng = np.random.RandomState(ep_seed)

    target = env_interface.primary_target
    model.pre_episode(target)
    env_interface.pre_episode(rng)

    ctx = RuntimeContext(rng=rng)
    step = 0
    t0 = time.time()
    while True:
        observations = env_interface.step(ctx, first=(step == 0))
        if model.check_reached_max_matching_steps(max_eval_steps):
            break
        if step >= mi.MAX_TOTAL_STEPS:
            model.deal_with_time_out()
            break
        if model.is_motor_only_step:
            model.pass_features_directly_to_motor_system(ctx, observations)
        else:
            model.step(ctx, observations)
        if model.is_done:
            break
        step += 1

    model.post_episode()

    # Scoring — mirror monty_inference.run_inference (lines ~319-348).
    target_object = target["object"] if target else "unknown"
    terminal_state = lm.terminal_state
    detected_object = lm.detected_object
    possible_matches = lm.get_possible_matches()

    primary_performance = terminal_state
    if terminal_state == "match" and detected_object is not None:
        target_to_graph = lm.graph_id_to_target.get(detected_object, set())
        primary_performance = (
            "correct" if target_object in target_to_graph else "confused"
        )
    elif terminal_state == "time_out":
        if len(possible_matches) == 1:
            primary_performance = "pose_time_out"
        mlh = lm.get_current_mlh()
        mlh_graph_id = mlh.get("graph_id")
        if mlh_graph_id is not None:
            target_to_graph = lm.graph_id_to_target.get(mlh_graph_id, set())
            primary_performance = (
                "correct_mlh" if target_object in target_to_graph
                else "confused_mlh"
            )
        detected_object = mlh_graph_id

    return dict(
        target_object=target_object,
        detected_object=detected_object,
        terminal_state=terminal_state,
        primary_performance=primary_performance,
        steps=step,
        elapsed_sec=round(time.time() - t0, 1),
    )


# ---------------------------------------------------------------------------
# Scene writing
# ---------------------------------------------------------------------------

def write_scene(base_dir, object_name, rgba, depth_m, version=0):
    """Write a single (rgb, depth) pair as a SaccadeOnImage scene dir.

    Returns the scene's *base* dir (parent of the `0_<object>` dir), which is
    what SaccadeOnImageEnvironment expects as data_path.
    """
    scene_dir = Path(base_dir) / f"0_{object_name}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(str(scene_dir / f"rgb_{version}.png"))
    depth_m.astype(np.float32).tofile(str(scene_dir / f"depth_{version}.data"))
    return Path(base_dir)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def default_model_path():
    import monty_inference as mi
    return mi.default_model_path()


def run(args):
    print("Loading Monty model (warm)...")
    keep = ([o.strip() for o in args.objects.split(",")] if args.objects
            else None)
    model, lm = build_warm_model(args.model_path, args.hfov, args.depth_clip,
                                 keep_objects=keep)
    clip = args.depth_clip if args.depth_clip is not None else 0.4
    print(f"Model ready. HFOV={args.hfov} deg  depth_clip={clip}m  "
          f"depth={'RAW' if args.no_isolate else 'ISOLATED'}")
    print(f"Place the object at ~0.2m (training data: 0.17-0.26m). "
          f"Controls: 'c'=capture+infer 'i'/'d'=panels 'q'=quit\n")

    cam = CAMERAS[args.camera](args)
    print(f"Camera: {cam.name}")

    clip_m = args.depth_clip if args.depth_clip is not None else 0.4
    # Scale the depth preview to the working distance, not the 4m default —
    # at 4m a 0.2m object occupies 5% of the colour range and looks flat.
    vis_max_m = (args.max_distance if args.max_distance is not None
                 else clip_m * 2.0)

    tmp_base = tempfile.mkdtemp(prefix="cap_infer_")
    save_version = 0

    bgr_frame = None
    depth_full = None
    n_infer = 0

    win = "capture+infer | 'c'=infer  'i'=iso panel  'd'=debug  'q'=quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    dbg_win = "isolate debug (last capture)"
    last_result = ""
    show_iso = True      # live isolate panel (what Monty will see)
    show_debug = True    # pop the 4-panel debug view after each capture

    try:
      with TermKeys() as keys:
        while cam.running():
            bgr, dm = cam.read()
            if bgr is not None:
                bgr_frame = bgr
            if dm is not None:
                depth_full = dm

            depth_m = None
            depth_iso = None
            info = {}
            if bgr_frame is not None and depth_full is not None:
                # Guide boxes: isolate_depth estimates object distance from the
                # central 30% ROI and discards anything outside the centered 80%
                # bbox, so framing against these directly drives isolate quality.
                guide = bgr_frame.copy()
                cv2.rectangle(guide, (CENTER_X0, CENTER_Y0),
                              (CENTER_X1, CENTER_Y1), (0, 255, 255), 1)
                cv2.rectangle(guide, (BBOX_X0, BBOX_Y0), (BBOX_X1, BBOX_Y1),
                              (90, 90, 90), 1)
                panels = [
                    label_panel(guide, "RGB  (yellow=put object here)"),
                    label_panel(raw_depth_colormap(depth_full, vis_max_m),
                                f"depth raw{' -> Monty' if args.no_isolate else ''}"
                                f"  0-{vis_max_m:.1f}m  (magenta = NO depth)"),
                ]
                # Isolate live so the object can be framed correctly BEFORE
                # capturing — this panel is exactly what Monty receives.
                if show_iso:
                    depth_m = depth_full
                    depth_iso, info = isolate_depth(depth_m)
                    cov = info.get("coverage_pct", 0.0)
                    ok = info.get("ok")
                    # Center-ROI depth validity: the object surface itself. OAK
                    # active stereo drops to ~55-60% here on glossy/white
                    # objects (vs 89-97% in the training-matched iPad data),
                    # which is what starves isolate of a connected blob.
                    roi = depth_m[CENTER_Y0:CENTER_Y1, CENTER_X0:CENTER_X1]
                    roi_valid = float((~np.isnan(roi)).mean() * 100)
                    if not ok:
                        cov_color, verdict = (90, 90, 255), "FAIL"
                    elif cov >= COV_GOOD:
                        cov_color, verdict = (120, 255, 120), "good"
                    elif cov >= COV_WEAK:
                        cov_color, verdict = (120, 255, 255), "weak"
                    else:
                        cov_color, verdict = (90, 90, 255), "TOO LOW"
                    # Distance is the dominant lever: training data sits at
                    # 0.17-0.26m, and Monty marks anything beyond depth_clip
                    # (0.4m) off-object, so a far object is silently truncated.
                    dist = info.get("obj_dist")
                    if dist is None:
                        dist_s = "dist=?"
                    else:
                        # A reading far beyond the working volume means the
                        # centre has no object depth at all — typically the
                        # object sits closer than the stereo minimum (~35cm,
                        # ~17cm with --extended-disparity), so only background
                        # pixels stay valid and the median jumps to metres.
                        flag = ("TOO CLOSE? no depth on object"
                                if dist > max(1.5, clip_m * 2)
                                else "OK" if dist <= clip_m * 0.75
                                else "NEAR CLIP" if dist <= clip_m
                                else "BEYOND CLIP")
                        dist_s = f"dist={dist:.2f}m [{flag}]"
                    # Only label this "-> Monty" when isolate is actually the
                    # depth being fed; with --no-isolate Monty gets raw.
                    feeds = "not fed (--no-isolate)" if args.no_isolate else "-> Monty"
                    panels.append(label_panel(
                        iso_to_colormap(depth_iso),
                        f"ISOLATED {feeds}  {dist_s} (aim 0.20)  "
                        f"cov={cov:.1f}% [{verdict}] target {COV_GOOD:.0f}-22%  "
                        f"surf={roi_valid:.0f}%"
                        + ("" if ok else f"  {info.get('reason')}"),
                        color=cov_color,
                    ))
                preview = np.hstack(panels)
                if last_result:
                    cv2.putText(preview, last_result, (8, preview.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                                cv2.LINE_AA)
                cv2.imshow(win, preview)

            # Accept keys from the preview window OR the terminal, so the
            # session stays drivable over SSH (see TermKeys).
            key = cv2.waitKey(1) & 0xFF
            k = keys.get()
            if k:
                key = ord(k)
            if key == ord("q"):
                break
            if key == ord("i"):
                show_iso = not show_iso
                print(f"  [live isolate panel {'ON' if show_iso else 'OFF'}]")
                continue
            if key == ord("d"):
                show_debug = not show_debug
                print(f"  [capture debug window {'ON' if show_debug else 'OFF'}]")
                continue
            if key == ord("c"):
                if bgr_frame is None or depth_full is None:
                    print("  [no frames yet]")
                    continue

                rgba = convert_rgb(bgr_frame)
                # Reuse the live-panel isolate when available, else compute now.
                if depth_iso is None:
                    depth_m = depth_full
                    depth_iso, info = isolate_depth(depth_m)
                cov = info.get("coverage_pct", 0.0)
                if not info.get("ok"):
                    # isolate_depth returns the RAW depth unchanged on failure,
                    # so Monty would get hand+background — the known 0% case.
                    print(f"  [isolate FAILED: {info.get('reason')}] "
                          f"-> raw depth passed through; result unreliable. "
                          f"Center the object in the yellow box.")
                elif cov < COV_WEAK:
                    print(f"  [coverage {cov:.1f}% TOO LOW (target "
                          f"{COV_GOOD:.0f}-22%)] object surface depth is too "
                          f"holey; move closer / adjust angle / lighting.")

                if show_debug:
                    dbg = make_debug_image(bgr_frame, depth_m, depth_iso, info,
                                           label=args.object)
                    if dbg.shape[1] > 1800:
                        s = 1800 / dbg.shape[1]
                        dbg = cv2.resize(dbg, None, fx=s, fy=s,
                                         interpolation=cv2.INTER_AREA)
                    cv2.imshow(dbg_win, dbg)
                    cv2.waitKey(1)

                fed = depth_m if args.no_isolate else depth_iso
                data_path = write_scene(tmp_base, args.object, rgba, fed)
                if args.save_dir:
                    from rgbd_capture import save_capture
                    save_capture(Path(args.save_dir) / f"0_{args.object}",
                                 save_version, rgba, fed)
                    save_version += 1

                n_infer += 1
                print(f"\n[capture {n_infer}] inferring "
                      f"(coverage={info.get('coverage_pct', 0):.1f}%)...")
                res = infer_scene(model, lm, data_path, args.seed,
                                  args.max_eval_steps)
                last_result = (f"-> {res['detected_object']} "
                               f"({res['primary_performance']})")
                print(f"  DETECTED: {res['detected_object']}   "
                      f"state={res['terminal_state']}  "
                      f"perf={res['primary_performance']}  "
                      f"steps={res['steps']}  {res['elapsed_sec']}s")
                if args.object != "live":
                    print(f"  (target={res['target_object']} -> "
                          f"{'CORRECT' if res['primary_performance'].startswith('correct') else 'wrong'})")
    finally:
        cam.close()
        cv2.destroyAllWindows()
        print(f"\nDone. {n_infer} inference(s).")


def main():
    p = argparse.ArgumentParser(
        description="Live OAK-D Pro capture + Monty inference (one process)."
    )
    p.add_argument("--camera", default="oak", choices=sorted(CAMERAS),
                   help="Capture backend. 'femto' (ToF) measures glossy/white "
                        "curved surfaces that active stereo cannot: on the same "
                        "mug, centre-ROI depth validity was 92%% vs OAK's 58%%.")
    p.add_argument("--rs-preset", default="none",
                   help="D405 only. Depth post-processing preset "
                        "(none/light/medium/heavy). Default 'none': smoothing "
                        "measured worse on D405 — it replaces real-but-noisy "
                        "surface with fake-but-smooth (see CLAUDE.md dead ends).")
    p.add_argument("--align", default="c2d", choices=["c2d", "none", "sw", "hw"],
                   help="Femto only. 'c2d' (default) warps COLOUR into the depth "
                        "frame, fixing hsv sampling (29%% of feature weight) "
                        "while leaving depth untouched. 'sw' does the opposite "
                        "and measured worse (2/4 -> 0/4). 'none' leaves a ~100px "
                        "offset so hue reads off-object.")
    p.add_argument("--objects", default=None,
                   help="Comma-separated object ids to keep in memory (e.g. "
                        "montys_brain,tomato_soup_can). Inference cost is "
                        "~linear in object count, so limiting to demo objects "
                        "is the biggest speedup. Default: all 10.")
    p.add_argument("--object", default="live",
                   help="Object name; if a real class name (e.g. numenta_mug) "
                        "the result is also scored correct/wrong (default: live)")
    p.add_argument("--model-path", default=None,
                   help="Pretrained model dir (default: Robot Lab model)")
    p.add_argument("--hfov", type=float, default=None,
                   help=f"Capture HFOV in deg (default: {OAKD_HFOV} = OAK-D Pro)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-eval-steps", type=int, default=500)
    p.add_argument("--save-dir", default=None,
                   help="Also persist each capture as a scene dir here")
    p.add_argument("--no-isolate", action="store_true",
                   help="Feed RAW depth to Monty instead of the isolated blob. "
                        "Monty's process_depth_data already clips beyond "
                        "--depth-clip (background removal), but it canNOT remove "
                        "a hand at the same depth as the object — that is what "
                        "isolate adds. Use to A/B the two.")
    p.add_argument("--depth-clip", type=float, default=None,
                   help="Monty's off-object depth threshold in meters "
                        "(MONTY_DEPTH_CLIP, default 0.4). Training data sits at "
                        "0.17-0.26m; captures at ~0.4m get partially clipped.")
    p.add_argument("--preset", default="DENSITY")
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--min-distance", type=float, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--color-res", default="THE_4_K",
                   help="ColorCamera sensor mode. IMX378 supports ONLY 1920x1080 / THE_4_K (16:9) and THE_1352X1012 / THE_2024X1520 / THE_12_MP (4:3) — an unsupported mode drops the device off USB at pipeline.start(). THE_4_K is a 16:9 crop of the "
                        "4:3 IMX378, so the RGB preview loses vertical FOV that "
                        "depth (aligned to CAM_A's native field) keeps. Try "
                        "THE_1352X1012 for a 4:3 mode, then verify with "
                        "check_rgb_depth_align.py.")
    p.add_argument("--extended-disparity", action="store_true",
                   help="Halve the stereo minimum distance (~35cm -> ~17cm). "
                        "Required to shoot at the training distance of "
                        "0.17-0.26m — closer than the default minimum the "
                        "object returns NO depth and dist reads as background.")
    p.add_argument("--ir-brightness", type=int, default=1200)
    args = p.parse_args()

    if args.model_path is None:
        args.model_path = default_model_path()
    if args.hfov is None:
        # Femto centre-crops to the 54.201 deg training FOV; OAK is 63.75.
        args.hfov = CAMERAS[args.camera].default_hfov

    run(args)


if __name__ == "__main__":
    main()

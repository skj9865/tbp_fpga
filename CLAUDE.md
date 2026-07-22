# CLAUDE.md — tbp_fpga session context

Project-level guidance for Claude Code sessions on this repo. Auto-loaded
at session start. Update this file when conventions, environment, or the
current work state changes.

---

## About the user

- KETI engineer, email skj9865@ketisoc.com
- Researching FPGA hardware acceleration for Monty's evidence-based
  object recognition. Background in HW design; comfortable reading
  Python but works mostly at the system-integration level.
- Owns three depth cameras: Intel RealSense D405 (active stereo),
  Orbbec Femto Bolt (ToF), Luxonis OAK-D Pro (active stereo).
- Works across two machines:
  - **Mac (skj_macpro)** — earlier dev environment; Femto Bolt depth
    stream is broken on macOS Sequoia (pyorbbecsdk issue), so capture
    moved off this machine.
  - **HP Z8 (skj-z8)** — current main workstation. Ubuntu 24.04 +
    NVIDIA RTX 5000 Pro Blackwell + CUDA 13.0. All physical-object
    capture and inference run here.

## Collaboration style

- **Reply in Korean unless the user writes English.** Comments and
  identifiers in code stay English.
- **Be terse.** No trailing summaries, no "I'll now...". State result,
  stop.
- **Present options, let the user choose.** When there's a real
  tradeoff (env layout, commit structure, library choice), give 2-4
  numbered options with the tradeoff and recommend one — don't decide
  silently. Use AskUserQuestion for binary choices that block work.
- **The user drives execution.** When work moves between machines or
  involves data movement (USB copies, manual file transfers), don't
  push or copy without being told. "내가 수동으로 옮길게" is a normal
  pattern.
- Prefer concrete commands the user can paste, with checks before
  each step (`which nvcc`, `git status`, …) so failures surface early.

## Project overview

`tbp_fpga` is a standalone harness for Numenta's Monty
(`tbp.monty`, vendored at root — *not* a submodule, 1064 tracked
files). The goal is to profile Monty's evidence-based recognition and
identify hot operations (`hw_analysis/`) for FPGA acceleration.

FPGA-target docs: `scripts/SETUP_FPGA.md` (ARM/PetaLinux Monty
inference setup) and `scripts/PATCHES.md` (record of every
`tbp.monty` source edit needed to run on that target). Note
`rgbd_capture.py` (no camera suffix) is the **OAK-D Pro** capture
script.

### Code layout (beyond `scripts/`)

- **`wrapper/`** — algorithm-plugin layer that decouples inference
  from Monty. `base_algorithm.py` defines the interface every
  algorithm implements; `registry.py` registers them by `name()`
  via a class decorator; `monty_algorithm.py` is the Monty
  implementation; `wrapper.py` is the entrypoint. Add a new
  recognition backend here, not by editing `monty_inference.py`.
- **`hw_analysis/`** — the FPGA-acceleration work itself.
  `hw_accel_ops.md` specs the candidate HW ops (Monty Steps 6-10).
  `extract_hw_data.py` dumps Monty's intermediate tensors
  (`hw_test_data.npz`); `hw_ops.py` (numpy-only, no Monty dep) and
  `hw_ops_gpu.py` (PyTorch) reimplement those ops standalone;
  `verify_hw_ops.py` checks them against Monty's real output;
  `profile_inference.py` cProfiles inference with threading off to
  expose true compute cost.
- **`outputs/`** — Hydra run dirs (`outputs/<date>/<time>/`),
  tracked in git. `monty_profile.prof` / `*.log` at root are
  gitignored per-machine artifacts.

End-to-end physical-object recognition pipeline:

```
camera capture  →  preprocess (isolate / smooth)  →  monty_inference
  (D405/Femto)    (mask hand+bg / fill holes)         (Robot Lab model)
```

### Pretrained model

Default is **`surf_agent_1lm_tbp_robot_lab`** (10 objects including
`numenta_mug`). Lives at
`~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/surf_agent_1lm_tbp_robot_lab/pretrained/`.
The older `surf_agent_1lm_numenta_lab_obj` is *not* used — don't suggest it.

### Monty patches in this fork

- `aa9bba5` — Replaces `EvidenceGoalStateGenerator` with
  `GraphGoalStateGenerator` when `use_goal_state_driven_actions=False`,
  skipping the expensive `_compute_graph_mismatch`.
- `tbp.monty/src/tbp/monty/frameworks/environments/two_d_data.py:563`
  (note: `environments/`, not `environment_utils/`) — HFOV hard-coded
  to **54.201°** (iPad TrueDepth = Robot Lab training value). All
  capture scripts center-crop to this. For OAK-D Pro path, patch to
  63.75 instead. Full edit history in `scripts/PATCHES.md`.

### Camera HFOVs (native, before crop)

| Camera | Native HFOV | Notes |
|---|---|---|
| D405 | 89.28° | Passive stereo (NO IR projector — unlike D435i). RGB derived from left depth imager, so RGB/depth natively aligned (no Δ offset, hence no align problem). Weak on untextured/curved-surface centers (no projector to add texture) → red invalid holes. Noisy on shiny/curved surfaces. Best raw mug accuracy was 25% (1/4). |
| Femto Bolt | 90.78° (WFOV 1024×1024) | ToF. Linux only — macOS depth stream broken. |
| OAK-D Pro | 63.75° (at 640×480 preview) | Patches `two_d_data.py` HFOV to match. |

## Environments

Two conda envs on Z8, both Python 3.11:

```
tbp_fpga   ← inference, monty, hw_analysis
orbbec     ← camera SDKs (pyrealsense2, pyorbbecsdk v2.1.1)
```

Why split: pyorbbecsdk pins specific cp311 ABI; keeping it out of
`tbp_fpga` avoids torch/CUDA conflicts.

### Critical install gotchas

- **`pip install -e tbp.monty/` MUST use `--no-deps`.** Its
  `setup.py` hard-pins `torch==1.13.1` and will downgrade your
  cu128 torch silently otherwise (then nothing runs on Blackwell).
- **The "torch==1.13.1 incompatible" warning from pip is just a
  warning.** torch 2.x runs Monty fine; setup.py pin is stale.
- **`pyorbbecsdk` PyPI 1.3.2 cp311 Linux wheel is corrupted** — it
  contains `cpython-311-darwin.so` files. Install v2.1.1 from GitHub
  Releases instead. Import name stays `pyorbbecsdk`.
- **Femto Bolt on Linux**: udev rules (`vendor 2bc5`, mode 0666,
  group plugdev) + `usermod -aG video,plugdev $USER` +
  **desktop session logout/login** (Terminal restart insufficient).
- **torch-scatter / torch-sparse build from source** fails under PEP 517
  build isolation (no torch in the build env). Either use PyG
  wheelhouse (`-f https://data.pyg.org/whl/torch-X.Y.Z+cuABC.html`),
  or `--no-build-isolation` with nvcc in PATH, or skip via the
  minimal variant + `patch_lazy_imports.py`.

## Repository conventions

- `tbp.monty/` is tracked directly, no submodule. Push the repo and
  monty changes ride along automatically.
- `.gitignore` excludes per-machine experiment outputs (`eval*.csv`,
  `Log/`, `*.log`, `*.prof`). Existing tracked CSVs
  (`eval_stats.csv`, `eval_stats_old.csv`, `eval_test.csv`) are
  reference baselines — don't add new eval CSVs to git.
- `.claude/settings.local.json` *is* tracked (force-added) so the
  per-machine allowed-command list stays in sync. Updates from a
  new machine should still be committed.
- Commits are split by *feature/topic*, not by file. Past commits
  followed: gitignore, capture/calibration scripts, preprocess +
  wrapper, inference defaults — keep that granularity.

## Common commands

```bash
# Inference (Z8)
conda activate tbp_fpga
python scripts/monty_inference.py --max-episodes 1                       # smoke
python scripts/monty_inference.py --output-csv eval_stats.csv            # full 48
python scripts/monty_inference.py \                                       # single object
    --scenes 0,0,0,0 --versions 0,1,2,3 \
    --data-path ~/tbp/data/worldimages/captured_scenes_isolated \
    --output-csv eval_mug.csv

# Capture (Z8, orbbec env)
conda activate orbbec
python scripts/rgbd_capture_d405.py  --object numenta_mug --index 0 --num-captures 4
python scripts/rgbd_capture_femto.py --object numenta_mug --index 0 --stdin
python scripts/rgbd_capture.py       --object numenta_mug --index 0   # OAK-D Pro

# Preprocess
python scripts/isolate_object.py \
    --input  ~/tbp/data/worldimages/captured_scenes \
    --output ~/tbp/data/worldimages/captured_scenes_isolated \
    --recursive --debug-dir /tmp/iso_debug

# Calibration sanity
python scripts/check_d405_calibration.py
python scripts/check_femto_calibration.py
python scripts/check_oak_calibration.py

# Inspect / verify captures
python scripts/rgbd_verify.py     <scene_dir>   # format check for SaccadeOnImageEnvironment
python scripts/visualize_depth.py <scene_dir>   # single or side-by-side depth view
python scripts/compare_depth.py   <a> <b>       # captured vs reference depth
```

## Multi-machine workflow

When something changes on Mac and needs to land on Z8 (or vice versa):

1. **Code changes** → commit + push to `origin/main`, then `git pull`
   on the other side.
2. **Data / pretrained models** → USB copy (`~/tbp/data/`,
   `~/tbp/results/`). Don't try to git-track these.
3. **Conda env state** → not synced. Recreate from
   `scripts/requirements-inference.txt` per README §3 (PC) or
   `requirements-inference-minimal.txt` + `patch_lazy_imports.py`
   for ARM/FPGA. (README §2 is the `tbp.monty` editable install.)
4. **`tbp.monty` editable install** → `pip install --no-deps -e tbp.monty/`
   on every fresh env. USB copying files alone doesn't register the
   package in site-packages.

## Current work state

Last update: 2026-06-26

- **Done**: Mac → Z8 migration of the repo. README rewritten with
  multi-machine + camera-pipeline sections (commit `614854c`). Femto
  Bolt detected on Z8 via pyorbbecsdk v2.1.1 + udev rules.
- **Done**: Z8 `tbp_fpga` env fully working — 1-episode inference
  passes (`numenta_mug` correct, CUDA).
  - PyTorch cu128 (2.11.0) installed and verified for Blackwell.
    Toolkit on disk is CUDA 12.8 (nvcc 12.8), matching cu128.
  - `tbp.monty` editable install done with `--no-deps`.
  - **`torch-scatter` / `torch-sparse` resolved via PyG wheelhouse**
    — prebuilt `pt211cu128` wheels exist after all (no build needed):
    `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.11.0+cu128.html`
    → scatter 2.1.2, sparse 0.6.18. The "404 / source build" decision
    tree in earlier notes turned out unnecessary.
  - **PyG 2.8.0 `Data.keys` is now a method**, not a property — broke
    `object_model_utils.py:torch_graph_to_numpy`. Applied PATCHES.md
    patch 4 (callable-guarded `keys()`) to the Z8 monty source.
- **In progress**: Femto Bolt end-to-end capture → isolate →
  inference for `numenta_mug`. Capturing to a **separate**
  `~/tbp/data/worldimages/captured_scenes_femto` (don't reuse the
  D405-populated `captured_scenes/0_numenta_mug` — versions would
  mix cameras/resolutions). Use `--align none` (see dead ends).
  Hypothesis: ToF cleaner depth → recognition accuracy >> D405's
  25% ceiling.

## Known dead ends (don't re-explore)

- Smoothing D405 depth (bilateral / hole-fill) **hurts** accuracy —
  replaces real-but-noisy surface with fake-but-smooth. `smooth_depth.py`
  exists but use sparingly; prefer raw + `isolate_object.py`.
- macOS Camera permission via `cv2` doesn't propagate to
  `pyorbbecsdk` — different access path, sudo workaround doesn't fix
  Femto depth either. Moved capture to Linux instead.
- `eval.csv` from `world_image_on_scanned_model` on `captured_scenes`
  without isolation = 0% (hand pixels dominate the saccade patch).
  Always run `isolate_object.py` before inferring on D405/Femto data.
- **Femto `--align sw` (depth→color) hurts recognition.** Measured:
  raw `--align none` got ~2/4 mug captures correct, SW-aligned got
  0/4. Same class of mistake as depth smoothing — SW align warps ToF
  depth into the color frame (occlusion holes, edge interpolation)
  and changes depth resolution to 1280×720, so the 517/1024 crop no
  longer lands on the trained 54.201° HFOV.
  Why none wins even though **RGB *is* a real feature**: the SM
  extracts `hsv` from the saccade-center pixel and matching uses it
  with `feature_weights hsv=[1.0,0.5,0.5]` (hue weight 1.0, =
  surface_normal/curvature) and tight `tolerances hsv=[0.1,0.2,0.2]`
  (`SaccadeOnImageEnvironment` feeds the real `rgb_patch`, not the
  depth-replicated rgba — that path is Omniglot-only). But hue is a
  *single center pixel*, while SW align degrades the whole-patch
  shape features (surface_normal, principal_curvatures, weight 1.0)
  AND the 3D location match (`location tolerance=0.015`, very
  sensitive) via the FOV/scale shift. Net: shape+location loss >
  hue-alignment gain. **Capture with `--align none`**, place the
  object at the **depth-frame center**, and make it fill enough of
  the frame that the small RGB↔depth offset still leaves the center
  pixel on the object (else hue samples background).

# tbp_fpga

FPGA hardware acceleration for Monty's evidence-based object recognition pipeline.

This repository contains:
- A standalone Monty inference script (`scripts/monty_inference.py`) for benchmarking and FPGA porting
- A depth-camera capture + preprocessing pipeline (D405, Femto Bolt, OAK-D Pro) for live object recognition
- Hardware acceleration analysis and reference implementations (`hw_analysis/`)
- An algorithm wrapper interface (`wrapper/`) for swapping inference backends
- A vendored copy of `tbp.monty` pinned to a known-good version (tracked directly, not a submodule)

## Requirements

- Python 3.9+ for inference (tested on 3.9 / 3.11 / 3.12)
- Python 3.11 for depth-camera SDKs (pyorbbecsdk v2.1.1 ships cp311 wheels only)
- conda / miniforge
- ~2 GB disk space for inference dependencies
- Pretrained Monty models (YCB v12)

## Setup

### 1. Clone

```bash
git clone https://github.com/skj9865/tbp_fpga.git
cd tbp_fpga
```

`tbp.monty/` is checked in as a regular directory (not a submodule), so no `submodule init` is needed.

### 2. Create the inference conda env (`tbp_fpga`)

```bash
conda create -n tbp_fpga python=3.11 -y
conda activate tbp_fpga
pip install --no-deps -e tbp.monty/
pip install -r scripts/requirements-inference.txt
```

`--no-deps` skips heavy dependencies like `habitat-sim` that are not needed for inference.

**ARM / FPGA (Cortex-A72, Zynq) variant:**

```bash
pip install -r scripts/requirements-inference-minimal.txt
python scripts/patch_lazy_imports.py
```

The minimal variant skips `torch-scatter`, `torch-sparse`, `pandas`, and `wandb` (not buildable or not needed on ARM), and the patch script converts their top-level imports inside `tbp.monty` to lazy imports.

### 3. (Optional) Create the depth-camera env (`orbbec`)

Required only if you'll capture from D405 or Femto Bolt. Keeping camera SDKs out of the inference env avoids version conflicts.

```bash
conda create -n orbbec python=3.11 -y
conda activate orbbec
pip install pyrealsense2 opencv-python pillow numpy
# Femto Bolt: install pyorbbecsdk v2.1.1 from GitHub Releases
# (PyPI's pyorbbecsdk 1.3.2 cp311 Linux wheel is broken — contains macOS .so files)
pip install pyorbbecsdk2-2.1.1-cp311-cp311-linux_x86_64.whl
```

### 4. Linux USB permissions for Femto Bolt

```bash
sudo tee /etc/udev/rules.d/99-orbbec.rules << 'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb_device", ATTR{idVendor}=="2bc5", MODE="0666", GROUP="plugdev"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo usermod -aG video,plugdev $USER
# Desktop session logout/login required (Terminal restart is NOT enough)
```

### 5. Download pretrained models

Monty requires the YCB v12 pretrained set. Place it under `~/tbp/results/monty/pretrained_models/`:

```
~/tbp/results/monty/pretrained_models/
└── pretrained_ycb_v12/
    └── surf_agent_1lm_tbp_robot_lab/
        └── pretrained/
```

`surf_agent_1lm_tbp_robot_lab` is the default model (it contains `numenta_mug` and 9 other Robot-Lab objects). Set `MONTY_MODELS` to override the location:

```bash
export MONTY_MODELS=/custom/path/to/pretrained_models
```

### 6. Download benchmark data

The default benchmark uses iPad-captured RGBD scenes. Place them under `~/tbp/data/worldimages/standard_scenes/`, or set `MONTY_DATA`:

```bash
export MONTY_DATA=/custom/path/to/data
```

## Running inference

### Quick test (1 episode)

```bash
python scripts/monty_inference.py --max-episodes 1
```

### Full benchmark (48 episodes on standard_scenes)

```bash
python scripts/monty_inference.py --output-csv eval_stats.csv
```

### Single object / version (e.g. just numenta_mug v0..v3)

`--scenes` and `--versions` take comma-separated integers and must have the same length. Scene 0 is `numenta_mug` in the Robot Lab model.

```bash
python scripts/monty_inference.py \
    --scenes 0,0,0,0 --versions 0,1,2,3 \
    --data-path ~/tbp/data/worldimages/captured_scenes \
    --output-csv eval_mug.csv
```

### Custom paths

```bash
python scripts/monty_inference.py \
    --model-path /custom/model/path \
    --data-path /custom/data/path \
    --max-episodes 10 \
    --output-csv my_results.csv
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model-path` | `$MONTY_MODELS/pretrained_ycb_v12/surf_agent_1lm_tbp_robot_lab/pretrained` | Pretrained model directory |
| `--data-path` | `$MONTY_DATA/worldimages/standard_scenes` | Benchmark data directory |
| `--scenes` | `None` (= built-in 48) | Comma-separated scene indices (must pair with `--versions`) |
| `--versions` | `None` (= built-in 48) | Comma-separated version indices, same length as `--scenes` |
| `--max-episodes` | All | Limit number of episodes |
| `--max-eval-steps` | 500 | Max matching steps per episode |
| `--output-csv` | `None` (no file) | Per-episode results CSV |
| `--log-level` | `WARNING` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `--seed` | 42 | Random seed |

## Camera capture pipeline

End-to-end flow for going from a physical object to a recognition result:

```
   ┌──────────────┐   ┌──────────┐   ┌────────────────────┐   ┌───────────┐
   │ rgbd_capture │ → │ smooth_  │ → │ isolate_object.py  │ → │ monty_    │
   │   _<cam>.py  │   │ depth.py │   │ (mask out hand/bg) │   │ inference │
   └──────────────┘   └──────────┘   └────────────────────┘   └───────────┘
       raw RGBD          (opt)           object-only depth        result
```

All capture scripts emit the same on-disk format (640×480 RGBA PNG + float32 depth in meters) that `SaccadeOnImageEnvironment` consumes, so swapping cameras requires no code change downstream. The capture HFOV is center-cropped to 54.201° to match Monty's training value (hard-coded in `tbp.monty/.../two_d_data.py`).

### Step 1: Calibration sanity check

Print intrinsics, HFOV, and the recommended center crop for each camera:

```bash
# In the orbbec env:
python scripts/check_d405_calibration.py
python scripts/check_femto_calibration.py
# In the tbp_fpga env (uses depthai):
python scripts/check_oak_calibration.py
```

### Step 2: Capture

**Intel RealSense D405** (active stereo, HFOV 89.28°, USB 3.0):
```bash
conda activate orbbec
python scripts/rgbd_capture_d405.py --object numenta_mug --index 0 --num-captures 4
# Headless / SSH:
python scripts/rgbd_capture_d405.py --object numenta_mug --index 0 --headless --warmup 3
```

**Orbbec Femto Bolt** (ToF, WFOV 1024×1024, USB 3.0; Linux only):
```bash
conda activate orbbec
python scripts/rgbd_capture_femto.py --object numenta_mug --index 0 --stdin
# --stdin = Enter to capture, q+Enter to quit; preview PNG at /tmp/femto_preview.png
```

**OAK-D Pro** (active stereo, HFOV 63.75°):
```bash
conda activate tbp_fpga
python scripts/rgbd_capture.py --object numenta_mug --index 0
# Note: also requires patching two_d_data.py HFOV to 63.75
```

Outputs land in `~/tbp/data/worldimages/captured_scenes/<index>_<object>/` as `rgb_<v>.png` + `depth_<v>.data`.

### Step 3: Preprocess (optional)

**Isolate the centered object** — masks out hand, background, and noise outside the object using a depth-band + connected-component heuristic. Strongly recommended for raw D405 captures where the saccade patch otherwise lands on the hand:

```bash
python scripts/isolate_object.py \
    --input  ~/tbp/data/worldimages/captured_scenes \
    --output ~/tbp/data/worldimages/captured_scenes_isolated \
    --recursive \
    --debug-dir /tmp/iso_debug   # optional: side-by-side overlays
```

**Smooth the depth** — light hole-fill + edge-preserving bilateral filter. Useful for noisy ToF but can replace real-but-noisy surface with fake-but-smooth, so prefer raw + isolate for testing:

```bash
python scripts/smooth_depth.py \
    --input  ~/tbp/data/worldimages/captured_scenes \
    --output ~/tbp/data/worldimages/captured_scenes_smoothed \
    --preset medium --recursive
```

**Visualize / compare** — side-by-side RGB / depth colormap for diagnosing NaN coverage and noise:

```bash
python scripts/visualize_depth.py \
    --scene   ~/tbp/data/worldimages/captured_scenes/0_numenta_mug \
    --compare ~/tbp/data/worldimages/standard_scenes/0_numenta_mug \
    --label-a "D405" --label-b "iPad" \
    --save-dir /tmp/depth_compare
```

### Step 4: Inference on captured data

```bash
python scripts/monty_inference.py \
    --data-path ~/tbp/data/worldimages/captured_scenes_isolated \
    --scenes 0,0,0,0 --versions 0,1,2,3 \
    --output-csv eval_mug.csv
```

## Multi-machine setup (Linux workstation, e.g. HP Z8)

The cleanest way to mirror a working setup on a second machine:

1. `git clone https://github.com/skj9865/tbp_fpga.git` — gets all source + the vendored `tbp.monty/` (already 1064 tracked files, no submodule init).
2. Run the conda env setup from §2–4 above.
3. `cd tbp.monty && pip install -e .` registers the package in the env's site-packages (USB-copying files is not enough — the env still needs the egg link).
4. Copy the **non-git** assets by USB:
   - `~/tbp/data/worldimages/standard_scenes/` — iPad reference scenes
   - `~/tbp/data/worldimages/captured_scenes*/` — your D405/Femto captures
   - `~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/surf_agent_1lm_tbp_robot_lab/` — pretrained model `.pt`
5. Sanity check: `python scripts/monty_inference.py --max-episodes 1` should reproduce the same first-episode result as the source machine.

## Expected results

**Full standard_scenes benchmark (48 episodes, surf_agent_1lm_tbp_robot_lab):**
- Accuracy (correct + MLH): expect ~60–70% depending on model variant
- Runtime on x86 PC: ~5–7 minutes total
- Runtime on Cortex-A72 (FPGA PS): ~70 minutes total (~88s per episode)

**D405-captured numenta_mug (4 episodes, raw):** ~25% (1/4 correct) — the active-stereo noise on shiny / curved surfaces is the recognition ceiling. Femto Bolt ToF or iPad TrueDepth give substantially cleaner depth.

## Profiling

To profile inference and identify hardware acceleration targets:

```bash
python hw_analysis/profile_inference.py
```

This runs 1 episode with multithreading disabled and saves `monty_profile.prof` plus a top-N function breakdown.

## Repository layout

```
tbp_fpga/
├── scripts/
│   ├── monty_inference.py              # Standalone Monty inference entrypoint
│   ├── patch_lazy_imports.py           # ARM/FPGA import workaround
│   │
│   ├── check_d405_calibration.py       # D405 intrinsics + HFOV + crop calc
│   ├── check_femto_calibration.py      # Femto Bolt intrinsics
│   ├── check_oak_calibration.py        # OAK-D Pro intrinsics
│   │
│   ├── rgbd_capture.py                 # OAK-D Pro capture (depthai)
│   ├── rgbd_capture_d405.py            # D405 capture (pyrealsense2)
│   ├── rgbd_capture_femto.py           # Femto Bolt capture (pyorbbecsdk v2)
│   ├── rgbd_verify.py                  # Visualize captured RGBD data
│   ├── compare_depth.py                # Compare depth outputs
│   │
│   ├── isolate_object.py               # Mask depth to centered object only
│   ├── smooth_depth.py                 # Edge-preserving depth smoothing
│   ├── visualize_depth.py              # RGB + depth side-by-side viewer
│   │
│   ├── requirements-inference.txt          # PC dependencies
│   ├── requirements-inference-minimal.txt  # ARM/FPGA dependencies
│   ├── requirements-capture.txt            # Capture dependencies
│   ├── SETUP_FPGA.md                   # FPGA-specific setup notes
│   └── PATCHES.md                      # Documented patches to tbp.monty
│
├── wrapper/                            # Algorithm wrapper for backend swap
│   ├── base_algorithm.py
│   ├── monty_algorithm.py
│   ├── registry.py
│   └── wrapper.py
│
├── hw_analysis/                        # HW acceleration analysis
│   ├── hw_accel_ops.md                 # HW op specifications (Steps 6-10)
│   ├── hw_ops.py                       # Reference Python implementation
│   ├── hw_ops_gpu.py                   # GPU reference implementation
│   ├── verify_hw_ops.py                # Correctness verification
│   └── profile_inference.py            # cProfile wrapper
│
├── tbp.monty/                          # Vendored Monty framework (pinned)
├── eval_stats.csv                      # Reference benchmark results
└── README.md
```

## Troubleshooting

**Model not found**
Verify `MONTY_MODELS` points to a directory containing `pretrained_ycb_v12/surf_agent_1lm_tbp_robot_lab/pretrained/`.

**Data not found**
Verify `MONTY_DATA` points to a directory containing `worldimages/standard_scenes/`.

**Femto Bolt: `devices: 0` or `usbEnumerator openUsbDevice failed`**
udev rules + user groups, see §4. Desktop session logout/login is required — Terminal restart is not enough to pick up new group membership.

**Femto Bolt: PyPI `pyorbbecsdk` import fails with `.so` not found**
The PyPI 1.3.2 cp311 Linux wheel is corrupted (contains darwin `.so` files). Install v2.1.1 from GitHub Releases instead. Import name is still `pyorbbecsdk`.

**D405: depth is all zeros on macOS**
macOS Camera permission must be granted to the *Terminal application*, not to a Python interpreter. Try recapturing under a fresh Terminal launched from Finder.

**Hand / background pixels dominate the saccade patch**
Run `scripts/isolate_object.py` over the captured scenes before inference. Use `--debug-dir` to verify the mask catches the object and rejects the hand.

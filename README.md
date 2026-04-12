# tbp_fpga

FPGA hardware acceleration for Monty's evidence-based object recognition pipeline.

This repository contains:
- A standalone Monty inference script (`scripts/monty_inference.py`) for benchmarking and FPGA porting
- Hardware acceleration analysis and reference implementations (`hw_analysis/`)
- A vendored copy of `tbp.monty` pinned to a known-good version

## Requirements

- Python 3.9+ (tested on 3.9 / 3.12)
- conda / miniforge
- ~2 GB disk space for dependencies
- Pretrained Monty models (YCB v12)

## Setup

### 1. Create conda environment

```bash
conda create -n tbp_fpga python=3.9 -y
conda activate tbp_fpga
```

### 2. Install tbp.monty (editable, no deps)

```bash
cd /path/to/tbp_fpga
pip install --no-deps -e tbp.monty/
```

The `--no-deps` flag is important — it skips heavy dependencies like `habitat-sim` that are not needed for inference.

### 3. Install inference dependencies

Choose one of the following based on your platform:

**x86 Mac / Linux (PC):**
```bash
pip install -r scripts/requirements-inference.txt
```

**ARM / FPGA (Cortex-A72, Zynq):**
```bash
pip install -r scripts/requirements-inference-minimal.txt
python scripts/patch_lazy_imports.py
```

The minimal variant skips `torch-scatter`, `torch-sparse`, `pandas`, and `wandb` (not buildable or not needed on ARM), and runs a patch script to convert their top-level imports inside `tbp.monty` to lazy imports.

### 4. Download pretrained models

Monty requires pretrained YCB v12 models. Place them under `~/tbp/results/monty/pretrained_models/` so the structure looks like:

```
~/tbp/results/monty/pretrained_models/
└── pretrained_ycb_v12/
    └── surf_agent_1lm_numenta_lab_obj/
        └── pretrained/
```

Alternatively, set the `MONTY_MODELS` environment variable to point to your models directory:

```bash
export MONTY_MODELS=/custom/path/to/pretrained_models
```

### 5. Download benchmark data

The default benchmark (`world_image_on_scanned_model`) requires RGBD scenes. Place them under `~/tbp/data/worldimages/standard_scenes/`, or set `MONTY_DATA`:

```bash
export MONTY_DATA=/custom/path/to/data
```

## Running inference

### Quick test (1 episode)

```bash
python scripts/monty_inference.py --max-episodes 1
```

### Full benchmark (48 episodes)

```bash
python scripts/monty_inference.py --output-csv eval_stats.csv
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
| `--model-path` | `$MONTY_MODELS/pretrained_ycb_v12/...` | Pretrained model directory |
| `--data-path` | `$MONTY_DATA/worldimages/standard_scenes` | Benchmark data directory |
| `--max-episodes` | All (48) | Limit number of episodes |
| `--max-eval-steps` | 500 | Max matching steps per episode |
| `--output-csv` | `eval_stats.csv` | Per-episode results CSV |
| `--log-level` | `INFO` | Logging verbosity |
| `--seed` | 42 | Random seed |

## Expected results

Running the full 48-episode benchmark should yield approximately:

- **Accuracy (correct + MLH):** ~66.7% (32/48)
- **Runtime on x86 PC:** ~5-7 minutes total
- **Runtime on Cortex-A72 (FPGA PS):** ~70 minutes total (~88s per episode)

## Profiling

To profile inference and identify hardware acceleration targets:

```bash
python hw_analysis/profile_inference.py
```

This runs 1 episode with multithreading disabled and saves `monty_profile.prof` along with top-N function breakdowns.

## Repository layout

```
tbp_fpga/
├── scripts/              # Inference and capture utilities
│   ├── monty_inference.py         # Standalone Monty inference entrypoint
│   ├── patch_lazy_imports.py      # ARM/FPGA import workaround
│   ├── rgbd_capture.py            # RGBD capture from cameras
│   ├── rgbd_verify.py             # Visualize captured RGBD data
│   ├── compare_depth.py           # Compare depth outputs
│   ├── requirements-inference.txt          # PC dependencies
│   ├── requirements-inference-minimal.txt  # ARM/FPGA dependencies
│   ├── requirements-capture.txt            # Capture dependencies
│   ├── SETUP_FPGA.md              # FPGA-specific setup notes
│   └── PATCHES.md                 # Documented patches to tbp.monty
├── hw_analysis/          # HW acceleration analysis
│   ├── hw_accel_ops.md            # HW op specifications (Steps 6-10)
│   ├── hw_ops.py                  # Reference Python implementation
│   ├── hw_ops_gpu.py              # GPU reference implementation
│   ├── verify_hw_ops.py           # Correctness verification
│   └── profile_inference.py       # cProfile wrapper
├── tbp.monty/            # Vendored Monty framework (pinned version)
├── eval_stats.csv        # Benchmark results
└── README.md
```

## Troubleshooting

**Model not found**
Verify `MONTY_MODELS` points to a directory containing `pretrained_ycb_v12/surf_agent_1lm_numenta_lab_obj/pretrained/`.

**Data not found**
Verify `MONTY_DATA` points to a directory containing `worldimages/standard_scenes/`.

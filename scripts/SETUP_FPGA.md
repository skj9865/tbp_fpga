# FPGA (ARM/PetaLinux) Monty Inference Setup

## Prerequisites
- Python 3.9+
- pip3 (upgrade first: `pip3 install --upgrade pip setuptools`)
- Sufficient disk space (~2GB for packages). PetaLinux rootfs may need expansion:
  ```bash
  # If rootfs is full, expand SD card partition:
  sudo fdisk /dev/mmcblk0   # delete p2, recreate with same start sector, full size
  sudo resize2fs /dev/mmcblk0p2
  ```

## Step-by-step Setup

```bash
# 1) Transfer files to FPGA board
scp -r scripts/ tbp.monty/ <user>@<fpga-ip>:~/tbp_fpga/

# 2) On FPGA board:
cd ~/tbp_fpga

# 3) Upgrade pip (required for pyproject.toml editable install)
pip3 install --upgrade pip setuptools

# 4) Install tbp.monty source (no dependencies)
pip3 install --no-deps -e tbp.monty/

# 5) Install minimal dependencies
pip3 install -r scripts/requirements-inference-minimal.txt

# For ARM torch, if default wheel fails:
#   pip3 install torch --index-url https://download.pytorch.org/whl/cpu

# For torch-geometric, if full install fails:
#   pip3 install --no-deps torch-geometric

# 6) Patch lazy imports (wandb, pandas — not needed for inference)
python3 scripts/patch_lazy_imports.py

# 7) Run inference
python3 scripts/monty_inference.py \
  --model-path ~/tbp/results/monty/pretrained_models/pretrained_ycb_v12/surf_agent_1lm_numenta_lab_obj/pretrained \
  --data-path ~/tbp/data/worldimages/standard_scenes \
  --output-csv /tmp/results.csv
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'tbp'` | `pip3 install --no-deps -e tbp.monty/` |
| `No module named 'wandb'` | `python3 scripts/patch_lazy_imports.py` |
| `No module named 'torch_geometric'` | `pip3 install --no-deps torch-geometric` (required for model.pt unpickle) |
| `WeightsUnpickler error` / `weights_only` | Already fixed in script (`weights_only=False`) |
| `No space left on device` | Expand SD card partition (see Prerequisites) |
| `setup.py not found` (editable install) | `pip3 install --upgrade pip setuptools` then retry |

## Expected Result
- 48 episodes, ~66-68% accuracy (32-33 correct out of 48)

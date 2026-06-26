#!/usr/bin/env python3
"""Edge-preserving smoothing for raw RGBD captures.

Reads raw depth_{v}.data + rgb_{v}.png produced by rgbd_capture_d405.py and
writes smoothed depth (+ unchanged RGB copy) to a separate output directory.

Pipeline (per depth frame):
  1. Identify NaN regions (invalid pixels).
  2. Inpaint only "small" NaN holes (within `fill_radius` px of a valid pixel)
     so background / deep-occlusion NaN stays NaN.
  3. Bilateral filter on the filled depth — edge-preserving, so the mug
     handle / object boundary is not blurred away.
  4. Restore NaN where it could not be confidently filled.

Workflow:
  # 1. Capture (raw, no smoothing)
  python scripts/rgbd_capture_d405.py --object numenta_mug --index 0

  # 2. Smooth (this script)
  python scripts/smooth_depth.py \\
      --input ~/tbp/data/worldimages/captured_scenes \\
      --output ~/tbp/data/worldimages/captured_scenes_smooth_medium \\
      --preset medium --recursive

  # 3. Inference on smoothed data
  python scripts/monty_inference.py \\
      --data-path ~/tbp/data/worldimages/captured_scenes_smooth_medium \\
      --scenes 0,0,0,0 --versions 0,1,2,3 \\
      --output-csv eval_smooth_medium.csv
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


# Output shape (matches rgbd_capture_d405.py)
TARGET_W, TARGET_H = 640, 480


# -----------------------------------------------------------------------------
# Smoothing presets
# -----------------------------------------------------------------------------
# fill_radius        — max distance (px) from a valid pixel that we still fill
# inpaint_radius     — cv2.inpaint propagation radius
# bilateral_d        — neighborhood diameter for bilateral filter
# bilateral_sigma_color — depth tolerance (meters) — differences larger than
#                         this are treated as edges (NOT smoothed across)
# bilateral_sigma_space — spatial sigma in pixels
# median_ksize       — pre-smoothing median filter (0 = disabled), removes speckle
#
# Tuning intent:
#   light : minimal smoothing — preserves fine curvature for high-detail objects
#   medium: balanced — recommended default
#   heavy : maximum noise reduction + aggressive hole fill (lossier curvature)

SMOOTH_PRESETS = {
    "light": dict(
        fill_holes=True,
        fill_radius=2,
        inpaint_radius=2,
        bilateral_d=5,
        bilateral_sigma_color=0.010,   # 1 cm
        bilateral_sigma_space=5,
        median_ksize=0,
    ),
    "medium": dict(
        fill_holes=True,
        fill_radius=4,
        inpaint_radius=3,
        bilateral_d=7,
        bilateral_sigma_color=0.015,   # 1.5 cm
        bilateral_sigma_space=7,
        median_ksize=3,
    ),
    "heavy": dict(
        fill_holes=True,
        fill_radius=8,
        inpaint_radius=5,
        bilateral_d=9,
        bilateral_sigma_color=0.025,   # 2.5 cm
        bilateral_sigma_space=9,
        median_ksize=5,
    ),
}


# -----------------------------------------------------------------------------
# Core smoothing
# -----------------------------------------------------------------------------

def smooth_depth(depth_m, preset):
    """Smooth a single depth array (float32, NaN for invalid).

    Returns a new float32 array. NaN is preserved for pixels that were
    too far from any valid measurement to be safely filled.
    """
    nan_mask = np.isnan(depth_m)
    depth_filled = np.where(nan_mask, 0.0, depth_m).astype(np.float32)

    # Step 1: optional median pre-filter to remove speckle (preserves edges).
    if preset["median_ksize"] >= 3:
        k = int(preset["median_ksize"]) | 1  # force odd
        valid_only = depth_filled.copy()
        valid_only[nan_mask] = 0
        # median can corrupt NaN region but we restore mask below
        depth_filled = cv2.medianBlur(valid_only, k)

    # Step 2: fill only "near" NaN holes (small gaps inside the object surface).
    if preset["fill_holes"] and nan_mask.any():
        valid_u8 = (~nan_mask).astype(np.uint8)
        dist_from_valid = cv2.distanceTransform(
            1 - valid_u8, cv2.DIST_L2, 3,
        )
        fillable = nan_mask & (dist_from_valid <= preset["fill_radius"])
        fill_u8 = fillable.astype(np.uint8)
        if fill_u8.any():
            depth_filled = cv2.inpaint(
                depth_filled, fill_u8,
                inpaintRadius=preset["inpaint_radius"],
                flags=cv2.INPAINT_TELEA,
            )
            # These pixels now have a value; remove from NaN mask
            nan_mask = nan_mask & ~fillable

    # Step 3: edge-preserving bilateral smoothing.
    depth_smooth = cv2.bilateralFilter(
        depth_filled,
        d=preset["bilateral_d"],
        sigmaColor=float(preset["bilateral_sigma_color"]),
        sigmaSpace=float(preset["bilateral_sigma_space"]),
    )

    # Step 4: restore unrecoverable NaN regions.
    depth_smooth[nan_mask] = np.nan
    return depth_smooth.astype(np.float32)


# -----------------------------------------------------------------------------
# I/O — scene-level processing
# -----------------------------------------------------------------------------

def process_scene(input_dir, output_dir, preset_name, copy_rgb=True):
    """Process one scene directory: depth_*.data + rgb_*.png."""
    preset = SMOOTH_PRESETS[preset_name]
    depth_files = sorted(input_dir.glob("depth_*.data"))
    if not depth_files:
        print(f"  (no depth files in {input_dir})")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for depth_path in depth_files:
        v = depth_path.stem.split("_")[1]
        depth_raw = np.fromfile(depth_path, np.float32).reshape(
            TARGET_H, TARGET_W,
        )

        nan_in = np.isnan(depth_raw).sum()
        depth_out = smooth_depth(depth_raw, preset)
        nan_out = np.isnan(depth_out).sum()
        total = depth_raw.size

        out_path = output_dir / f"depth_{v}.data"
        depth_out.tofile(out_path)
        print(
            f"  v={v}: NaN {nan_in*100/total:5.1f}% -> {nan_out*100/total:5.1f}%"
            f"   saved {out_path.name}"
        )

        if copy_rgb:
            rgb_src = input_dir / f"rgb_{v}.png"
            if rgb_src.exists():
                shutil.copy(rgb_src, output_dir / rgb_src.name)

    return len(depth_files)


def main():
    parser = argparse.ArgumentParser(
        description="Smooth raw RGBD depth captures (edge-preserving)."
    )
    parser.add_argument(
        "--input", required=True,
        help="Input dir: single scene (e.g. .../0_numenta_mug) or "
             "parent of scenes if --recursive.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output dir. Must differ from input. Mirrors input structure.",
    )
    parser.add_argument(
        "--preset", choices=list(SMOOTH_PRESETS.keys()), default="medium",
        help="Smoothing intensity (default: medium).",
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Treat input as parent dir; process every subdirectory as a scene.",
    )
    parser.add_argument(
        "--no-rgb-copy", action="store_true",
        help="Don't copy rgb_*.png to output (depth only).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if input_dir == output_dir:
        parser.error("--input and --output must differ (would overwrite raw data)")

    if not input_dir.exists():
        parser.error(f"Input dir does not exist: {input_dir}")

    print(f"Preset:  {args.preset}")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")

    total = 0
    if args.recursive:
        scenes = sorted(p for p in input_dir.iterdir() if p.is_dir())
        print(f"Scenes:  {len(scenes)}")
        for scene in scenes:
            print(f"\n[{scene.name}]")
            n = process_scene(
                scene, output_dir / scene.name, args.preset,
                copy_rgb=not args.no_rgb_copy,
            )
            total += n
    else:
        n = process_scene(
            input_dir, output_dir, args.preset,
            copy_rgb=not args.no_rgb_copy,
        )
        total += n

    print(f"\nDone. {total} frame(s) processed.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Verify captured RGBD data matches the format expected by SaccadeOnImageEnvironment.

Checks:
  - RGB: PNG, RGBA mode, 640x480, uint8, alpha=255
  - Depth: 1,228,800 bytes, float32, reshape(480,640), value ranges
  - Monty process_depth_data() simulation
  - Optional: side-by-side comparison with reference dataset

Usage:
    python scripts/rgbd_verify.py \
        --captured-dir ~/tbp/data/worldimages/captured_scenes/0_numenta_mug
    python scripts/rgbd_verify.py \
        --captured-dir ~/tbp/data/worldimages/captured_scenes/0_numenta_mug \
        --reference-dir ~/tbp/data/worldimages/standard_scenes/0_numenta_mug
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXPECTED_W, EXPECTED_H = 640, 480
EXPECTED_DEPTH_BYTES = EXPECTED_W * EXPECTED_H * 4  # float32 = 4 bytes


def verify_rgb(rgb_path):
    """Verify RGB file format. Returns (pass, image_array, messages)."""
    msgs = []
    ok = True

    if not rgb_path.exists():
        return False, None, [f"FAIL: {rgb_path} not found"]

    img = Image.open(rgb_path)
    arr = np.array(img)

    # Mode
    if img.mode != "RGBA":
        msgs.append(f"FAIL: mode={img.mode}, expected RGBA")
        ok = False
    else:
        msgs.append(f"  OK: mode=RGBA")

    # Size
    if img.size != (EXPECTED_W, EXPECTED_H):
        msgs.append(f"FAIL: size={img.size}, expected ({EXPECTED_W}, {EXPECTED_H})")
        ok = False
    else:
        msgs.append(f"  OK: size={EXPECTED_W}x{EXPECTED_H}")

    # dtype
    if arr.dtype != np.uint8:
        msgs.append(f"FAIL: dtype={arr.dtype}, expected uint8")
        ok = False
    else:
        msgs.append(f"  OK: dtype=uint8")

    # Alpha channel
    if img.mode == "RGBA":
        alpha = arr[:, :, 3]
        if np.all(alpha == 255):
            msgs.append(f"  OK: alpha=255 (all pixels)")
        else:
            non255 = (alpha != 255).sum()
            msgs.append(f"WARN: {non255} pixels with alpha != 255")

    # Shape
    msgs.append(f"  OK: shape={arr.shape}")

    return ok, arr, msgs


def verify_depth(depth_path):
    """Verify depth file format. Returns (pass, depth_array, messages)."""
    msgs = []
    ok = True

    if not depth_path.exists():
        return False, None, [f"FAIL: {depth_path} not found"]

    file_size = depth_path.stat().st_size
    if file_size != EXPECTED_DEPTH_BYTES:
        msgs.append(
            f"FAIL: file_size={file_size}, expected {EXPECTED_DEPTH_BYTES}"
        )
        ok = False
    else:
        msgs.append(f"  OK: file_size={EXPECTED_DEPTH_BYTES} bytes")

    # Load
    try:
        depth = np.fromfile(str(depth_path), np.float32).reshape(
            EXPECTED_H, EXPECTED_W
        )
        msgs.append(f"  OK: reshape({EXPECTED_H}, {EXPECTED_W}) success")
    except Exception as e:
        return False, None, msgs + [f"FAIL: reshape failed: {e}"]

    # dtype
    if depth.dtype != np.float32:
        msgs.append(f"FAIL: dtype={depth.dtype}, expected float32")
        ok = False
    else:
        msgs.append(f"  OK: dtype=float32")

    # Statistics
    nan_count = np.isnan(depth).sum()
    nan_pct = nan_count / depth.size * 100
    valid = depth[~np.isnan(depth)]

    msgs.append(f"  Stats: NaN={nan_pct:.1f}% ({nan_count}/{depth.size})")
    if valid.size > 0:
        msgs.append(f"  Stats: min={valid.min():.4f}m, max={valid.max():.4f}m")
        msgs.append(f"  Stats: mean={valid.mean():.4f}m, std={valid.std():.4f}m")

        # Fraction <= 0.4m (Monty clips above this)
        close_pct = (valid <= 0.4).sum() / valid.size * 100
        msgs.append(f"  Stats: <=0.4m = {close_pct:.1f}% of valid pixels")

        if valid.min() < 0:
            msgs.append(f"WARN: negative depth values found (min={valid.min():.4f})")
    else:
        msgs.append(f"WARN: no valid (non-NaN) depth pixels")

    return ok, depth, msgs


def simulate_monty_load(rgb_path, depth_path):
    """Simulate Monty's loading logic (process_depth_data).

    Returns (pass, messages).
    """
    msgs = []
    ok = True

    try:
        # RGB load (Monty uses PIL)
        rgb_arr = np.array(Image.open(rgb_path))
        msgs.append(f"  OK: RGB loaded, shape={rgb_arr.shape}")
    except Exception as e:
        msgs.append(f"FAIL: RGB load error: {e}")
        ok = False

    try:
        # Depth load
        height, width = rgb_arr.shape[:2]
        depth = np.fromfile(str(depth_path), np.float32).reshape(height, width)
        msgs.append(f"  OK: Depth loaded, shape={depth.shape}")

        # process_depth_data simulation
        depth[np.isnan(depth)] = 10
        depth_clipped = depth.copy()
        depth_clipped[depth > 0.4] = 10

        near_pixels = (depth_clipped < 10).sum()
        total = depth_clipped.size
        msgs.append(
            f"  OK: After clipping: {near_pixels}/{total} "
            f"({near_pixels / total * 100:.1f}%) near pixels"
        )

        if near_pixels == 0:
            msgs.append(
                f"WARN: No near pixels after clipping. Object may be too far "
                f"(>0.4m) or depth invalid."
            )
    except Exception as e:
        msgs.append(f"FAIL: Depth processing error: {e}")
        ok = False

    return ok, msgs


def visualize_single(cap_dir, version, save_dir=None):
    """Visualize RGB + depth for a single capture (no reference needed)."""
    rgb_path = cap_dir / f"rgb_{version}.png"
    depth_path = cap_dir / f"depth_{version}.data"

    if not rgb_path.exists() or not depth_path.exists():
        return

    rgb = np.array(Image.open(rgb_path))[:, :, :3]
    depth = np.fromfile(str(depth_path), np.float32).reshape(EXPECTED_H, EXPECTED_W)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{cap_dir.name} / version {version}", fontsize=14)

    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("RGB")
    axes[0, 0].axis("off")

    # Depth colormap (auto-range from actual data)
    valid = depth[~np.isnan(depth)]
    depth_vis = depth.copy()
    depth_vis[np.isnan(depth_vis)] = 0
    vmax = valid.max() if valid.size > 0 else 1.0
    im = axes[0, 1].imshow(depth_vis, cmap="turbo", vmin=0, vmax=vmax)
    axes[0, 1].set_title(f"Depth (0-{vmax:.2f}m)")
    axes[0, 1].axis("off")
    plt.colorbar(im, ax=axes[0, 1], label="meters", shrink=0.8)

    # NaN mask
    nan_mask = np.isnan(depth).astype(np.uint8)
    axes[1, 0].imshow(nan_mask, cmap="Reds", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Invalid (NaN) pixels: {np.isnan(depth).mean()*100:.1f}%")
    axes[1, 0].axis("off")

    # Depth histogram
    if valid.size > 0:
        axes[1, 1].hist(valid, bins=100, range=(0, 2.0), color="steelblue", alpha=0.8)
        axes[1, 1].axvline(x=0.4, color="red", linestyle="--", label="Clip 0.4m")
        axes[1, 1].set_xlabel("Depth (m)")
        axes[1, 1].set_ylabel("Pixel count")
        axes[1, 1].set_title("Depth Distribution")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No valid depth", ha="center", va="center")

    plt.tight_layout()

    if save_dir:
        out_path = save_dir / f"verify_v{version}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def compare_with_reference(cap_dir, ref_dir, version=0, save_dir=None):
    """Side-by-side visual comparison + depth histogram."""
    cap_rgb_path = cap_dir / f"rgb_{version}.png"
    cap_depth_path = cap_dir / f"depth_{version}.data"
    ref_rgb_path = ref_dir / f"rgb_{version}.png"
    ref_depth_path = ref_dir / f"depth_{version}.data"

    cap_exists = cap_rgb_path.exists() and cap_depth_path.exists()
    ref_exists = ref_rgb_path.exists() and ref_depth_path.exists()

    if not cap_exists and not ref_exists:
        print(f"  Neither captured nor reference v={version} found, skipping.")
        return
    if not cap_exists:
        print(f"  Captured v={version} not found, skipping.")
        return
    if not ref_exists:
        print(f"  Reference v={version} not found, skipping.")
        return

    # Load both
    cap_rgb = np.array(Image.open(cap_rgb_path))[:, :, :3]
    ref_rgb = np.array(Image.open(ref_rgb_path))[:, :, :3]

    cap_depth = np.fromfile(str(cap_depth_path), np.float32).reshape(
        EXPECTED_H, EXPECTED_W
    )
    ref_depth = np.fromfile(str(ref_depth_path), np.float32).reshape(
        EXPECTED_H, EXPECTED_W
    )

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle(f"Comparison: version {version}", fontsize=14)

    # RGB
    axes[0, 0].imshow(cap_rgb)
    axes[0, 0].set_title("Captured RGB")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(ref_rgb)
    axes[0, 1].set_title("Reference RGB")
    axes[0, 1].axis("off")

    # Depth colormap
    cap_vis = cap_depth.copy()
    cap_vis[np.isnan(cap_vis)] = 0
    ref_vis = ref_depth.copy()
    ref_vis[np.isnan(ref_vis)] = 0

    im1 = axes[1, 0].imshow(cap_vis, cmap="turbo", vmin=0, vmax=1.0)
    axes[1, 0].set_title(f"Captured Depth (NaN={np.isnan(cap_depth).mean()*100:.1f}%)")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], label="meters", shrink=0.8)

    im2 = axes[1, 1].imshow(ref_vis, cmap="turbo", vmin=0, vmax=1.0)
    axes[1, 1].set_title(f"Reference Depth (NaN={np.isnan(ref_depth).mean()*100:.1f}%)")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], label="meters", shrink=0.8)

    # Depth histograms
    cap_valid = cap_depth[~np.isnan(cap_depth)]
    ref_valid = ref_depth[~np.isnan(ref_depth)]
    bins = np.linspace(0, 2.0, 100)

    axes[2, 0].hist(ref_valid, bins=bins, alpha=0.7, label="Reference", color="blue")
    axes[2, 0].hist(cap_valid, bins=bins, alpha=0.7, label="Captured", color="orange")
    axes[2, 0].set_xlabel("Depth (m)")
    axes[2, 0].set_ylabel("Pixel count")
    axes[2, 0].set_title("Depth Distribution (full range)")
    axes[2, 0].legend()

    bins_close = np.linspace(0, 0.5, 50)
    ref_close = ref_valid[ref_valid <= 0.5]
    cap_close = cap_valid[cap_valid <= 0.5]
    axes[2, 1].hist(ref_close, bins=bins_close, alpha=0.7, label="Reference", color="blue")
    axes[2, 1].hist(cap_close, bins=bins_close, alpha=0.7, label="Captured", color="orange")
    axes[2, 1].axvline(x=0.4, color="red", linestyle="--", label="Clip threshold")
    axes[2, 1].set_xlabel("Depth (m)")
    axes[2, 1].set_ylabel("Pixel count")
    axes[2, 1].set_title("Depth Distribution (near range)")
    axes[2, 1].legend()

    plt.tight_layout()

    if save_dir:
        out_path = save_dir / f"compare_v{version}.png"
        fig.savefig(out_path, dpi=150)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Verify captured RGBD data for Monty compatibility"
    )
    parser.add_argument(
        "--captured-dir",
        required=True,
        help="Path to captured scene directory (e.g. .../0_numenta_mug)",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Path to reference scene directory for comparison",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save visualization as PNG files (for headless/remote use)",
    )
    args = parser.parse_args()

    cap_dir = Path(args.captured_dir)
    if not cap_dir.exists():
        print(f"ERROR: {cap_dir} does not exist")
        sys.exit(1)

    # Find all versions
    rgb_files = sorted(cap_dir.glob("rgb_*.png"))
    if not rgb_files:
        print(f"ERROR: No rgb_*.png files found in {cap_dir}")
        sys.exit(1)

    versions = []
    for f in rgb_files:
        try:
            versions.append(int(f.stem.split("_")[1]))
        except (ValueError, IndexError):
            continue

    print(f"Scene: {cap_dir.name}")
    print(f"Versions found: {versions}")
    print("=" * 60)

    all_ok = True

    for v in versions:
        rgb_path = cap_dir / f"rgb_{v}.png"
        depth_path = cap_dir / f"depth_{v}.data"

        print(f"\n--- Version {v} ---")

        # RGB verification
        print(f"\n[RGB] {rgb_path.name}")
        rgb_ok, rgb_arr, rgb_msgs = verify_rgb(rgb_path)
        for m in rgb_msgs:
            print(m)
        if not rgb_ok:
            all_ok = False

        # Depth verification
        print(f"\n[Depth] {depth_path.name}")
        depth_ok, depth_arr, depth_msgs = verify_depth(depth_path)
        for m in depth_msgs:
            print(m)
        if not depth_ok:
            all_ok = False

        # Monty load simulation
        print(f"\n[Monty Load Test]")
        if rgb_path.exists() and depth_path.exists():
            monty_ok, monty_msgs = simulate_monty_load(rgb_path, depth_path)
            for m in monty_msgs:
                print(m)
            if not monty_ok:
                all_ok = False
        else:
            print("  SKIP: missing files")

    # Determine save directory
    save_dir = None
    if args.save_images:
        save_dir = cap_dir / "verify_output"
        save_dir.mkdir(exist_ok=True)
        print(f"\nSaving images to: {save_dir}")

    # Single-capture visualization (always, no reference needed)
    for v in versions:
        visualize_single(cap_dir, v, save_dir=save_dir)

    # Reference comparison — show all captured versions, plus all ref versions
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        if ref_dir.exists():
            # Gather reference versions
            ref_versions = []
            for f in sorted(ref_dir.glob("rgb_*.png")):
                try:
                    ref_versions.append(int(f.stem.split("_")[1]))
                except (ValueError, IndexError):
                    continue

            # Show all versions: captured vs reference
            all_versions = sorted(set(versions + ref_versions))
            for v in all_versions:
                print(f"\n{'=' * 60}")
                print(f"Reference comparison: version {v}")
                compare_with_reference(cap_dir, ref_dir, version=v,
                                       save_dir=save_dir)
        else:
            print(f"\nWARN: Reference dir not found: {ref_dir}")

    # Summary
    print(f"\n{'=' * 60}")
    if all_ok:
        print("RESULT: All checks PASSED")
    else:
        print("RESULT: Some checks FAILED (see above)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

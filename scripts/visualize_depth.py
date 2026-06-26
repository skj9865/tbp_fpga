#!/usr/bin/env python3
"""Visualize RGBD captures — single scene or side-by-side comparison.

Use cases:
  - Inspect raw captures from rgbd_capture_d405.py
  - Compare raw vs smoothed (smooth_depth.py output)
  - Compare your captures against the reference dataset (standard_scenes)

Each version (v0, v1, ...) is shown as one row:
  [RGB] [depth colormap] [(compare) RGB] [(compare) depth] [(compare) diff]

Press any key for next version, 'q' to quit (interactive mode).
Use --save-dir for headless / batch screenshot mode.

Examples:
  # Look at one scene
  python scripts/visualize_depth.py \\
      --scene ~/tbp/data/worldimages/captured_scenes/0_numenta_mug

  # Compare raw vs smoothed
  python scripts/visualize_depth.py \\
      --scene  ~/tbp/data/worldimages/captured_scenes/0_numenta_mug \\
      --compare ~/tbp/data/worldimages/captured_scenes_smooth_medium/0_numenta_mug

  # Compare D405 capture against the reference (standard_scenes)
  python scripts/visualize_depth.py \\
      --scene  ~/tbp/data/worldimages/captured_scenes/0_numenta_mug \\
      --compare ~/tbp/data/worldimages/standard_scenes/0_numenta_mug

  # Save instead of display
  python scripts/visualize_depth.py --scene <dir> --save-dir /tmp/viz
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TARGET_W, TARGET_H = 640, 480


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------

def load_depth(p):
    arr = np.fromfile(p, np.float32)
    if arr.size != TARGET_H * TARGET_W:
        raise ValueError(
            f"{p}: expected {TARGET_H*TARGET_W} float32 values, got {arr.size}"
        )
    return arr.reshape(TARGET_H, TARGET_W)


def load_rgb_bgr(p):
    img = np.array(Image.open(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------------
# Rendering helpers
# -----------------------------------------------------------------------------

def depth_to_colormap(depth_m, vmin=None, vmax=None):
    """Color-map a depth array. NaN shown as red."""
    invalid = np.isnan(depth_m)
    valid_vals = depth_m[~invalid]
    if vmin is None:
        vmin = float(np.percentile(valid_vals, 2)) if valid_vals.size else 0.05
    if vmax is None:
        vmax = float(np.percentile(valid_vals, 98)) if valid_vals.size else 0.50
    if vmax <= vmin:
        vmax = vmin + 1e-3

    d = np.where(invalid, vmin, depth_m)
    norm = np.clip((d - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    cmap[invalid] = [0, 0, 255]  # BGR red for invalid
    return cmap, vmin, vmax


def diff_colormap(diff, span=0.02):
    """Signed diff color-map ([-span, +span] meters)."""
    valid = ~np.isnan(diff)
    norm = np.clip((np.where(valid, diff, 0) + span) / (2 * span), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    # Blue = lower (-), Red = higher (+)
    cmap = cv2.applyColorMap(u8, cv2.COLORMAP_COOL)
    cmap[~valid] = [40, 40, 40]
    return cmap


def label_strip(text, width, height=28, bg=(40, 40, 40), fg=(230, 230, 230)):
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(
        img, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
        0.55, fg, 1, cv2.LINE_AA,
    )
    return img


def stats_strip(text, width, height=28, bg=(20, 20, 20), fg=(180, 220, 255)):
    img = np.full((height, width, 3), bg, dtype=np.uint8)
    cv2.putText(
        img, text, (8, height - 9), cv2.FONT_HERSHEY_SIMPLEX,
        0.50, fg, 1, cv2.LINE_AA,
    )
    return img


def depth_stats(depth_m):
    """Compact stats string for a depth array."""
    nan_pct = float(np.isnan(depth_m).mean() * 100)
    valid = depth_m[~np.isnan(depth_m)]
    if valid.size:
        s = (f"NaN {nan_pct:5.1f}%   range {valid.min():.3f}-{valid.max():.3f}m"
             f"   mean {valid.mean():.3f}m")
    else:
        s = f"NaN {nan_pct:5.1f}%   (no valid pixels)"
    return s


# -----------------------------------------------------------------------------
# Per-version frame
# -----------------------------------------------------------------------------

def build_frame(rgb_a, depth_a, label_a, version,
                rgb_b=None, depth_b=None, label_b=None):
    """Compose a single comparison frame."""
    # Use a SHARED depth range across the two depths so colors are comparable
    valid_a = depth_a[~np.isnan(depth_a)]
    if depth_b is not None:
        valid_b = depth_b[~np.isnan(depth_b)]
        combined = np.concatenate([valid_a, valid_b]) if valid_b.size else valid_a
    else:
        combined = valid_a
    if combined.size:
        vmin = float(np.percentile(combined, 2))
        vmax = float(np.percentile(combined, 98))
    else:
        vmin, vmax = 0.05, 0.5

    depth_a_cmap, _, _ = depth_to_colormap(depth_a, vmin, vmax)
    panels = [rgb_a, depth_a_cmap]
    labels = [f"[{label_a}] RGB", f"[{label_a}] depth"]
    stats = [f"v={version}", depth_stats(depth_a)]

    if depth_b is not None:
        depth_b_cmap, _, _ = depth_to_colormap(depth_b, vmin, vmax)
        panels.append(rgb_b)
        panels.append(depth_b_cmap)
        labels.append(f"[{label_b}] RGB")
        labels.append(f"[{label_b}] depth")
        stats.append("")
        stats.append(depth_stats(depth_b))

        both_valid = ~np.isnan(depth_a) & ~np.isnan(depth_b)
        diff = np.where(both_valid, depth_b - depth_a, np.nan)
        panels.append(diff_colormap(diff))
        labels.append("diff (B - A) ±2cm")
        valid_diff = diff[both_valid]
        if valid_diff.size:
            stats.append(
                f"mean |Δ| {np.abs(valid_diff).mean()*1000:6.2f}mm  "
                f"std {valid_diff.std()*1000:6.2f}mm  "
                f"both-valid {both_valid.mean()*100:5.1f}%"
            )
        else:
            stats.append("(no overlap)")

    # Stack each column: label / image / stats
    columns = []
    for lab, panel, st in zip(labels, panels, stats):
        col = np.vstack([
            label_strip(lab, panel.shape[1]),
            panel,
            stats_strip(st, panel.shape[1]),
        ])
        columns.append(col)
    return np.hstack(columns)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def scene_versions(scene_dir):
    """Discover available v indices from rgb_*.png in a scene dir."""
    vs = []
    for p in sorted(scene_dir.glob("rgb_*.png")):
        try:
            vs.append(int(p.stem.split("_")[1]))
        except (ValueError, IndexError):
            pass
    return vs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RGBD captures and compare scenes."
    )
    parser.add_argument(
        "--scene", required=True,
        help="Primary scene directory (e.g. .../0_numenta_mug)",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Optional second scene dir to compare against the primary",
    )
    parser.add_argument(
        "--label-a", default="A",
        help="Short label for primary scene in the display (default: A)",
    )
    parser.add_argument(
        "--label-b", default="B",
        help="Short label for compare scene in the display (default: B)",
    )
    parser.add_argument(
        "--versions", default=None,
        help="Comma-separated version indices (default: all available)",
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="If set, save each frame as PNG here (no interactive window)",
    )
    parser.add_argument(
        "--max-width", type=int, default=2200,
        help="Resize wide frames down to this width for display (default: 2200)",
    )
    args = parser.parse_args()

    scene_a = Path(args.scene).expanduser().resolve()
    if not scene_a.is_dir():
        parser.error(f"--scene not found: {scene_a}")

    scene_b = None
    if args.compare:
        scene_b = Path(args.compare).expanduser().resolve()
        if not scene_b.is_dir():
            parser.error(f"--compare not found: {scene_b}")

    # Determine label defaults from dir names if user kept defaults
    label_a = args.label_a if args.label_a != "A" else scene_a.name
    label_b = (
        args.label_b if args.label_b != "B"
        else (scene_b.name if scene_b else "B")
    )

    if args.versions:
        versions = [int(x) for x in args.versions.split(",")]
    else:
        versions = scene_versions(scene_a)
        if scene_b is not None:
            versions = [v for v in versions if v in set(scene_versions(scene_b))]

    if not versions:
        parser.error("No versions to display (no rgb_*.png found)")

    print(f"Scene A: {scene_a}  (label: {label_a})")
    if scene_b is not None:
        print(f"Scene B: {scene_b}  (label: {label_b})")
    print(f"Versions: {versions}")

    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)

    for v in versions:
        rgb_a = load_rgb_bgr(scene_a / f"rgb_{v}.png")
        depth_a = load_depth(scene_a / f"depth_{v}.data")
        rgb_b = depth_b = None
        if scene_b is not None:
            rgb_b = load_rgb_bgr(scene_b / f"rgb_{v}.png")
            depth_b = load_depth(scene_b / f"depth_{v}.data")

        frame = build_frame(
            rgb_a, depth_a, label_a, v,
            rgb_b=rgb_b, depth_b=depth_b, label_b=label_b,
        )

        # Print stats also to stdout for easy logging
        print(f"\n[v={v}] {label_a}: {depth_stats(depth_a)}")
        if depth_b is not None:
            print(f"      {label_b}: {depth_stats(depth_b)}")

        # Optional downscale for display
        if frame.shape[1] > args.max_width:
            scale = args.max_width / frame.shape[1]
            new_w = args.max_width
            new_h = int(frame.shape[0] * scale)
            frame_show = cv2.resize(frame, (new_w, new_h),
                                    interpolation=cv2.INTER_AREA)
        else:
            frame_show = frame

        if save_dir is not None:
            name = (f"compare_v{v}.png" if scene_b is not None
                    else f"scene_v{v}.png")
            out = save_dir / name
            cv2.imwrite(str(out), frame)
            print(f"  saved {out}")
        else:
            win = f"v={v}"
            cv2.imshow(win, frame_show)
            print("  (any key = next, 'q' = quit)")
            k = cv2.waitKey(0) & 0xFF
            cv2.destroyWindow(win)
            if k == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

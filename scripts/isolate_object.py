#!/usr/bin/env python3
"""Isolate the centered object in raw RGBD captures (mug-style scenes).

iPad's standard_scenes depth files are object-only — background and hands
are masked out. D405 raw captures include everything in view (object,
hand holding it, table, background), which makes the patch land on hand
or background and the recognition fail.

This script applies a lightweight heuristic to keep only the centered
object's depth pixels (rest set to NaN):

  1. Estimate object distance: median depth of the central ROI.
  2. Build a binary mask: depth within (obj_dist ± depth_band).
  3. Restrict to a centered bbox so a hand reaching from the edge is rejected.
  4. Connected components -> keep the blob that contains the image center.
  5. Light erosion to break thin connections (e.g. mug touching hand at the
     handle) and discard very small components.
  6. Save isolated depth (NaN outside the mask) + unchanged RGB.

Usage:
  python scripts/isolate_object.py \\
      --input  ~/tbp/data/worldimages/captured_scenes \\
      --output ~/tbp/data/worldimages/captured_scenes_isolated \\
      --recursive

  # Save diagnostic overlay images for visual verification
  python scripts/isolate_object.py \\
      --input  ~/tbp/data/worldimages/captured_scenes/0_numenta_mug \\
      --output ~/tbp/data/worldimages/captured_scenes_isolated/0_numenta_mug \\
      --debug-dir /tmp/iso_debug
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TARGET_W, TARGET_H = 640, 480


# -----------------------------------------------------------------------------
# Isolation
# -----------------------------------------------------------------------------

def isolate_depth(depth_m,
                  center_roi_frac=0.30,
                  bbox_frac=0.80,
                  depth_band=0.08,
                  erode_iter=1,
                  min_area_frac=0.005):
    """Mask depth to keep only the centered object.

    Args:
        depth_m: float32 (H,W), NaN for invalid.
        center_roi_frac: fraction of image used as the central window for
            estimating object distance.
        bbox_frac: fraction of image (centered) outside of which pixels are
            discarded (rejects hand reaching from frame edge).
        depth_band: ± distance from estimated object depth that is kept (m).
        erode_iter: number of 3x3 erosion iterations to break thin bridges.
        min_area_frac: fraction of image — components smaller than this are
            ignored when choosing the object blob.

    Returns:
        (depth_out, info) where:
            depth_out: float32 (H,W) with non-object pixels set to NaN.
            info: dict with diagnostics (obj_dist, coverage_pct, ok).
    """
    h, w = depth_m.shape
    cy, cx = h // 2, w // 2

    # 1. Object distance from central ROI median (depth in meters).
    rh, rw = int(h * center_roi_frac / 2), int(w * center_roi_frac / 2)
    patch = depth_m[cy - rh:cy + rh, cx - rw:cx + rw]
    valid_patch = patch[~np.isnan(patch)]
    if valid_patch.size < 200:
        return depth_m.copy(), dict(ok=False, reason="center too sparse",
                                    obj_dist=None, coverage_pct=0.0)
    obj_dist = float(np.median(valid_patch))

    # 2. Depth band mask.
    band_mask = (
        ~np.isnan(depth_m)
        & (depth_m >= obj_dist - depth_band)
        & (depth_m <= obj_dist + depth_band)
    )

    # 3. Restrict to centered bbox.
    bh, bw = int(h * bbox_frac / 2), int(w * bbox_frac / 2)
    bbox_mask = np.zeros((h, w), dtype=bool)
    bbox_mask[cy - bh:cy + bh, cx - bw:cx + bw] = True
    combined = band_mask & bbox_mask

    # 4. Erode lightly to break narrow bridges (hand grip area).
    if erode_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_u8 = cv2.erode(
            combined.astype(np.uint8), kernel, iterations=erode_iter,
        )
        combined = combined_u8.astype(bool)

    # 5. Connected components: choose blob containing the image center.
    num, labels = cv2.connectedComponents(combined.astype(np.uint8))
    if num < 2:
        return depth_m.copy(), dict(ok=False, reason="no components",
                                    obj_dist=obj_dist, coverage_pct=0.0)

    min_area = int(h * w * min_area_frac)
    center_label = labels[cy, cx]
    if center_label == 0:
        # Center isn't on any blob -> pick the largest centered enough blob.
        best_label = 0
        best_score = -1.0
        for lab in range(1, num):
            mask = labels == lab
            if mask.sum() < min_area:
                continue
            ys, xs = np.where(mask)
            blob_cy = ys.mean()
            blob_cx = xs.mean()
            dist_to_center = np.hypot(blob_cy - cy, blob_cx - cx)
            score = mask.sum() / max(1.0, dist_to_center)
            if score > best_score:
                best_score = score
                best_label = lab
        center_label = best_label
        if center_label == 0:
            return depth_m.copy(), dict(ok=False, reason="no centered blob",
                                        obj_dist=obj_dist, coverage_pct=0.0)

    obj_mask = labels == center_label
    if obj_mask.sum() < min_area:
        return depth_m.copy(), dict(ok=False, reason="blob too small",
                                    obj_dist=obj_dist,
                                    coverage_pct=obj_mask.mean() * 100)

    # 6. Apply mask.
    depth_out = depth_m.copy()
    depth_out[~obj_mask] = np.nan
    coverage = float(obj_mask.mean() * 100)
    info = dict(ok=True, obj_dist=obj_dist, coverage_pct=coverage,
                mask=obj_mask)
    return depth_out, info


# -----------------------------------------------------------------------------
# Debug overlay
# -----------------------------------------------------------------------------

def make_debug_image(rgb_bgr, depth_raw, depth_iso, info, label=""):
    """Side-by-side: RGB | raw depth | isolated depth | mask overlay."""
    def to_cmap(d):
        valid = ~np.isnan(d)
        vmin = 0.10
        vmax = 0.40
        norm = np.clip(
            (np.where(valid, d, vmin) - vmin) / (vmax - vmin), 0, 1,
        )
        u8 = (norm * 255).astype(np.uint8)
        cmap = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        cmap[~valid] = [0, 0, 200]  # NaN as red (BGR)
        return cmap

    raw = to_cmap(depth_raw)
    iso = to_cmap(depth_iso)

    overlay = rgb_bgr.copy()
    if info.get("ok") and "mask" in info:
        m = info["mask"]
        overlay[m] = (0.6 * overlay[m] + 0.4 *
                      np.array([0, 255, 0])).astype(np.uint8)

    def labelled(img, txt):
        h, w = img.shape[:2]
        band = np.full((28, w, 3), 30, dtype=np.uint8)
        cv2.putText(band, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return np.vstack([band, img])

    cols = [
        labelled(rgb_bgr, f"{label} RGB"),
        labelled(raw, "raw depth"),
        labelled(iso, f"isolated  cov={info.get('coverage_pct',0):.1f}%"),
        labelled(overlay, f"mask overlay  d={info.get('obj_dist') or 0:.3f}m"),
    ]
    return np.hstack(cols)


# -----------------------------------------------------------------------------
# Scene-level processing
# -----------------------------------------------------------------------------

def process_scene(input_dir, output_dir, debug_dir=None,
                  isolate_kwargs=None, copy_rgb=True):
    isolate_kwargs = isolate_kwargs or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    depth_files = sorted(input_dir.glob("depth_*.data"))
    if not depth_files:
        print(f"  (no depth files in {input_dir})")
        return 0

    for dp in depth_files:
        v = dp.stem.split("_")[1]
        depth = np.fromfile(dp, np.float32).reshape(TARGET_H, TARGET_W)

        iso, info = isolate_depth(depth, **isolate_kwargs)
        nan_in = float(np.isnan(depth).mean() * 100)
        nan_out = float(np.isnan(iso).mean() * 100)
        status = "ok" if info["ok"] else f"SKIP ({info.get('reason')})"
        d_str = (f"d={info['obj_dist']:.3f}m"
                 if info.get("obj_dist") is not None else "d=?")
        print(f"  v={v}: NaN {nan_in:5.1f}% -> {nan_out:5.1f}%   "
              f"{d_str}   cov={info['coverage_pct']:5.1f}%   {status}")

        (output_dir / f"depth_{v}.data").write_bytes(iso.astype(np.float32).tobytes())

        if copy_rgb:
            src = input_dir / f"rgb_{v}.png"
            if src.exists():
                shutil.copy(src, output_dir / src.name)

        if debug_dir is not None:
            rgb_src = input_dir / f"rgb_{v}.png"
            if rgb_src.exists():
                rgb = np.array(Image.open(rgb_src))[:, :, :3]
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                dbg = make_debug_image(rgb_bgr, depth, iso, info, label=f"v{v}")
                cv2.imwrite(str(debug_dir / f"iso_v{v}.png"), dbg)

    return len(depth_files)


def main():
    parser = argparse.ArgumentParser(
        description="Isolate centered object in raw RGBD captures."
    )
    parser.add_argument("--input", required=True,
                        help="Single scene dir or parent of scenes")
    parser.add_argument("--output", required=True,
                        help="Output dir (mirrors input structure)")
    parser.add_argument("--recursive", action="store_true",
                        help="Treat --input as parent of multiple scenes")
    parser.add_argument("--debug-dir", default=None,
                        help="If set, save side-by-side debug PNGs here")
    parser.add_argument("--depth-band", type=float, default=0.08,
                        help="± depth band around object distance (m, default 0.08)")
    parser.add_argument("--bbox-frac", type=float, default=0.80,
                        help="Fraction of image kept around center (default 0.80)")
    parser.add_argument("--center-roi-frac", type=float, default=0.30,
                        help="Central ROI used to estimate object depth (default 0.30)")
    parser.add_argument("--erode-iter", type=int, default=1,
                        help="Erosion iterations to break thin bridges (default 1)")
    parser.add_argument("--no-rgb-copy", action="store_true",
                        help="Don't copy rgb_*.png alongside")
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if input_dir == output_dir:
        parser.error("--input and --output must differ")
    if not input_dir.exists():
        parser.error(f"Input not found: {input_dir}")

    debug_dir = Path(args.debug_dir).expanduser().resolve() \
        if args.debug_dir else None

    kw = dict(
        depth_band=args.depth_band,
        bbox_frac=args.bbox_frac,
        center_roi_frac=args.center_roi_frac,
        erode_iter=args.erode_iter,
    )

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if debug_dir:
        print(f"Debug:  {debug_dir}")
    print(f"Params: {kw}")

    total = 0
    if args.recursive:
        scenes = sorted(p for p in input_dir.iterdir() if p.is_dir())
        print(f"Scenes: {len(scenes)}")
        for sc in scenes:
            print(f"\n[{sc.name}]")
            total += process_scene(
                sc, output_dir / sc.name,
                debug_dir=(debug_dir / sc.name) if debug_dir else None,
                isolate_kwargs=kw,
                copy_rgb=not args.no_rgb_copy,
            )
    else:
        total += process_scene(
            input_dir, output_dir,
            debug_dir=debug_dir, isolate_kwargs=kw,
            copy_rgb=not args.no_rgb_copy,
        )

    print(f"\nDone. {total} frame(s) processed.")


if __name__ == "__main__":
    main()

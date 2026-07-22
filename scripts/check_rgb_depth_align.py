#!/usr/bin/env python3
"""Measure whether a capture's RGB and depth cover the same field of view.

Monty projects the *depth* image to 3D with a single HFOV (MONTY_HFOV) and
samples hsv from the *RGB* image at the same pixel index. If the two images
span different fields, both go wrong: 3D locations carry a systematic scale
error (location tolerance is only 0.015m) and hue is read from the wrong
point, worsening toward the frame edges.

This is a real risk on the OAK-D Pro: IMX378 is natively 4056x3040 (4:3),
but THE_4_K is a 16:9 crop of it, so a 4:3 preview taken from 4K loses
vertical FOV — while StereoDepth aligned to CAM_A keeps the native field.
Symptom: a close object is cut off top/bottom in RGB but whole in depth.

Method: extract edges from RGB (Canny) and from depth (thresholded gradient),
then search the scale about the image centre that maximises normalised
cross-correlation. Scale ~1.00 in both axes means the fields agree.

Usage:
  python scripts/check_rgb_depth_align.py --scene ~/tbp/data/worldimages/captured_scenes_oak/0_numenta_mug
  python scripts/check_rgb_depth_align.py --scene <dir> --save /tmp/align.png

Interpretation:
  sy ~ 1.00            -> RGB and depth share the field. Good.
  sy ~ 1.2-1.35        -> depth spans more vertical FOV than RGB, the classic
                          16:9-crop-vs-4:3 mismatch. Try a 4:3 colour sensor
                          mode (capture_and_infer.py --color-res THE_1352X1012)
                          and re-measure.
  low ncc everywhere   -> scene too low-texture to judge; retry on a scene with
                          strong depth discontinuities.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TARGET_W, TARGET_H = 640, 480


def edges_rgb(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.Canny(gray, 40, 120).astype(np.float32), (9, 9), 0)


def edges_depth(depth_m):
    d = np.nan_to_num(depth_m, nan=0.0)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    strong = (mag > np.percentile(mag, 97)).astype(np.float32)
    return cv2.GaussianBlur(strong, (9, 9), 0)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    return float((a * b).sum() / denom)


def scale_about_centre(img, sx, sy):
    m = np.array(
        [[sx, 0, (1 - sx) * TARGET_W / 2], [0, sy, (1 - sy) * TARGET_H / 2]],
        np.float32,
    )
    return cv2.warpAffine(img, m, (TARGET_W, TARGET_H))


def measure(bgr, depth_m, coarse=True):
    """Return (best_sx, best_sy, best_ncc, ncc_at_unity)."""
    e_rgb, e_depth = edges_rgb(bgr), edges_depth(depth_m)
    unity = ncc(e_rgb, e_depth)

    step = 0.02 if coarse else 0.01
    best = (1.0, 1.0, -2.0)
    for sy in np.arange(0.80, 1.41, step):
        for sx in np.arange(0.90, 1.61, 0.04):
            c = ncc(e_rgb, scale_about_centre(e_depth, sx, sy))
            if c > best[2]:
                best = (float(sx), float(sy), c)
    return best[0], best[1], best[2], unity


def load_pair(scene_dir, version):
    scene = Path(scene_dir).expanduser()
    rgb_p = scene / f"rgb_{version}.png"
    depth_p = scene / f"depth_{version}.data"
    if not rgb_p.exists() or not depth_p.exists():
        raise FileNotFoundError(f"missing rgb_{version}.png / depth_{version}.data in {scene}")
    rgb = np.array(Image.open(rgb_p))[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_m = np.fromfile(depth_p, np.float32).reshape(TARGET_H, TARGET_W)
    return bgr, depth_m


def verdict(sy, best_ncc, unity):
    if best_ncc < 0.02:
        return "INCONCLUSIVE — scene too low-texture; retry with strong depth edges"
    if abs(sy - 1.0) <= 0.06:
        return "ALIGNED — RGB and depth share the field"
    if sy > 1.06:
        return (f"MISMATCH — depth spans ~{sy:.2f}x more vertical FOV than RGB; "
                f"try --color-res THE_1352X1012 (4:3) and re-measure")
    return f"MISMATCH — depth spans ~{sy:.2f}x the vertical FOV of RGB"


def main():
    ap = argparse.ArgumentParser(
        description="Measure RGB vs depth field-of-view agreement in a capture."
    )
    ap.add_argument("--scene", required=True, help="Scene dir (e.g. .../0_numenta_mug)")
    ap.add_argument("--versions", default=None,
                    help="Comma-separated versions (default: all rgb_*.png found)")
    ap.add_argument("--fine", action="store_true", help="Finer scale search (slower)")
    ap.add_argument("--save", default=None, help="Save an overlay image here")
    args = ap.parse_args()

    scene = Path(args.scene).expanduser()
    if args.versions:
        versions = [int(v) for v in args.versions.split(",")]
    else:
        versions = sorted(int(p.stem.split("_")[1]) for p in scene.glob("rgb_*.png"))
    if not versions:
        ap.error(f"no rgb_*.png under {scene}")

    sys_ = []
    for v in versions:
        bgr, depth_m = load_pair(scene, v)
        sx, sy, best, unity = measure(bgr, depth_m, coarse=not args.fine)
        sys_.append((sy, best))
        print(f"  v{v}: best scale sx={sx:.2f} sy={sy:.2f}  ncc={best:.4f}  "
              f"(ncc at 1.00 = {unity:.4f})  -> {verdict(sy, best, unity)}")

        if args.save and v == versions[0]:
            e_rgb = edges_rgb(bgr)
            warped = scale_about_centre(edges_depth(depth_m), sx, sy)
            ov = bgr.copy()
            ov[e_rgb > 0.15] = (0, 255, 255)      # RGB edges = yellow
            ov[warped > 0.15] = (0, 0, 255)       # depth edges (fitted) = red
            cv2.imwrite(str(Path(args.save).expanduser()), ov)
            print(f"  saved overlay -> {args.save}")

    trusted = [s for s, b in sys_ if b >= 0.02]
    if trusted:
        print(f"\n  median vertical scale over {len(trusted)} usable frame(s): "
              f"{np.median(trusted):.2f}")
    else:
        print("\n  no frame had enough texture to judge")


if __name__ == "__main__":
    main()

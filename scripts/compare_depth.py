#!/usr/bin/env python3
"""Compare captured vs reference depth data."""
import numpy as np
from PIL import Image
from pathlib import Path

cap_dir = Path.home() / 'tbp/data/worldimages/captured_scenes/0_numenta_mug'
ref_dir = Path.home() / 'tbp/data/worldimages/standard_scenes/0_numenta_mug'

print('=' * 70)

for v in range(4):
    print(f'\n--- Version {v} ---')

    ref_depth = np.fromfile(str(ref_dir / f'depth_{v}.data'), np.float32).reshape(480, 640)
    cap_depth = np.fromfile(str(cap_dir / f'depth_{v}.data'), np.float32).reshape(480, 640)

    for label, d in [('Reference', ref_depth), ('Captured', cap_depth)]:
        valid = d[~np.isnan(d)]
        nan_pct = np.isnan(d).sum() / d.size * 100
        if valid.size > 0:
            d_monty = d.copy()
            d_monty[np.isnan(d_monty)] = 10
            d_monty[d_monty > 0.4] = 10
            on_obj = (d_monty < 10).sum()
            on_obj_pct = on_obj / d_monty.size * 100
            near = valid[valid <= 0.4]
            print(f'  {label:10s}: min={valid.min():.4f}m max={valid.max():.3f}m '
                  f'mean={valid.mean():.4f}m NaN={nan_pct:.1f}% '
                  f'on_object={on_obj_pct:.1f}%({on_obj}px)')
            if near.size > 0:
                print(f'             near(<=0.4m): mean={near.mean():.4f}m '
                      f'std={near.std():.4f}m count={near.size}')
        else:
            print(f'  {label:10s}: ALL NaN')

print(f'\n{"=" * 70}')
print('SUMMARY: on_object pixels (Monty clips >0.4m to 10.0)')
print(f'{"=" * 70}')
for v in range(4):
    ref_d = np.fromfile(str(ref_dir / f'depth_{v}.data'), np.float32).reshape(480, 640)
    cap_d = np.fromfile(str(cap_dir / f'depth_{v}.data'), np.float32).reshape(480, 640)
    ref_d[np.isnan(ref_d)] = 10; ref_d[ref_d > 0.4] = 10
    cap_d[np.isnan(cap_d)] = 10; cap_d[cap_d > 0.4] = 10
    ref_on = (ref_d < 10).sum()
    cap_on = (cap_d < 10).sum()
    ratio = cap_on / ref_on * 100 if ref_on > 0 else 0
    print(f'  v{v}: ref={ref_on:6d}px  cap={cap_on:6d}px  cap/ref={ratio:.0f}%')

#!/usr/bin/env python3
"""Profile monty_inference.py with multithreading disabled to see true compute costs."""
import cProfile
import pstats
import io
import sys
import os

# Override sys.argv to run 1 episode only
sys.argv = [
    "monty_inference.py",
    "--max-episodes", "1",
    "--log-level", "WARNING",
]

# Add scripts/ to path for monty_inference imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

# Monkey-patch to disable multithreading before importing
import monty_inference

_orig_create_lm = monty_inference.create_learning_module

def create_lm_no_threading():
    lm = _orig_create_lm()
    lm.use_multithreading = False
    print("[PROFILER] Multithreading disabled for accurate profiling")
    return lm

monty_inference.create_learning_module = create_lm_no_threading

pr = cProfile.Profile()
pr.enable()
monty_inference.main()
pr.disable()

# Top by self time (most useful for FPGA targeting)
s = io.StringIO()
ps = pstats.Stats(pr, stream=s)
ps.sort_stats('tottime')
print("\n" + "=" * 80)
print("TOP 50 BY SELF TIME (where CPU actually spends time)")
print("=" * 80)
ps.print_stats(50)
print(s.getvalue())

# Top by cumulative time
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2)
ps2.sort_stats('cumulative')
print("\n" + "=" * 80)
print("TOP 50 BY CUMULATIVE TIME")
print("=" * 80)
ps2.print_stats(50)
print(s2.getvalue())

pr.dump_stats('monty_profile.prof')
print("Full profile saved to monty_profile.prof")

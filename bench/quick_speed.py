"""Warm single-request timings only (1 and 3 questions, median of 50): for A/B checks.

    .venv/Scripts/python bench/quick_speed.py d1-2B [--fp8] [--compile]
"""
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases  # noqa: E402
import torch  # noqa: E402

import d1_fast_cuda  # noqa: E402

name = sys.argv[1]
m = d1_fast_cuda.load(f"LiquidAI/{name}", fp8="--fp8" in sys.argv)
if "--compile" in sys.argv:
    m.compile()
out = {}
with torch.inference_mode():
    for label, qs in (("1q", {"team": cases.TEAM}), ("3q", cases.THREE)):
        for _ in range(10):
            m.system_one(cases.TEXT_CASES[0][0], qs)
        ts = []
        for _ in range(50):
            torch.cuda.synchronize()
            t = time.perf_counter()
            m.system_one(cases.TEXT_CASES[0][0], qs)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        out[label] = round(statistics.median(ts), 2)
print(name, " ".join(sys.argv[2:]), f"kernel={os.environ.get('D1_FP8_KERNEL', '1')}",
      f"names={os.environ.get('D1_FP8_NAMES', 'all')}", out, flush=True)

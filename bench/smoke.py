"""Quick check of the fast engine against the stored references (no timing)."""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import answers  # noqa: E402
import cases  # noqa: E402
import torch  # noqa: E402

import d1_fast_cuda  # noqa: E402

name = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else "fast"
m = d1_fast_cuda.load(f"LiquidAI/{name}", dtype=torch.float32 if "--fp32" in sys.argv else torch.bfloat16,
                      fp8="--fp8" in sys.argv)
media_kw = "media" if name.endswith("2B") else "images"
t = time.time()
with torch.inference_mode():
    out = answers.run_cases(m.system_one, cases.all_cases(name), media_kw)
print(f"{len(out)} cases in {time.time() - t:.1f}s")
answers.save(f"{name}-{label}", out)
answers.report(f"{name}-torch-cpu", f"{name}-{label}")

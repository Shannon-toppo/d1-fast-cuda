"""Where one warm single question spends its time: host (prompt, readout) vs GPU pass."""
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
from d1_fast_cuda import graphs as G  # noqa: E402

name = sys.argv[1]
m = d1_fast_cuda.load(f"LiquidAI/{name}", fp8="--fp8" in sys.argv)
fast = m.d1_fast
if "--compile" in sys.argv:
    m.compile()
state, qs = cases.TEXT_CASES[0][0], {"team": cases.TEAM}
if "--three" in sys.argv:
    qs = cases.THREE
spent = []
orig = G.Graphs.run


def timed_run(self, key, inputs, forward):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = orig(self, key, inputs, forward)
    torch.cuda.synchronize()
    spent.append((time.perf_counter() - t) * 1000)
    return out


G.Graphs.run = timed_run
with torch.inference_mode():
    for _ in range(10):
        m.system_one(state, qs)
    spent.clear()
    tot = []
    for _ in range(30):
        torch.cuda.synchronize()
        t = time.perf_counter()
        m.system_one(state, qs)
        torch.cuda.synchronize()
        tot.append((time.perf_counter() - t) * 1000)
print(f"total {statistics.median(tot):.2f} ms, graph run (incl copies) {statistics.median(spent):.2f} ms, keys {list(fast.graphs.cache)}")
# replay only
key = next(iter(fast.graphs.cache))
g = fast.graphs.cache[key][0]
torch.cuda.synchronize()
ts = []
for _ in range(30):
    t = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t) * 1000)
print(f"replay only {statistics.median(ts):.2f} ms")
if "--cprofile" in sys.argv:
    import cProfile, pstats
    with torch.inference_mode():
        cProfile.run("for _ in range(20): m.system_one(state, qs)", "prof.out")
    pstats.Stats("prof.out").sort_stats("cumulative").print_stats(30)

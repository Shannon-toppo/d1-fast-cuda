"""The CUTLASS fp8 GEMM (d1_fast_cuda.cutlass_fp8) against cuBLAS' `_scaled_mm` at d1's shapes:
correctness, then time per call from a CUDA graph with weights beyond L2, and the share
of the 4070 Ti's 504 GB/s that reading the e4m3 weight takes.

    .venv/Scripts/python bench/cutlass_fp8_bench.py [--m 128,192]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from d1_fast_cuda import cutlass_fp8 as C  # noqa: E402

F8 = torch.float8_e4m3fn
SHAPES = [(2048, 3088), (2048, 2048), (2048, 6144), (2048, 15360), (7680, 2048), (2048, 3072), (2048, 21504),
          (10752, 2048)]  # (K, N)
MS = [int(a) for a in sys.argv[sys.argv.index("--m") + 1].split(",")] if "--m" in sys.argv else [16, 64, 128, 192, 256]
POOL_BYTES = 256 << 20
BW = 504e9


def timeit(fn, n, reps=30, rounds=3):
    for i in range(3):
        fn(i % n)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i % n)
    best = float("inf")
    for _ in range(rounds):
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        best = min(best, a.elapsed_time(b) / reps * 1000)
    return best


def main():
    names = C.configs()
    one = torch.ones((), device="cuda")
    # correctness, every config
    x8 = (torch.randn(100, 2048, device="cuda") * 0.5).to(F8)
    w8 = (torch.randn(3088, 2048, device="cuda") * 0.5).to(F8)
    ref = x8.double() @ w8.double().T
    cb = torch._scaled_mm(x8, w8.T, scale_a=one, scale_b=one, out_dtype=torch.float32).double()
    e_cb = ((cb - ref).abs().mean() / ref.abs().mean()).item()
    worst = 0.0
    for i, name in enumerate(names):
        err = ((C.mm(x8, w8, i).double() - ref).abs().mean() / ref.abs().mean()).item()
        assert err < max(4 * e_cb, 1e-6), (name, err, e_cb)
        worst = max(worst, err)
    print(f"all configs match a float64 reference: mean error {worst:.2e} at worst, cuBLAS {e_cb:.2e}")
    for k, n in SHAPES:
        count = max(4, POOL_BYTES // (n * k))
        ws = [(torch.randn(n, k, device="cuda") * 0.5).to(F8) for _ in range(count)]
        for m in MS:
            x8 = (torch.randn(m, k, device="cuda") * 0.5).to(F8)
            t_cb = timeit(lambda i: torch._scaled_mm(x8, ws[i].T, scale_a=one, scale_b=one,
                                                     out_dtype=torch.float32), count)
            best = (float("inf"), "")
            for c, name in enumerate(names):
                try:
                    t = timeit(lambda i: C.mm(x8, ws[i], c), count)
                except RuntimeError:
                    continue
                if t < best[0]:
                    best = (t, name)
            ideal = n * k / BW * 1e6
            print(f"M={m:4d} K={k:6d} N={n:6d}  cublas {t_cb:7.1f}us ({ideal / t_cb:4.0%})  "
                  f"cutlass {best[0]:7.1f}us ({ideal / best[0]:4.0%})  {best[1]}", flush=True)
        del ws
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""Tile table for d1_fast_cuda.fp8 on this GPU, per d1 projection shape and row bucket:

- `shapes` / `triton_us` (buckets `ms`): the fastest Triton configuration and its time,
  with cuBLAS' `_scaled_mm` (float32 out, as `lm.linear` calls it) in `cublas_us`.
- `gemm` / `gemm_us` (buckets `gemm_ms`, up to batch sizes): for the float32 matmul
  `fp8.gemm32`, the CUTLASS configuration (d1_fast_cuda.cutlass_fp8) where it beats cuBLAS
  by 3%, else "cublas", and the time of the choice.

Weights come from a pool larger than the L2 cache, as in a real pass; calls replay from a
CUDA graph. Writes d1_fast_cuda/fp8_table.json.

    .venv/Scripts/python bench/fp8_tune.py [--gemm-only]   # --gemm-only keeps the Triton part
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402
import triton  # noqa: E402

from d1_fast_cuda import fp8  # noqa: E402

SHAPES = [(2048, 3088), (2048, 2048), (2048, 6144), (2048, 15360), (7680, 2048), (2048, 3072), (2048, 21504),
          (10752, 2048)]  # (K, N)
MS = [16, 32, 64, 96, 128, 160, 192, 256, 320, 384, 512]
GEMM_MS = MS + [768, 1024, 2048, 4096, 8192]
CANDIDATES = [  # (BM, BN, BK, SPLIT, warps, stages): the winners of a wider sweep
    (16, 64, 128, 1, 4, 4), (16, 64, 256, 1, 4, 3), (16, 128, 128, 1, 4, 4), (16, 64, 128, 2, 4, 4),
    (32, 32, 128, 1, 8, 3), (32, 64, 128, 1, 4, 3), (32, 64, 256, 1, 8, 3), (32, 64, 128, 2, 4, 3),
    (32, 64, 128, 4, 4, 3), (32, 128, 128, 1, 4, 3),
    (64, 64, 128, 1, 4, 3), (64, 64, 256, 1, 8, 3), (64, 128, 128, 1, 4, 3), (64, 128, 256, 1, 4, 3),
    (64, 128, 128, 1, 8, 4), (64, 64, 128, 2, 4, 3),
    (128, 64, 128, 1, 4, 3), (128, 128, 128, 1, 8, 3),
    # deeper pipelines and split K for narrow outputs at 100-300 rows
    (32, 64, 128, 1, 4, 5), (64, 64, 128, 1, 4, 4), (64, 64, 128, 1, 4, 5), (64, 64, 256, 1, 4, 3),
    (64, 64, 128, 2, 4, 4), (64, 64, 256, 2, 8, 3), (64, 128, 128, 2, 4, 3), (32, 128, 128, 2, 4, 3),
    (64, 32, 128, 1, 4, 4), (64, 32, 256, 1, 4, 3), (128, 32, 128, 1, 4, 3),
    (64, 64, 128, 3, 4, 3), (32, 64, 128, 3, 4, 4),
]
POOL_BYTES = 256 << 20
OUT = Path(__file__).resolve().parents[1] / "d1_fast_cuda" / "fp8_table.json"


def timeit(fn, n, reps=30, rounds=3):
    """GPU time per call, launched from a captured CUDA graph as the model's passes are
    (Python launch overhead, ~20 us a Triton launch on Windows, would hide the kernel)."""
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
    del g
    return best


def tune_gemm(table):
    from d1_fast_cuda import cutlass_fp8

    names = cutlass_fp8.configs()
    one = torch.ones((), device="cuda")
    table.update(gemm_ms=GEMM_MS, gemm={}, gemm_us={})
    for k, n in SHAPES:
        count = max(4, POOL_BYTES // (n * k))
        ws = [(torch.randn(n, k, device="cuda") * 0.5).to(fp8.F8) for _ in range(count)]
        row, t_row = [], []
        for m in GEMM_MS:
            x8 = (torch.randn(m, k, device="cuda") * 0.5).to(fp8.F8)
            t_cublas = timeit(lambda i: torch._scaled_mm(x8, ws[i].T, scale_a=one, scale_b=one,
                                                         out_dtype=torch.float32), count)
            best = (t_cublas * 0.97, "cublas")
            for c, name in enumerate(names):
                try:
                    t = timeit(lambda i: cutlass_fp8.mm(x8, ws[i], c), count)
                except RuntimeError:
                    continue
                if t < best[0]:
                    best = (t, name)
            row.append(best[1])
            t_row.append(round(min(best[0], t_cublas), 2))
            print(f"M={m:5d} K={k:6d} N={n:6d}  cublas {t_cublas:7.1f}us  pick {t_row[-1]:7.1f}us  {best[1]}",
                  flush=True)
        table["gemm"][f"{k}x{n}"] = row
        table["gemm_us"][f"{k}x{n}"] = t_row
        del ws
        torch.cuda.empty_cache()


def main():
    if "--gemm-only" in sys.argv:
        table = json.loads(OUT.read_text())
        tune_gemm(table)
        OUT.write_text(json.dumps(table, indent=1))
        print("wrote", OUT)
        return
    table = {"device": torch.cuda.get_device_name(), "ms": MS, "shapes": {}, "triton_us": {}, "cublas_us": {}}
    one = torch.ones((), device="cuda")
    for k, n in SHAPES:
        count = max(4, POOL_BYTES // (n * k))
        q = [fp8.quantize_rows(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)) for _ in range(count)]
        row, t_tr, t_cb = [], [], []
        for m in MS:
            x8, sx = fp8.quantize_rows(torch.randn(m, k, device="cuda", dtype=torch.bfloat16))
            t_cublas = timeit(lambda i: torch._scaled_mm(x8, q[i][0].T, scale_a=one, scale_b=one,
                                                         out_dtype=torch.float32), count)
            best = (float("inf"), None)
            for bm, bn, bk, split, warps, stages in CANDIDATES:
                if bm > max(16, 2 * m) or bk * split * 2 > k:
                    continue
                cfg = dict(BM=bm, BN=bn, BK=bk, SPLIT=split, GROUP=8, num_warps=warps, num_stages=stages)
                try:
                    t = timeit(lambda i: fp8.mm(x8, sx, q[i][0], q[i][1], cfg), count)
                except triton.runtime.errors.OutOfResources:  # too much shared memory on this GPU
                    continue
                if t < best[0]:
                    best = (t, cfg)
            row.append(best[1])
            t_tr.append(round(best[0], 2))
            t_cb.append(round(t_cublas, 2))
            name = ",".join(str(v) for v in best[1].values())
            print(f"M={m:4d} K={k:6d} N={n:6d}  cublas {t_cublas:7.1f}us  triton {best[0]:7.1f}us  {name}", flush=True)
        table["shapes"][f"{k}x{n}"] = row
        table["triton_us"][f"{k}x{n}"] = t_tr
        table["cublas_us"][f"{k}x{n}"] = t_cb
        del q
        torch.cuda.empty_cache()
    tune_gemm(table)
    OUT.write_text(json.dumps(table, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

"""cuBLAS bf16 matmul time for d1's linear shapes at a single question's row counts,
as a chain (each product waits for the last), against the weight-read floor."""
import os
import statistics
import sys

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
SHAPES = {  # (K, N): what
    "2B": {(2048, 4096 + 2 * 512 + 16): "qkv+gate", (2048, 2048): "out / o", (2048, 6144): "conv in",
           (2048, 15360): "mlp up", (7680, 2048): "mlp down"},
    "3B": {(2048, 3072): "qkv", (2048, 2048): "out", (2048, 6144): "conv in", (2048, 21504): "mlp up",
           (10752, 2048): "mlp down"},
}
BW = float(os.environ.get("BW", "504"))  # GB/s, RTX 4070 Ti


def chain(m, k, n, reps=50):
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(4)]  # rotate: no L2 reuse
    for _ in range(5):
        x @ ws[0].T
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for i in range(reps):
        y = x @ ws[i % 4].T
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps


for model, shapes in SHAPES.items():
    print(model)
    for (k, n), what in shapes.items():
        floor = k * n * 2 / BW / 1e6
        cells = []
        for m in (32, 64, 128, 256):
            t = chain(m, k, n)
            cells.append(f"M={m}: {t*1000:6.1f}us ({floor/t*100:3.0f}% bw, {2*m*k*n/t/1e9:5.1f} TF)")
        print(f"  {what:9s} ({k},{n}) floor {floor*1000:5.1f}us | " + " | ".join(cells))

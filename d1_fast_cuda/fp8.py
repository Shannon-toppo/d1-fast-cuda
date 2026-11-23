"""Float8 matmuls for Ada (sm_89) and newer: a Triton kernel for few rows, and CUTLASS
or cuBLAS for the float32 product otherwise.

A pass over one short state is 100-300 rows, so each projection is bound by reading
its weight. cuBLAS' fp8 kernels (`torch._scaled_mm`) reach about two thirds of the
memory bandwidth there, and on Ada take one scale per tensor only, so the per-row
scales have to be applied in a separate pass over a float32 product. This module's
kernel multiplies e4m3 by e4m3 on the fp8 tensor cores and applies the activations'
per-row and the weight's per-channel scales in its epilogue, writing bfloat16. The
fp8 MMA accumulates with fewer bits than float32 (as on Hopper), so each K tile's
product is added to a float32 accumulator separately, as cuBLAS does.

Which tile shape is fastest depends on the projection and the row count:
`fp8_table.json` (made by bench/fp8_tune.py, on an RTX 4070 Ti) holds the best per
d1 projection shape and row bucket, with its time and cuBLAS'. The kernel wins up to
a few hundred rows; past that the matmuls turn bound by arithmetic and cuBLAS is as
fast or faster. `Table.rows` picks, per model, the row range in which the kernel
runs for its wide projections; elsewhere `lm.linear` takes the float32 product from
`gemm32` and leaves the scales to torch.compile to fuse.

`gemm32` is cuBLAS' `_scaled_mm`, or CUTLASS (`cutlass_fp8`, Stream-K) where the table
measured it faster: mainly the narrow, long-K MLP down projections (N = 2048), whose
few output tiles leave SMs idle in cuBLAS' data-parallel kernels. CUTLASS needs nvcc,
a host compiler and its headers to build; without them `gemm32` is cuBLAS alone.

`split > 1` divides K over several programs (float32 partials summed afterwards) so
narrow outputs (N = 2048) still fill every SM.
"""

from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import torch

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9 warns unless it is set
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

Tensor = torch.Tensor
F8 = torch.float8_e4m3fn
F8_MAX = 448.0
KERNEL = os.environ.get("D1_FP8_KERNEL", "1") != "0"  # "0": no Triton kernel, for comparison
CUTLASS = os.environ.get("D1_CUTLASS_GEMM", "1") != "0"  # "0": no CUTLASS, for comparison
WIDE = 3072  # outputs at least this wide may take the Triton kernel (it is weak on narrow ones)
MARGIN = 0.95
MIN_ROWS = 32
_TABLE = Path(__file__).with_name("fp8_table.json")


def supported(device=None) -> bool:
    """fp8 tensor cores: compute capability 8.9 (Ada) or newer."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability(device) >= (8, 9)


@triton.jit
def _mm(x_ptr, w_ptr, sx_ptr, sw_ptr, y_ptr, M, N, K, sxm, swn, sym, syk,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    pk = tl.program_id(1)
    nm = tl.cdiv(M, BM)
    nn = tl.cdiv(N, BN)
    # grouped order: consecutive programs share weight columns while M has several tiles
    per_group = GROUP * nn
    first = (pid // per_group) * GROUP
    size = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per_group) % size
    pn = (pid % per_group) // size

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm = rm < M
    mn = rn < N
    x_rows = x_ptr + rm[:, None].to(tl.int64) * sxm
    w_cols = w_ptr + rn[None, :].to(tl.int64) * swn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    chunk = tl.cdiv(tl.cdiv(K, BK), SPLIT) * BK
    k0 = pk * chunk
    k1 = tl.minimum(k0 + chunk, K)
    for k in range(k0, k1, 2 * BK):
        # two K tiles per partial sum, added to `acc` in float32: the fp8 MMA's own
        # accumulation is coarser (a lone `acc += dot` is folded back into the MMA)
        part = tl.zeros((BM, BN), dtype=tl.float32)
        for j in tl.static_range(2):
            kk = k + j * BK + rk
            mk = kk < k1
            x = tl.load(x_rows + kk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
            w = tl.load(w_cols + kk[:, None], mask=mk[:, None] & mn[None, :], other=0.0)  # (BK, BN)
            part = tl.dot(x, w, part)
        acc += part
    out = rm[:, None].to(tl.int64) * sym + rn[None, :]
    if SPLIT == 1:
        acc = acc * tl.load(sw_ptr + rn, mask=mn, other=0.0)[None, :]
        acc = acc * tl.load(sx_ptr + rm, mask=mm, other=0.0)[:, None]
        tl.store(y_ptr + out, acc.to(tl.bfloat16), mask=mm[:, None] & mn[None, :])
    else:
        tl.store(y_ptr + pk.to(tl.int64) * syk + out, acc, mask=mm[:, None] & mn[None, :])


def default_config(m: int, n: int, k: int) -> dict:
    """Tiles for a shape the table lacks."""
    bm = 16 if m <= 16 else 32 if m <= 128 else 64
    bn = 64
    split = 1
    while triton.cdiv(m, bm) * triton.cdiv(n, bn) * split < 120 and k // (128 * split * 2) >= 4:
        split *= 2
    return dict(BM=bm, BN=bn, BK=128, SPLIT=split, GROUP=8, num_warps=4, num_stages=3)


def mm(x8: Tensor, sx: Tensor, w8: Tensor, sw: Tensor, cfg: dict | None = None) -> Tensor:
    """(M, K) e4m3 rows times an (N, K) e4m3 weight, transposed, scaled back by the
    rows' scales `sx` (M,) and the weight's `sw` (N,): bfloat16 (M, N)."""
    m, k = x8.shape
    n = w8.shape[0]
    c = dict(cfg or default_config(m, n, k))
    split = c["SPLIT"]
    grid = (triton.cdiv(m, c["BM"]) * triton.cdiv(n, c["BN"]), split)
    if split == 1:
        y = torch.empty((m, n), device=x8.device, dtype=torch.bfloat16)
        _mm[grid](x8, w8, sx, sw, y, m, n, k, x8.stride(0), w8.stride(0), y.stride(0), 0, **c)
        return y
    part = torch.empty((split, m, n), device=x8.device, dtype=torch.float32)
    _mm[grid](x8, w8, sx, sw, part, m, n, k, x8.stride(0), w8.stride(0), part.stride(1), part.stride(0), **c)
    return (part.sum(0) * sw * sx[:, None]).to(torch.bfloat16)


class Table:
    """`fp8_table.json`: per "KxN", one entry per row bucket (`ms`)."""

    def __init__(self, path: Path = _TABLE):
        data = json.loads(path.read_text()) if path.exists() else {"ms": [], "shapes": {}}
        self.ms, self.shapes = data["ms"], data["shapes"]
        self.triton_us, self.cublas_us = data.get("triton_us", {}), data.get("cublas_us", {})
        self.gemm_ms, self.gemm, self.gemm_us = data.get("gemm_ms", []), data.get("gemm", {}), data.get("gemm_us", {})
        self._cutlass = None  # config name -> index, once the library is loaded

    def _alt_us(self, key: str, i: int, m: int, n: int) -> float:
        """`gemm32`'s time at Triton bucket i, plus writing and reading back float32
        instead of bfloat16 (4 more bytes per output) for the scales applied after it."""
        t = self.cublas_us[key][i]
        if self.cutlass_ready() and key in self.gemm_us and m in self.gemm_ms:
            t = min(t, self.gemm_us[key][self.gemm_ms.index(m)])
        return t + m * n * 4 / 504e9 * 1e6

    def rows(self, shapes: list[tuple[int, int]]) -> tuple[int, int]:
        """The row range (lo, hi] in which the kernel runs for a model whose projections
        are `shapes` [(n, k), ...] (those at least `WIDE` wide): the longest run of row
        buckets above `MIN_ROWS` in which their summed kernel time never exceeds
        `gemm32`'s (with its float32 output) and somewhere stays `MARGIN` below it. (0, 0) when there is none or the table lacks a shape.
        Few rows stay with cuBLAS: there the table's buckets are coarse, and d1-2B's
        last-layer windows (1-21 rows) measured slower through the kernel."""
        if not KERNEL or not self.ms:
            return 0, 0
        shapes = [s for s in shapes if s[0] >= WIDE]
        keys = [f"{k}x{n}" for n, k in shapes]
        if not keys or any(key not in self.triton_us for key in keys):
            return 0, 0
        ratio = [sum(self.triton_us[key][i] for key in keys)
                 / sum(self._alt_us(key, i, m, n) for key, (n, _) in zip(keys, shapes))
                 for i, m in enumerate(self.ms)]
        best, run = (0, 0), None
        for i, m in enumerate(self.ms):
            if m <= MIN_ROWS or ratio[i] > 1.0:
                run = None
                continue
            run = [run[0] if run else (self.ms[i - 1] if i else 0), m, (run[2] if run else False) or ratio[i] <= MARGIN]
            if run[2] and run[1] - run[0] > best[1] - best[0]:
                best = (run[0], run[1])
        return best

    def cutlass_ready(self) -> bool:
        if self._cutlass is None:
            self._cutlass = {}
            if CUTLASS and self.gemm:
                try:
                    from . import cutlass_fp8

                    self._cutlass = {name: i for i, name in enumerate(cutlass_fp8.configs())}
                except Exception as e:  # no nvcc, compiler or headers: cuBLAS only
                    import warnings

                    warnings.warn(f"CUTLASS fp8 GEMM unavailable, using cuBLAS only: {e}")
        return bool(self._cutlass)

    def pick_gemm(self, m: int, n: int, k: int) -> int | None:
        """The CUTLASS config index for `gemm32`, or None for cuBLAS."""
        row = self.gemm.get(f"{k}x{n}")
        if row is None or not self.cutlass_ready():
            return None
        name = row[min(bisect.bisect_left(self.gemm_ms, m), len(self.gemm_ms) - 1)]
        return self._cutlass.get(name)

    def pick(self, m: int, n: int, k: int) -> dict:
        row = self.shapes.get(f"{k}x{n}")
        if row is None:
            return default_config(m, n, k)
        i = min(bisect.bisect_left(self.ms, m), len(self.ms) - 1)
        return row[i]


TABLE = Table()


@torch.library.custom_op("d1fast::linear8", mutates_args=())
def linear8(x8: Tensor, sx: Tensor, w8: Tensor, sw: Tensor) -> Tensor:
    """`mm` with the table's tiles, as one op opaque to torch.compile (which fuses the
    activations' quantization before it)."""
    m, k = x8.shape
    return mm(x8, sx, w8, sw, TABLE.pick(m, w8.shape[0], k))


@linear8.register_fake
def _(x8, sx, w8, sw):
    return x8.new_empty((x8.shape[0], w8.shape[0]), dtype=torch.bfloat16)


_ONE: dict = {}


@torch.library.custom_op("d1fast::gemm32", mutates_args=())
def gemm32(x8: Tensor, w8: Tensor) -> Tensor:
    """float32 `x8 @ w8.T` of e4m3 rows and weight, unscaled: CUTLASS where the table
    picked it, else cuBLAS. Opaque to torch.compile, which fuses the scales after it."""
    m, k = x8.shape
    cfg = TABLE.pick_gemm(m, w8.shape[0], k)
    if cfg is not None:
        from . import cutlass_fp8

        return cutlass_fp8.mm(x8, w8, cfg)
    one = _ONE.get(x8.device)
    if one is None:
        one = _ONE[x8.device] = torch.ones((), device=x8.device)
    return torch._scaled_mm(x8, w8.T, scale_a=one, scale_b=one, out_dtype=torch.float32)


@gemm32.register_fake
def _(x8, w8):
    return x8.new_empty((x8.shape[0], w8.shape[0]), dtype=torch.float32)


def quantize_rows(x: Tensor) -> tuple[Tensor, Tensor]:
    """(M, K) -> e4m3 rows each scaled to the format's range, and the scales (M,)."""
    s = x.abs().amax(-1).float().clamp(min=1e-12) / F8_MAX
    return (x.float() / s[:, None]).to(F8), s

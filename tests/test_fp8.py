"""The Ada fp8 kernel (d1_fast_cuda.fp8) against a float32 matmul of the same e4m3 values."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from d1_fast_cuda import fp8  # noqa: E402

if not fp8.supported():
    pytest.skip("needs a GPU of compute capability 8.9 or newer", allow_module_level=True)


@pytest.mark.parametrize("m", [1, 7, 100, 192, 333])
@pytest.mark.parametrize("k,n", [(2048, 3088), (7680, 2048), (2048, 300)])
def test_kernel_matches_dequantized_matmul(m, k, n):
    g = torch.Generator(device="cuda").manual_seed(m * 31 + n)
    x = torch.randn(m, k, device="cuda", generator=g, dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", generator=g, dtype=torch.bfloat16) * 0.02
    x8, sx = fp8.quantize_rows(x)
    w8, sw = fp8.quantize_rows(w)
    ref = (x8.float() * sx[:, None]) @ (w8.float() * sw[:, None]).T
    for cfg in (None, fp8.default_config(m, n, k) | {"SPLIT": 2}):
        y = fp8.mm(x8, sx, w8, sw, cfg).float()
        err = ((y - ref).abs().max() / ref.abs().max()).item()
        assert err < 1e-2, (cfg, err)  # the bfloat16 output rounding
        mean = ((y - ref).abs().mean() / ref.abs().mean()).item()
        assert mean < 2e-3, (cfg, mean)  # float32 accumulation: bfloat16's rounding (1.4e-3), no more
    assert torch.equal(fp8.linear8(x8, sx, w8, sw), fp8.mm(x8, sx, w8, sw, fp8.TABLE.pick(m, n, k)))


@pytest.mark.parametrize("m", [1, 100, 128, 333, 1500])
@pytest.mark.parametrize("k,n", [(10752, 2048), (2048, 3088)])
def test_gemm32_matches_float64(m, k, n):
    """`gemm32` (CUTLASS where the table picks it, else cuBLAS) against float64, and every
    compiled CUTLASS configuration on one shape."""
    g = torch.Generator(device="cuda").manual_seed(m + n)
    x8 = (torch.randn(m, k, device="cuda", generator=g) * 0.5).to(fp8.F8)
    w8 = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(fp8.F8)
    ref = x8.double() @ w8.double().T
    err = ((fp8.gemm32(x8, w8).double() - ref).abs().mean() / ref.abs().mean()).item()
    assert err < 3e-4, err  # cuBLAS measures 1.3e-4: the fp8 MMA rounds between its float32 promotions
    if m == 100 and n == 3088 and fp8.TABLE.cutlass_ready():
        from d1_fast_cuda import cutlass_fp8

        for i, name in enumerate(cutlass_fp8.configs()):
            e = ((cutlass_fp8.mm(x8, w8, i).double() - ref).abs().mean() / ref.abs().mean()).item()
            assert e < 3e-4, (name, e)

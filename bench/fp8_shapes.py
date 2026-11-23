"""FP8 (e4m3) matmul through torch._scaled_mm against bf16 cuBLAS, d1's shapes."""
import torch

SHAPES = [(2048, 5136), (2048, 2048), (2048, 6144), (2048, 15360), (7680, 2048), (2048, 3072), (2048, 21504), (10752, 2048)]
F8 = torch.float8_e4m3fn


def timeit(fn, reps=50):
    for _ in range(5):
        fn(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for i in range(reps):
        fn(i)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / reps * 1000


for m in (128,):
    for k, n in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
        w8 = [w.to(F8) for w in ws]
        sw = torch.ones(1, n, device="cuda")
        one = torch.ones((), device="cuda")
        t_bf = timeit(lambda i: x @ ws[i % 4].T)
        x8 = x.to(F8)
        sx = torch.ones(m, 1, device="cuda")
        try:
            t_t = timeit(lambda i: torch._scaled_mm(x8, w8[i % 4].T, scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
        except Exception as e:
            t_t = float("nan"); print("tensorwise", str(e)[:100])
        try:
            t_r = timeit(lambda i: torch._scaled_mm(x8, w8[i % 4].T, scale_a=sx, scale_b=sw, out_dtype=torch.bfloat16))
        except Exception as e:
            t_r = float("nan"); print("rowwise", str(e)[:100])
        # include dynamic activation quantization
        def full(i):
            s = x.abs().amax(-1, keepdim=True).float() / 448
            q = (x / s).to(F8)
            return torch._scaled_mm(q, w8[i % 4].T, scale_a=s, scale_b=sw, out_dtype=torch.bfloat16)
        try:
            t_f = timeit(full)
        except Exception as e:
            t_f = float("nan")
        print(f"M={m} ({k},{n}): bf16 {t_bf:6.1f}us  fp8 tensorwise {t_t:6.1f}  rowwise {t_r:6.1f}  rowwise+quant {t_f:6.1f}")

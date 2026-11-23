# d1-fast-cuda

Fast CUDA inference for [LiquidAI d1-3B](https://huggingface.co/LiquidAI/d1-3B) and
[d1-2B](https://huggingface.co/LiquidAI/d1-2B), on Windows and Linux.
The CUDA sibling of d1-fast-mlx. [日本語版 README](README.ja.md)

d1 models answer typed questions (`noul`, `choice`, `score`, and on d1-2B span reads
of `text`, `number`, `date`) about a state in one forward pass: nothing is decoded.
This package loads the checkpoints' own `trust_remote_code` model and replaces only
its language model's forward pass and option readout, so the API, prompts, image
processing and answers stay the original's.

```python
import d1_fast_cuda

model = d1_fast_cuda.load("LiquidAI/d1-3B")       # or "LiquidAI/d1-2B"; fp8=True for the fast mode
model.compile()                                   # optional: fused kernels (about a minute, once)
model.system_one(
    "I was charged twice and the app crashes when I open settings.",
    {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {"type": "choice", "instructions": "Which team should handle this?",
                 "criteria": {"billing": "Payments.", "engineering": "Bugs and crashes."}},
    },
)
# {"answers": {"refund": {"type": "noul", "noul": 0.97}, "team": {...}}, "usage": {...}}
```

## Why

- **It runs on Windows.** The original code calls PyTorch's flash-attention ops
  directly, and Windows builds of PyTorch have none (`USE_FLASH_ATTENTION was not
  enabled for build`). This package attends with masked SDPA only.
- **It is faster.** Single requests replay as CUDA graphs, the last layers compute only
  the rows the readout needs, the option logits read only the options' embedding rows,
  and a packed tree of questions attends by parts. An opt-in fp8 mode (Ada, RTX 40xx,
  or newer) halves the bytes each pass reads, with a Triton fp8 matmul for the wide
  projections and a CUTLASS Stream-K one for the narrow MLP down projections, where
  each beats cuBLAS.

## Speed and accuracy

RTX 4070 Ti, warm calls, median, best of two sweeps. "Original" is the checkpoints'
own code at its faster setting per row (with or without `compile`, as each model
card measures), run in WSL2 because it cannot run on Windows; this package ran on
Windows (its WSL2 numbers are within a few percent; see the full table).

| ms (lower is better) | d1-2B original | d1-2B bf16 | d1-2B fp8 | d1-3B original | d1-3B bf16 | d1-3B fp8 |
|---|---:|---:|---:|---:|---:|---:|
| 1 question | 15.6 | 9.4 | 6.4 | 16.5 | 13.8 | 8.4 |
| 3 questions | 22.8 | 11.2 | 7.5 | 36.2 | 18.4 | 10.7 |
| long state, ~4200 tokens | 199 | 187 | 135 | 321 | 283 | 188 |
| 384px image, 1 question | 27.5 | 19.6 | 16.1 | 36.8 | 32.6 | 25.2 |
| 64 states, per second | 196 | 206 | 285 | 109 | 155 | 215 |
| GPU memory, resident (GB) | | 3.8 | 2.5 | | 6.3 | 4.0 |

bf16 and fp8 here are after `model.compile()`. Without it, single requests still run
as CUDA graphs (2B 12.4 ms, 3B 15.8 ms a question). Full table and method:
[results/benchmark.md](results/benchmark.md), [docs/design.ja.md](docs/design.ja.md).

Accuracy on compare.py's 118 questions matches the original in every setting. Against
the original code in float32, this engine in bfloat16 differs by at most 0.0069 (d1-2B)
and 0.0118 (d1-3B) on any probability, while the original code in bfloat16 on CUDA
differs by 0.0094 and 0.0351; in float32 it matches to 5e-5. fp8 differs by up to
0.035 / 0.044 with the same labels and accuracy. Design notes and measurements
(Japanese): [docs/design.ja.md](docs/design.ja.md).

## Install

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/Scripts/python.exe -e ".[dev,compile]"
```

`compile` adds `triton-windows` on Windows, which `model.compile()` and the fp8 mode
need. fp8 needs an Ada (RTX 40xx) or newer GPU.

Optional, for fp8: the CUTLASS matmul is built with nvcc on first use (Windows: Visual
Studio 2022's MSVC, found by itself) and cached in `~/.cache/d1_fast_cuda`. It needs the
CUTLASS headers; without them fp8 uses cuBLAS alone, with a warning.

```powershell
git clone --depth 1 --branch v4.7.1 https://github.com/NVIDIA/cutlass third_party/cutlass
```

## Tests

The tests compare against answers of the original code in float32 on the CPU
(`results/runs/*-torch-cpu.json`, included). To make them again:

```powershell
.venv/Scripts/python.exe bench/reference_torch.py LiquidAI/d1-2B --device cpu
.venv/Scripts/python.exe -m pytest tests -q
```

## License

This package runs the d1 checkpoints' code and follows parts of it closely; it is
under the LFM Open License v1.0. See [NOTICE](NOTICE) and [LICENSES/](LICENSES).
Model weights and the checkpoints' code are downloaded from the Hugging Face Hub
under the LFM Open License v1.0.

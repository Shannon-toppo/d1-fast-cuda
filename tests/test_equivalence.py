"""The fast CUDA engine against the original PyTorch code.

The reference is the original code in float32 on the CPU
(`bench/reference_torch.py MODEL --device cpu`, results/runs/<model>-torch-cpu.json).

- float32 runs check the implementation: within `FP32_TOLERANCE` of the reference.
- bfloat16 runs (the default) stay within `TOLERANCE` and give the same labels.
- The 118 accuracy questions score as the reference does.

    .venv/Scripts/python -m pytest tests -q            # needs a CUDA GPU and the reference files
"""
import gc
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import answers  # noqa: E402
import cases  # noqa: E402

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA GPU", allow_module_level=True)

# The original code's own bfloat16 run on CUDA differs from its float32 run by up to
# 0.009 (d1-2B) and 0.035 (d1-3B); this engine measures 0.007 and 0.012
# (docs/design.ja.md), and in float32 stays within 1e-4.
TOLERANCE = float(os.environ.get("D1_TOLERANCE", "0.015"))
FP32_TOLERANCE = 0.002
# A score's expected level (d1-3B) moves by up to (levels - 1) x the probability
# difference; a read number or date must not move at all.
OTHER_TOLERANCE = 0.1
MODELS = ["d1-2B", "d1-3B"]


def reference(name):
    path = answers.RESULTS / "runs" / f"{name}-torch-cpu.json"
    if not path.exists():
        pytest.skip(f"no reference {path.name}: run bench/reference_torch.py LiquidAI/{name} --device cpu")
    return answers.load(f"{name}-torch-cpu")


def run(name, dtype):
    import d1_fast_cuda

    model = d1_fast_cuda.load(f"LiquidAI/{name}", dtype=dtype)
    media_kw = "media" if name.endswith("2B") else "images"
    with torch.inference_mode():
        out = answers.run_cases(model.system_one, cases.all_cases(name), media_kw)
    free(model)
    return out


def free(model):
    """The model and its engine refer to each other: collect them before the next load
    (d1-3B in float32 needs nearly all of a 12 GB card)."""
    model.d1_fast.graphs.cache.clear()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def accuracy(responses):
    ok = total = 0
    for cid, _, qs, _, gold in cases.accuracy_cases():
        for k in qs:
            total += 1
            ok += cases.pred(responses[cid]["answers"][k]) == gold[k]
    return ok, total


_runs: dict = {}


def run_apart(name, label, *flags):
    """`bench/smoke.py` in a process of its own (d1-3B in float32 takes about 12.4 GB,
    more than a 12 GB card has left once this process has used the GPU)."""
    import subprocess

    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "bench" / "smoke.py"), name, label, *flags], check=True,
                   cwd=root, stdout=subprocess.DEVNULL)
    return answers.load(f"{name}-{label}")


def cached_run(name, dtype="bf16"):
    if (name, dtype) not in _runs:
        _runs.clear()
        gc.collect()  # one model in GPU memory at a time
        if dtype == "fp32":
            _runs[name, dtype] = run_apart(name, "fast-fp32", "--fp32")
        else:
            _runs[name, dtype] = run(name, torch.bfloat16)
    return _runs[name, dtype]


@pytest.mark.parametrize("name", MODELS)
def test_fp32_is_exact(name):
    ref = reference(name)
    r = answers.compare(ref, cached_run(name, "fp32"))
    assert r["label_mismatches"] == [], r["label_mismatches"][:5]
    assert r["max_prob_diff"] <= FP32_TOLERANCE, r["worst_prob"][:3]


@pytest.mark.parametrize("name", MODELS)
def test_bf16_matches_reference(name):
    ref = reference(name)
    out = cached_run(name)
    r = answers.compare(ref, out)
    assert r["label_mismatches"] == [], r["label_mismatches"][:5]
    assert r["max_prob_diff"] <= TOLERANCE, r["worst_prob"][:3]
    assert r["max_other_diff"] <= OTHER_TOLERANCE, r["worst_other"][:3]
    assert accuracy(out) == accuracy(ref)


@pytest.mark.parametrize("name", MODELS)
def test_api_shape(name):
    """The answers keep the original checkpoint's API: types, keys, usage."""
    out = cached_run(name)
    first = out["tree3:text"]
    assert set(first) == {"answers", "usage"}
    assert first["usage"]["output_tokens"] == 0 and first["usage"]["input_tokens"] > 0
    a = first["answers"]
    assert a["team"]["type"] == "choice" and a["team"]["choice"] in cases.TEAM["criteria"]
    assert a["refund"]["type"] == "noul" and 0 <= a["refund"]["noul"] <= 1
    assert a["urgency"]["type"] == "score"


@pytest.mark.parametrize("name", MODELS)
def test_batch_matches_single(name):
    """system_one_batch (padded batches / packed trees, no CUDA graphs) answers as
    system_one does, one request at a time (bucketed CUDA graphs)."""
    import d1_fast_cuda

    model = d1_fast_cuda.load(f"LiquidAI/{name}")
    reqs = [(s, {"team": cases.TEAM}) for s, _, _ in cases.TEXT_CASES[:12]]
    with torch.inference_mode():
        batch = model.system_one_batch(reqs)
        single = [model.system_one(*r) for r in reqs]
    r = answers.compare(dict(enumerate(single)), dict(enumerate(batch)))
    free(model)
    assert r["label_mismatches"] == []
    assert r["max_prob_diff"] <= TOLERANCE, r["worst_prob"][:3]

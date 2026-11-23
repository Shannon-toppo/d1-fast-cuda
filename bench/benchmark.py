"""Speed and accuracy of one engine on one checkpoint (CUDA).

    .venv/Scripts/python bench/benchmark.py LiquidAI/d1-2B torch            # original code, eager
    .venv/Scripts/python bench/benchmark.py LiquidAI/d1-2B torch --compile  # original code, as its model card measures
    .venv/Scripts/python bench/benchmark.py LiquidAI/d1-2B fast [--compile] [--fp8]   # d1_fast_cuda

`--compile` follows each model card's speed section: d1-2B calls `model.compile()`,
d1-3B calls `model.compile(mode="reduce-overhead")` (CUDA graphs for single
questions). Timings are warm calls, median of 20 (the cards' method), one request
at a time. Writes results/bench/<model>-<engine>.json; bench/report.py makes the table.
"""
import json
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cases  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results" / os.environ.get("D1_BENCH_DIR", "bench")


def arg(flag, default=None, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


class Engine:
    def __init__(self, model_id, kind, compile_=False, extra=None):
        import torch

        self.torch = torch
        self.name = model_id.split("/")[-1]
        self.is_2b = self.name.endswith("2B")
        self.media_kw = "media" if self.is_2b else "images"
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if kind == "torch":
            from transformers import AutoModel

            m = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16).to("cuda")
            if compile_:
                m.compile() if self.is_2b else m.compile(mode="reduce-overhead")
        else:
            import d1_fast_cuda

            m = d1_fast_cuda.load(model_id, **(extra or {}))
            if compile_:
                m.compile()
        self.model = m
        self._one, self._batch = m.system_one, m.system_one_batch
        self.load_s = time.time() - t0

    def one(self, state, qs, img=None):
        with self.torch.inference_mode():
            return self._one(state, qs, **({self.media_kw: [img]} if img is not None else {}))

    def batch(self, reqs):
        with self.torch.inference_mode():
            return self._batch(reqs)

    def sync(self):
        self.torch.cuda.synchronize()

    def memory_gb(self):
        return self.torch.cuda.max_memory_allocated() / 1e9


def main():
    model_id, kind = sys.argv[1], sys.argv[2]
    compile_ = "--compile" in sys.argv
    fp8 = "--fp8" in sys.argv
    label = kind + ("-fp8" if fp8 else "") + ("-compile" if compile_ else "") + (arg("--label", "") and "-" + arg("--label"))
    e = Engine(model_id, kind, compile_, {"fp8": True} if fp8 else None)

    def timed(fn, n=20, warm=5):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(n):
            e.sync()
            t = time.perf_counter()
            fn()
            e.sync()
            ts.append((time.perf_counter() - t) * 1000)
        return statistics.median(ts)

    # accuracy first (also warms every path up)
    t_acc = time.time()
    ok = total = 0
    wrong = []
    for cid, state, qs, img, gold in cases.accuracy_cases():
        a = e.one(state, qs, img)["answers"]
        for k in qs:
            total += 1
            good = cases.pred(a[k]) == gold[k]
            ok += good
            if not good:
                wrong.append(f"{cid}.{k}")
    t_acc = time.time() - t_acc

    long_state = " ".join(s for s, _, _ in cases.TEXT_CASES) * 6
    img = cases.image_cases()[0][0]
    batch = [(s, {"team": cases.TEAM}) for s, _, _ in cases.TEXT_CASES[:16]] * 4
    n_long = e.one(long_state, {"team": cases.TEAM})["usage"]["input_tokens"]
    speed = {
        "1q": timed(lambda: e.one(cases.TEXT_CASES[0][0], {"team": cases.TEAM})),
        "3q": timed(lambda: e.one(cases.TEXT_CASES[0][0], cases.THREE)),
        "long": timed(lambda: e.one(long_state, {"team": cases.TEAM}), n=10, warm=3),
        "image": timed(lambda: e.one(None, {"count": cases.COUNT_Q}, img)),
    }
    bt = timed(lambda: e.batch(batch), n=5, warm=2)
    speed["batch64_per_s"] = 64 / (bt / 1000)
    out = {"model": e.name, "engine": label, "gpu": e.torch.cuda.get_device_name(0),
           "load_s": e.load_s, "accuracy_pass_s": t_acc, "memory_gb": e.memory_gb(),
           "long_tokens": n_long, "speed_ms": speed, "accuracy": [ok, total], "wrong": wrong}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{e.name}-{label}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()

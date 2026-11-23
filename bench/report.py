"""One Markdown table per checkpoint from results/bench/*.json.

    .venv/Scripts/python bench/report.py > results/benchmark.md
"""
import json
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1] / "results" / "bench"
ORDER = ["torch", "torch-compile", "fast", "fast-compile", "fast-fp8-compile", "fast-compile-wsl"]
NAMES = {"torch": "元実装 (WSL)", "torch-compile": "元実装 +compile (WSL)", "fast": "d1-fast-cuda",
         "fast-compile": "d1-fast-cuda +compile", "fast-fp8-compile": "d1-fast-cuda fp8 +compile",
         "fast-compile-wsl": "d1-fast-cuda +compile (WSL)"}
ROWS = [("1q", "1問 (ms)"), ("3q", "3問同時 (ms)"), ("long", "長文 (ms)"), ("image", "画像384px 1問 (ms)"),
        ("batch64_per_s", "バッチ64件 (件/秒)")]


def best(group: list[dict]) -> dict:
    """Several sweeps of one setting: each time its fastest, the throughput its highest."""
    out = dict(group[0])
    out["speed_ms"] = {k: (max if k.endswith("_per_s") else min)(g["speed_ms"][k] for g in group)
                       for k in group[0]["speed_ms"]}
    out["memory_gb"] = max(g["memory_gb"] for g in group)
    return out


def main():
    found: dict = {}
    for d in [BENCH, *sorted(BENCH.parent.glob("bench-*"))]:  # results/bench, results/bench-2, ...
        for p in sorted(d.glob("*.json")):
            r = json.loads(p.read_text())
            found.setdefault((r["model"], r["engine"]), []).append(r)
    runs = [best(g) for g in found.values()]
    sweeps = max(len(g) for g in found.values())
    for model in sorted({r["model"] for r in runs}):
        rs = {r["engine"]: r for r in runs if r["model"] == model}
        cols = [e for e in ORDER if e in rs]
        print(f"### {model}\n")
        print("| 項目 | " + " | ".join(NAMES[c] for c in cols) + " |")
        print("|---|" + "---:|" * len(cols))
        originals = [rs[e] for e in ("torch", "torch-compile") if e in rs]
        for key, label in ROWS:
            cells = []
            for c in cols:
                v = rs[c]["speed_ms"][key]
                cell = f"{v:.1f}"
                if originals and c.startswith("fast"):  # against the original's faster setting on this row
                    pick = max if key == "batch64_per_s" else min
                    b = pick(o["speed_ms"][key] for o in originals)
                    ratio = v / b if key == "batch64_per_s" else b / v
                    cell += f" ({ratio:.2f}x)"
                cells.append(cell)
            if key == "long":
                label = f"長文 約{rs[cols[0]]['long_tokens']}トークン (ms)"
            print(f"| {label} | " + " | ".join(cells) + " |")
        print("| 正解率 (118問) | " + " | ".join(f"{rs[c]['accuracy'][0]}/{rs[c]['accuracy'][1]}" for c in cols) + " |")
        print("| GPU メモリ ピーク (GB) | " + " | ".join(f"{rs[c]['memory_gb']:.1f}" for c in cols) + " |")
        print()
    print(f"RTX 4070 Ti、bf16（fp8 列を除く）、ウォーム呼び出しの中央値（20回、長文10回）を{sweeps}回の通し計測で比べ、"
          "各設定の最速を載せた。括弧内は、その行で速いほうの元実装（compileあり・なし）に対する速さ（大きいほど速い）。")
    print("元実装は Windows 版 PyTorch では動かない（flash attention なし）ため WSL2 で測った。d1-fast-cuda は Windows で測った。")


if __name__ == "__main__":
    main()

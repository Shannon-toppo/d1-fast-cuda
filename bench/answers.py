"""Run every case through an engine and compare two runs. Torch-free and MLX-free."""
import json
import math
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run_cases(system_one, cases, media_kw):
    """{case id: response} for `system_one(state, questions, **{media_kw: [image]})`."""
    out = {}
    for cid, state, qs, img in cases:
        kw = {media_kw: [img]} if img is not None else {}
        out[cid] = system_one(state, qs, **kw)
    return out


def save(name, data):
    path = RESULTS / "runs" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str))
    return path


def load(name):
    return json.loads((RESULTS / "runs" / f"{name}.json").read_text())


def _walk(a, b, path, diffs, labels):
    """Pair the numbers and labels of two answers by their path."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            labels.append((path, a, b))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diffs.append((abs(float(a) - float(b)), path, a, b))
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                labels.append((f"{path}.{k}", a.get(k, "<missing>"), b.get(k, "<missing>")))
            else:
                _walk(a[k], b[k], f"{path}.{k}", diffs, labels)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            labels.append((path, f"len {len(a)}", f"len {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", diffs, labels)
        return
    labels.append((path, repr(a), repr(b)))


# Numeric fields that are probabilities (bounded by 1); others (a score's expected
# level, a read number) are reported apart.
PROB_KEYS = ("probabilities", "confidence", "noul", "presence", "found")


def compare(ref, run, ignore=("usage",)):
    """Largest probability difference, other numeric differences, and label mismatches."""
    probs, others, labels = [], [], []
    for cid in ref:
        if cid not in run:
            labels.append((cid, "present", "missing"))
            continue
        diffs, lab = [], []
        _walk({k: v for k, v in ref[cid].items() if k not in ignore},
              {k: v for k, v in run[cid].items() if k not in ignore}, cid, diffs, lab)
        for d in diffs:
            (probs if any(f".{k}" in d[1] for k in PROB_KEYS) else others).append(d)
        labels += lab
    probs.sort(reverse=True)
    others.sort(reverse=True)
    return {"max_prob_diff": probs[0][0] if probs else 0.0, "worst_prob": probs[:5],
            "max_other_diff": others[0][0] if others else 0.0, "worst_other": others[:5],
            "label_mismatches": labels, "n_probs": len(probs)}


def report(name_ref, name_run):
    r = compare(load(name_ref), load(name_run))
    print(f"{name_run} vs {name_ref}: max prob diff {r['max_prob_diff']:.4f} over {r['n_probs']} values, "
          f"max other diff {r['max_other_diff']:.4f}, label mismatches {len(r['label_mismatches'])}")
    for d in r["worst_prob"][:3]:
        print(f"   prob {d[0]:.4f} at {d[1]}")
    for d in r["worst_other"][:3]:
        print(f"   other {d[0]:.4f} at {d[1]} ({d[2]} vs {d[3]})")
    for m in r["label_mismatches"][:10]:
        print("   label", m)
    return r


if __name__ == "__main__":
    import sys

    report(sys.argv[1], sys.argv[2])

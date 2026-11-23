"""Distribution of probability differences of runs against a reference.

    python bench/diffstats.py d1-3B-torch-cpu d1-3B-fast d1-3B-torch-mps ...
"""
import sys

import answers


def main():
    ref = answers.load(sys.argv[1])
    for name in sys.argv[2:]:
        run = answers.load(name)
        r = answers.compare(ref, run)
        diffs = []
        for cid in ref:
            c = answers.compare({cid: ref[cid]}, {cid: run[cid]})
            diffs.append((c["max_prob_diff"], cid))
        diffs.sort(reverse=True)
        values = [d for d, _ in diffs]
        n = len(values)
        print(f"{name:28s} max {values[0]:.4f}  p95 {values[int(n * 0.05)]:.4f}  median {values[n // 2]:.4f}  "
              f">0.01: {sum(v > 0.01 for v in values)}/{n}  worst: {', '.join(c for _, c in diffs[:3])}")


if __name__ == "__main__":
    main()

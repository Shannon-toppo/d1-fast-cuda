#!/usr/bin/env bash
# Every setting in its own process, one after another (bench/report.py makes the table).
# The original code runs in WSL2 (Windows PyTorch has no flash kernels); this package on Windows.
#   bash bench/run_all.sh            # writes results/bench
#   bash bench/run_all.sh bench-2    # a second sweep; report.py takes each setting's best
set -e
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 D1_BENCH_DIR=${1:-bench} WSLENV=D1_BENCH_DIR
PY=.venv/Scripts/python.exe
for m in LiquidAI/d1-2B LiquidAI/d1-3B; do
  wsl -d Ubuntu -- bash bench/wsl.sh bench/benchmark.py $m torch
  wsl -d Ubuntu -- bash bench/wsl.sh bench/benchmark.py $m torch --compile
  $PY bench/benchmark.py $m fast
  $PY bench/benchmark.py $m fast --compile
  $PY bench/benchmark.py $m fast --compile --fp8
  wsl -d Ubuntu -- bash bench/wsl.sh bench/benchmark.py $m fast --compile --label wsl
done
$PY bench/report.py > results/benchmark.md

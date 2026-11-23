#!/usr/bin/env bash
# Run a Python command in the WSL2 Ubuntu env (Linux PyTorch has the flash kernels the
# original d1 code calls; Windows builds do not). Weights come from the Windows HF cache.
#   wsl -d Ubuntu -- bash bench/wsl.sh bench/benchmark.py LiquidAI/d1-2B torch --compile
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=/mnt/c/Users/user/.cache/huggingface/hub HF_HOME=$HOME/.hf HF_HUB_OFFLINE=1
exec "$HOME/d1env/.venv/bin/python" "$@"

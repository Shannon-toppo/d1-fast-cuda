"""Answers of the original PyTorch code (transformers, trust_remote_code) for every case.

Run where the original code runs (WSL: bench/wsl.sh, Linux PyTorch has its flash kernels):
    wsl -d Ubuntu -- bash bench/wsl.sh bench/reference_torch.py LiquidAI/d1-2B --device cuda|cpu
Writes results/runs/<model>-torch-<device>.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import answers  # noqa: E402
import cases  # noqa: E402


def main():
    import torch
    from transformers import AutoModel

    model_id = sys.argv[1]
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    if "--bf16" in sys.argv:
        dtype = torch.bfloat16
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=dtype).to(device)
    name = model_id.split("/")[-1]
    media_kw = "media" if name.endswith("2B") else "images"
    with torch.inference_mode():
        out = answers.run_cases(model.system_one, cases.all_cases(name), media_kw)
    suffix = device + ("-bf16" if device == "cpu" and dtype == torch.bfloat16 else "")
    print(answers.save(f"{name}-torch-{suffix}", out))


if __name__ == "__main__":
    main()

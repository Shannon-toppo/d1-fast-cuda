"""`load(repo)`: the checkpoint as its own `trust_remote_code` model, with this
package's forward installed under its engine (`engine.py`)."""

from __future__ import annotations

import torch


def load(path_or_repo: str, device: str = "cuda", dtype=torch.bfloat16, revision: str | None = None,
         fp8: bool = False):
    """The original model (`AutoModel.from_pretrained(..., trust_remote_code=True)`) on
    `device`, its `system_one` / `system_one_batch` answered through this package.
    `model.d1_fast` is the installed engine (`Fast2B` or `Fast3B`). `fp8=True` runs the
    language model's wide projections in float8 (`HybridLM.to_fp8`): faster, at some
    cost in precision (docs/design.ja.md)."""
    from transformers import AutoModel

    from .engine import Fast2B, Fast3B

    model = AutoModel.from_pretrained(path_or_repo, trust_remote_code=True, dtype=dtype, revision=revision)
    model.to(device).eval()
    kind = model.config.model_type
    if kind == "lfm3_vl":
        fast = Fast2B(model)
    elif kind == "lfm2_vl":
        fast = Fast3B(model)
    else:
        raise ValueError(f"{path_or_repo} is not a d1 checkpoint this package serves ({kind})")
    if fp8:
        fast.lm.to_fp8()
        _drop_layer_weights(fast.lang)
    object.__setattr__(model, "d1_fast", fast)
    model.compile = fast.compile  # the original API's `model.compile()`: here the fused forward
    return model


def _drop_layer_weights(lang) -> None:
    """Free the original decoder layers' parameters once the fast stack holds what it
    needs (it keeps its own references to the bfloat16 tensors it still reads), so the
    float8 copies do not sit beside the bfloat16 ones. The original language model's
    own forward cannot run afterwards; `system_one` never calls it."""
    for module in lang.layers.modules():
        for name, p in list(module._parameters.items()):
            if p is not None:
                module._parameters[name] = torch.nn.Parameter(p.new_empty(0), requires_grad=False)
    torch.cuda.empty_cache()

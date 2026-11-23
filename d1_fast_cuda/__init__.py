"""Fast CUDA inference for LiquidAI d1-3B and d1-2B, on the original checkpoints' own code.

    import d1_fast_cuda
    model = d1_fast_cuda.load("LiquidAI/d1-3B")        # or "LiquidAI/d1-2B"
    model.system_one(state, questions)                # the original API, unchanged
"""

from .loader import load

__all__ = ["load"]

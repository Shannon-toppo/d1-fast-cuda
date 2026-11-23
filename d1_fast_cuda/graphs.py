"""Single requests as CUDA graphs.

A d1 question is about 100 tokens; at that size the GPU finishes each kernel of an
eager forward faster than the host launches the next one (Windows' WDDM driver makes
launches dearer still). A captured graph replays the whole forward with one launch.

Shapes are bucketed so a few graphs serve every prompt: the length rounds up to a
multiple of `STEP` tokens and the number of needed rows up to a multiple of 4. The
padding rows sit after the real ones: causal operators never let them reach a real
row, and the attention masks keep them out of the real rows' keys (a padding row
attends to itself only). Graphs share one memory pool; outputs are cloned before
the next replay can overwrite them.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable

import torch

STEP = int(os.environ.get("D1_GRAPH_STEP", "32"))
MAX_TOKENS = int(os.environ.get("D1_GRAPH_MAX_TOKENS", "1024"))  # longer prompts run eagerly
MAX_GRAPHS = int(os.environ.get("D1_GRAPH_MAX", "96"))
ENABLED = os.environ.get("D1_GRAPHS", "1") != "0"


def bucket(n: int, step: int = STEP) -> int:
    return -(-n // step) * step


class Graphs:
    def __init__(self):
        self.pool = None
        self.cache: OrderedDict = OrderedDict()

    def run(self, key, inputs: dict, forward: Callable[[dict], torch.Tensor]) -> torch.Tensor:
        """`forward(static)` replayed as a graph keyed by `key`; `inputs` (tensors of
        fixed shapes for that key, or None) are copied into the graph's buffers."""
        entry = self.cache.get(key)
        if entry is None:
            if self.pool is None:
                self.pool = torch.cuda.graph_pool_handle()
            static = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up outside the capture (cuBLAS workspaces, kernel selection)
                for _ in range(2):
                    forward(static)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                out = forward(static)
            entry = (graph, static, out)
            self.cache[key] = entry
            while len(self.cache) > MAX_GRAPHS:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
            graph, static, out = entry
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    static[k].copy_(v)
        graph, static, out = entry
        graph.replay()
        return out.clone()

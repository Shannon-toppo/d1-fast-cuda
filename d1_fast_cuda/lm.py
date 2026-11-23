"""The language model of both d1 checkpoints as plain PyTorch, over the weights of the
loaded original model (no copies but the fused projections).

d1-2B's LFM3 and d1-3B's LFM2 are the same hybrid stack: per layer an operator
(a gated short convolution, or GQA attention with q/k RMSNorm and RoPE, on d1-2B
also a sliding window, a sigmoid output gate per head and sandwich norms) and a
SwiGLU MLP. This module runs it with nothing but matmuls, `F.conv1d` and
`F.scaled_dot_product_attention` under an explicit mask, so it needs no flash
kernels (Windows builds of PyTorch have none) and runs on any CUDA GPU.

What it does differently from the original forward (docs/design.ja.md):

- `Plan.rows`: only the rows the readout needs leave the last layers. After the
  last attention layer only short convolutions follow, so each needed row takes a
  window of `c + 1` rows (c = the convolutions' context) through the last
  attention layer's queries and MLP, and the convolutions shrink the windows to the
  row itself (`_tail`). Every computed row gets exactly the inputs it gets in a
  full pass.
- The fused projections run as one matmul (q/k/v[/gate], gate/up of the MLP).
- RoPE angles stay in float32 (the original d1-2B rounds cos/sin to bfloat16).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


def rms(x: Tensor, w: Tensor, eps: float) -> Tensor:
    """Both originals' RMSNorm: normalized in float32, rounded, then scaled."""
    x32 = x.float()
    return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * w


F8 = torch.float8_e4m3fn
F8_MAX = 448.0


@dataclass
class FP8Weight:
    """An (N, K) weight in float8 e4m3, one float32 scale per output row."""

    w8: Tensor
    scale: Tensor  # (N,)
    one: Tensor  # () 1.0, the scales `_scaled_mm` takes (Ada has no row-wise scaling)

    @classmethod
    def of(cls, w: Tensor) -> "FP8Weight":
        scale = w.detach().abs().amax(dim=1).float().clamp(min=1e-12) / F8_MAX
        w8 = (w.detach().float() / scale[:, None]).to(F8)
        return cls(w8, scale, torch.ones((), device=w.device))


def linear(x: Tensor, w) -> Tensor:
    """`x @ w.T`; an `FP8Weight` multiplies in float8 with each row of `x` scaled to
    e4m3's range, both scales applied to the float32 product."""
    if isinstance(w, Tensor):
        return x @ w.T
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    s = x2.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / F8_MAX
    y = torch._scaled_mm((x2.float() / s).to(F8), w.w8.T, scale_a=w.one, scale_b=w.one, out_dtype=torch.float32)
    return (y * s * w.scale).to(x.dtype).view(*shape[:-1], -1)


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Half-split RoPE in float32 over (..., T, H, D); cos/sin (T, D) or (B, T, D)."""
    x32 = x.float()
    a, b = x32.chunk(2, dim=-1)
    return (x32 * cos[..., None, :] + torch.cat((-b, a), dim=-1) * sin[..., None, :]).to(x.dtype)


@dataclass
class Layer:
    kind: str  # "conv", "full_attention", "sliding_attention"
    norm: Tensor
    ffn_norm: Tensor
    post_norm: Tensor | None
    post_ffn_norm: Tensor | None
    w13: Tensor  # (2I, D)
    w2: Tensor  # (D, I)
    # attention
    w_in: Tensor | None = None  # (q | k | v [| gate], D)
    split: tuple[int, ...] = ()
    q_norm: Tensor | None = None
    k_norm: Tensor | None = None
    w_o: Tensor | None = None
    inv_freq: Tensor | None = None  # (D/2,) float32, zeros past the rotated pairs
    window: int | None = None
    # convolution
    conv_in: Tensor | None = None  # (3C, D)
    conv_w: Tensor | None = None  # (C, 1, K)
    conv_b: Tensor | None = None
    conv_out: Tensor | None = None

    @property
    def attention(self) -> bool:
        return self.kind != "conv"

    @property
    def taps(self) -> int:
        return self.conv_w.shape[-1]


@dataclass
class Plan:
    """What one forward reads and returns.

    `full` and `sliding`: boolean masks (B, 1, T, T) for the attention layers
    (True = may attend), or None for plain causal attention. `positions`: (T,) or
    (B, T). `rows` (B == 1 only): the flat rows to return; None returns every row.
    `conv`: an optional tree layout for the convolutions (`tree_conv_order`).
    """

    positions: Tensor
    full: Tensor | None = None
    sliding: Tensor | None = None
    rows: Tensor | None = None
    windows: Tensor | None = None  # (R, c + 1) row of each window slot, clamped to 0
    valid: Tensor | None = None  # (R, c + 1) 1 where the slot is a real row, else 0
    conv: tuple[Tensor, Tensor] | None = None  # (gather order, kept outputs) over the left-padded input
    full_all: bool = False  # full attention with no mask at all: every row sees every row (no padding)
    window: int | None = None  # with `sliding` None: causal within this window, attended block by block
    # a tree attended by parts (`HybridLM._tree_attend`), with `full` None:
    # (trunk length, branch rows padded (N, L) into the packed axis, their validity (N, L),
    #  the valid padded slots in packed order, each row's branch (T,) with -1 on the trunk)
    parts: tuple | None = None
    extras: dict = field(default_factory=dict)


class HybridLM:
    """The stack over a loaded original model's language model."""

    def __init__(self, layers: list[Layer], embed: Tensor, final_norm: Tensor, eps: float, heads: int,
                 kv_heads: int, head_dim: int, gated: bool):
        self.layers, self.embed, self.final_norm, self.eps = layers, embed, final_norm, eps
        self.heads, self.kv_heads, self.head_dim, self.gated = heads, kv_heads, head_dim, gated
        self.scale = head_dim ** -0.5
        self.last_attention = max(i for i, l in enumerate(layers) if l.attention)
        self.context = sum(l.taps - 1 for l in layers[self.last_attention + 1:])
        self.run = self.forward  # `compile()` swaps in the inductor-fused forward

    def to_fp8(self, names=("w_in", "conv_in", "w13", "w2")) -> None:
        """Opt-in: the wide projections as `FP8Weight`s (half the bytes a pass reads).
        The 2048 x 2048 output projections stay bfloat16: cuBLAS runs them faster
        than `_scaled_mm` at d1's row counts (bench/fp8_shapes.py)."""
        for layer in self.layers:
            for name in names:
                w = getattr(layer, name)
                if isinstance(w, Tensor):
                    setattr(layer, name, FP8Weight.of(w))

    def compile(self, dynamic: bool | None = None, mode: str | None = None) -> None:
        """Fuse the elementwise work (norms, RoPE, gates, the residual adds) around the
        matmuls with torch.compile. Graph buckets still replay as CUDA graphs on top."""
        self.run = torch.compile(self.forward, dynamic=dynamic, mode=mode)

    # ------------------------------------------------------------------ plans

    def windows(self, rows: list[int], prev: list[int]) -> tuple[list[list[int]], list[list[int]]]:
        """Each needed row's window: its `context` predecessors (through `prev`, the
        convolution order) then itself; -1 slots are clamped to 0 and marked invalid."""
        out, valid = [], []
        for r in rows:
            w = [r]
            for _ in range(self.context):
                w.append(prev[w[-1]] if w[-1] >= 0 else -1)
            w.reverse()
            out.append([max(i, 0) for i in w])
            valid.append([int(i >= 0) for i in w])
        return out, valid

    # ---------------------------------------------------------------- forward

    def rotary(self, layer: Layer, positions: Tensor) -> tuple[Tensor, Tensor]:
        freqs = positions.float()[..., None] * layer.inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _qkv(self, layer: Layer, x: Tensor, cos: Tensor, sin: Tensor):
        b, t, _ = x.shape
        parts = linear(x, layer.w_in).split(layer.split, dim=-1)
        q = rms(parts[0].view(b, t, self.heads, self.head_dim), layer.q_norm, self.eps)
        k = rms(parts[1].view(b, t, self.kv_heads, self.head_dim), layer.k_norm, self.eps)
        v = parts[2].view(b, t, self.kv_heads, self.head_dim)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        gate = parts[3] if self.gated else None
        return q, k, v, gate

    def _attend(self, layer: Layer, q: Tensor, k: Tensor, v: Tensor, gate: Tensor | None, mask: Tensor | None,
                causal: bool = True, window: int | None = None, parts: tuple | None = None):
        """Attention of (B, T, H, D) queries: under `mask`, else causal (within `window`
        if given), else (`causal=False`) every query over every key."""
        b, t = q.shape[:2]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if parts is not None and mask is None:
            out = self._tree_attend(q, k, v, parts)
        elif mask is None and window is not None and t > window:
            out = self._windowed(q, k, v, window)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and causal,
                                                 scale=self.scale, enable_gqa=True)
        out = out.transpose(1, 2)
        if gate is not None:
            out = out * torch.sigmoid(gate)[..., None]
        return linear(out.reshape(b, t, -1), layer.w_o)

    def _mask(self, layer: Layer, plan: Plan) -> Tensor | None:
        return plan.sliding if layer.kind == "sliding_attention" else plan.full

    def _how(self, layer: Layer, plan: Plan) -> dict:
        """`_attend`'s keywords for a layer under `plan`."""
        if layer.kind == "sliding_attention":
            return {"mask": plan.sliding, "window": plan.window}
        return {"mask": plan.full, "causal": not plan.full_all, "parts": plan.parts}

    def _tree_attend(self, q: Tensor, k: Tensor, v: Tensor, parts: tuple) -> Tensor:
        """A packed tree's attention by parts, (1, H, T, D): the trunk causally over
        itself; the branches as a padded batch, each over the trunk's keys and its own
        earlier tokens. T x (P + L) scores for the branches instead of a T x T mask."""
        p, rows, valid, keep, _ = parts
        n, length = rows.shape
        h, d = q.shape[1], q.shape[3]
        hk = k.shape[1]
        out_trunk = (F.scaled_dot_product_attention(q[:, :, :p], k[:, :, :p], v[:, :, :p], is_causal=True,
                                                    scale=self.scale, enable_gqa=True) if p else q[:, :, :0])

        def branch(x):  # (1, H, T, D) -> (N, H, L, D)
            return x[0][:, rows].permute(1, 0, 2, 3)

        qb, kb, vb = branch(q), branch(k), branch(v)
        if p:
            kb = torch.cat((k[:, :, :p].expand(n, -1, -1, -1), kb), dim=2)
            vb = torch.cat((v[:, :, :p].expand(n, -1, -1, -1), vb), dim=2)
        i = torch.arange(length, device=q.device)
        own = ((i[None, :] <= i[:, None])[None] & valid[:, None, :]) | (i[None, :] == i[:, None])[None]
        mask = torch.cat((own.new_ones(n, length, p), own), dim=2) if p else own  # (N, L, P + L)
        out = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask[:, None], scale=self.scale, enable_gqa=True)
        out_branches = out.permute(0, 2, 1, 3).reshape(n * length, h, d)[keep]  # (T - P, H, D) in packed order
        return torch.cat((out_trunk, out_branches.permute(1, 0, 2)[None]), dim=2)

    def _windowed(self, q: Tensor, k: Tensor, v: Tensor, window: int) -> Tensor:
        """Causal attention within `window` over (B, H, T, D), as blocks of `window`
        queries that each see their own block and the one before it: T x 2w scores
        instead of T x T (the original's `attention._windowed`). Right padding only
        reaches the padding rows after it."""
        b, h, t, d = q.shape
        nb = -(-t // window)
        pad = nb * window - t

        def blocks(x):
            return F.pad(x, (0, 0, 0, pad)).view(b, x.shape[1], nb, window, d)

        def with_previous(x):  # (b, h, nb, w, d) -> (b, h, nb, 2w, d)
            return torch.cat((F.pad(x, (0, 0, 0, 0, 1, 0))[:, :, :-1], x), dim=3)

        qb, kb, vb = blocks(q), with_previous(blocks(k)), with_previous(blocks(v))
        qpos = torch.arange(window, device=q.device)[:, None] + window
        kpos = torch.arange(2 * window, device=q.device)[None, :]
        allowed = (kpos <= qpos) & (kpos > qpos - window)
        first = allowed & (kpos >= window)  # block 0 has no block before it
        mask = torch.cat((first[None], allowed[None].expand(nb - 1, -1, -1)))[:, None]  # (nb, 1, w, 2w)
        mask = mask.repeat(b, 1, 1, 1)

        def fold(x):  # (b, h, nb, n, d) -> (b * nb, h, n, d)
            return x.transpose(1, 2).reshape(b * nb, x.shape[1], x.shape[3], d)

        out = F.scaled_dot_product_attention(fold(qb), fold(kb), fold(vb), attn_mask=mask, scale=self.scale,
                                             enable_gqa=True)
        return out.view(b, nb, h, window, d).transpose(1, 2).reshape(b, h, nb * window, d)[:, :, :t]

    def _conv(self, layer: Layer, x: Tensor, plan: Plan) -> Tensor:
        u = linear(x, layer.conv_in)
        bb, c, xx = u.chunk(3, dim=-1)
        bx = (bb * xx).transpose(1, 2)  # (B, C, T)
        keep = layer.taps - 1
        padded = F.pad(bx, (keep, 0))
        if plan.conv is None:
            z = F.conv1d(padded, layer.conv_w, groups=bx.shape[1])
        else:
            order, kept = plan.conv
            z = F.conv1d(padded[..., order], layer.conv_w, groups=bx.shape[1])[..., kept]
        if layer.conv_b is not None:
            z = z + layer.conv_b[:, None]
        return linear(c * z.transpose(1, 2), layer.conv_out)

    def _mlp(self, layer: Layer, x: Tensor) -> Tensor:
        g, u = linear(x, layer.w13).chunk(2, dim=-1)
        return linear(F.silu(g) * u, layer.w2)

    def _block_tail(self, layer: Layer, h: Tensor, y: Tensor) -> Tensor:
        if layer.post_norm is not None:
            y = rms(y, layer.post_norm, self.eps)
        h = h + y
        f = self._mlp(layer, rms(h, layer.ffn_norm, self.eps))
        if layer.post_ffn_norm is not None:
            f = rms(f, layer.post_ffn_norm, self.eps)
        return h + f

    def forward(self, h: Tensor, plan: Plan) -> Tensor:
        """Final-normed hidden states: (B, T, D), or (R, D) at `plan.windows`' last slots."""
        rot = {}
        stop = self.last_attention if plan.windows is not None else len(self.layers)
        for layer in self.layers[:stop]:
            x = rms(h, layer.norm, self.eps)
            if layer.attention:
                if layer.kind not in rot:
                    rot[layer.kind] = self.rotary(layer, plan.positions)
                q, k, v, gate = self._qkv(layer, x, *rot[layer.kind])
                y = self._attend(layer, q, k, v, gate, **self._how(layer, plan))
            else:
                y = self._conv(layer, x, plan)
            h = self._block_tail(layer, h, y)
        if plan.windows is None:
            return rms(h, self.final_norm, self.eps)
        return self._tail(h, plan, rot)

    def _tail(self, h: Tensor, plan: Plan, rot: dict) -> Tensor:
        """The last attention layer on the windows' rows only, then the convolutions
        over the windows (valid: each shrinks a window by its taps - 1)."""
        layer = self.layers[self.last_attention]
        r, width = plan.windows.shape
        flat = plan.windows.flatten()
        x = rms(h, layer.norm, self.eps)
        if layer.kind not in rot:
            rot[layer.kind] = self.rotary(layer, plan.positions)
        q, k, v, gate = self._qkv(layer, x, *rot[layer.kind])
        how = self._how(layer, plan)
        mask = how["mask"]
        q = q[:, flat]
        gate = gate[:, flat] if gate is not None else None
        if mask is not None:
            mask = mask[:, :, flat]
        elif how.get("parts") is not None:  # a tree: the keys up to the row, on the trunk or its own branch
            branch = how["parts"][4]
            keys = torch.arange(h.shape[1], device=h.device)[None, :]
            mask = ((keys <= flat[:, None]) & ((branch[None, :] < 0) | (branch[None, :] == branch[flat][:, None])))
            mask = mask[None, None]
        elif how.get("causal", True):  # causal (within the window): each query sees the keys up to its own row
            keys = torch.arange(h.shape[1], device=h.device)[None, :]
            mask = keys <= flat[:, None]
            if how.get("window"):
                mask = mask & (keys > flat[:, None] - how["window"])
            mask = mask[None, None]
        y = self._attend(layer, q, k, v, gate, mask, causal=mask is not None)
        hw = self._block_tail(layer, h[:, flat], y)[0].view(r, width, -1)  # (R, W, D)
        valid = plan.valid.to(hw.dtype)[..., None]
        for layer in self.layers[self.last_attention + 1:]:
            x = rms(hw, layer.norm, self.eps)
            u = linear(x, layer.conv_in)
            bb, c, xx = u.chunk(3, dim=-1)
            keep = layer.taps - 1
            bx = (bb * xx * valid).transpose(1, 2)  # (R, C, W): slots before the sequence start read as zeros
            z = F.conv1d(bx, layer.conv_w, groups=bx.shape[1])  # (R, C, W - keep)
            if layer.conv_b is not None:
                z = z + layer.conv_b[:, None]
            y = linear(c[:, keep:] * z.transpose(1, 2), layer.conv_out)
            hw = self._block_tail(layer, hw[:, keep:], y)
            valid = valid[:, keep:]
        return rms(hw[:, -1], self.final_norm, self.eps)


# ---------------------------------------------------------------------- loading


def _fused_weight(module, names: tuple[str, ...]) -> Tensor:
    """One weight for `names` (bias-free linears of `module`) stacked by rows. The
    original d1-2B fuses them itself (`module.fused`); otherwise the stack is built
    once and the linears' weights become views of it, so nothing is held twice."""
    fused = getattr(module, "fused", None)
    if fused is not None and tuple(fused.names) == names:
        return fused.weight.detach()
    linears = [getattr(module, n) for n in names]
    weight = torch.cat([lin.weight.detach() for lin in linears])
    at = 0
    for lin in linears:
        n = lin.weight.shape[0]
        lin.weight = torch.nn.Parameter(weight[at:at + n], requires_grad=False)
        at += n
    return weight


def _inv_freq(head_dim: int, theta: float, partial: float = 1.0) -> Tensor:
    rotated = int(partial * head_dim) // 2
    inv = 1.0 / (theta ** (torch.arange(0, 2 * rotated, 2, dtype=torch.float) / head_dim))
    return torch.cat((inv, torch.zeros(head_dim // 2 - rotated)))


def from_lfm3(lm, config) -> HybridLM:
    """d1-2B: transformers' (vendored) `Lfm3Model`."""
    layers = []
    hd = config.head_dim
    for i, block in enumerate(lm.layers):
        kind = config.layer_types[i]
        sandwich = getattr(block, "use_sandwich_norm", False)
        common = dict(kind=kind, norm=block.operator_norm.weight, ffn_norm=block.ffn_norm.weight,
                      post_norm=block.post_operator_norm.weight if sandwich else None,
                      post_ffn_norm=block.post_ffn_norm.weight if sandwich else None,
                      w13=_fused_weight(block.feed_forward, ("w1", "w3")),
                      w2=block.feed_forward.w2.weight)
        if kind == "conv":
            conv = block.conv
            layers.append(Layer(**common, conv_in=conv.in_proj.weight, conv_w=conv.conv.weight,
                                conv_b=conv.conv.bias, conv_out=conv.out_proj.weight))
        else:
            attn = block.self_attn
            params = config.rope_parameters[kind]
            inv = _inv_freq(hd, params["rope_theta"], params.get("partial_rotary_factor", 1.0))
            names = ("q_proj", "k_proj", "v_proj", "gate_proj")
            layers.append(Layer(**common, w_in=_fused_weight(attn, names),
                                split=(config.num_attention_heads * hd, config.num_key_value_heads * hd,
                                       config.num_key_value_heads * hd, config.num_attention_heads),
                                q_norm=attn.q_norm.weight, k_norm=attn.k_norm.weight, w_o=attn.o_proj.weight,
                                inv_freq=inv.to(lm.embed_tokens.weight.device),
                                window=config.sliding_window if kind == "sliding_attention" else None))
    return HybridLM(layers, lm.embed_tokens.weight, lm.embedding_norm.weight, config.norm_eps,
                    config.num_attention_heads, config.num_key_value_heads, hd, gated=True)


def from_lfm2(lm, config) -> HybridLM:
    """d1-3B: the original `hybrid.LanguageModel` (LFM2)."""
    layers = []
    heads, kv = config.num_attention_heads, config.num_key_value_heads
    hd = getattr(config, "head_dim", None) or config.hidden_size // heads
    for i, block in enumerate(lm.layers):
        kind = config.layer_types[i]
        common = dict(kind=kind, norm=block.operator_norm.weight, ffn_norm=block.ffn_norm.weight,
                      post_norm=None, post_ffn_norm=None,
                      w13=_fused_weight(block.feed_forward, ("w1", "w3")), w2=block.feed_forward.w2.weight)
        if kind == "conv":
            conv = block.conv
            layers.append(Layer(**common, conv_in=conv.in_proj.weight, conv_w=conv.conv.weight,
                                conv_b=conv.conv.bias, conv_out=conv.out_proj.weight))
        else:
            attn = block.self_attn
            inv = _inv_freq(hd, config.rope_parameters["rope_theta"])
            layers.append(Layer(**common, w_in=_fused_weight(attn, ("q_proj", "k_proj", "v_proj")),
                                split=(heads * hd, kv * hd, kv * hd),
                                q_norm=attn.q_layernorm.weight, k_norm=attn.k_layernorm.weight,
                                w_o=attn.out_proj.weight, inv_freq=inv.to(lm.embed_tokens.weight.device)))
    return HybridLM(layers, lm.embed_tokens.weight, lm.embedding_norm.weight, config.norm_eps,
                    heads, kv, hd, gated=False)

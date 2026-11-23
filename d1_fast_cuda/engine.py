"""The fast forward under each original engine (`model.engine`, the checkpoint's own
`runner.SystemOne`). Prompts, media, readout and answers stay the original code's;
only the language model's pass and the option logits are replaced.

- d1-2B: `SystemOne._slot_logits` (one padded batch -> its answer slots' logits and
  pointer reads). The multimodal wrapper still embeds the ids and places the media;
  its language model's forward is this package's (`HybridLM`), told by `_ctx` which
  rows the slots need.
- d1-3B: `SystemOne.run` with the original's batching, passes on this package's
  stack: one plain chain for one question, a packed tree for a state with several
  questions (or many one-question requests, which share their system prompt).

Both read the option rows of the tied embedding only, in float32, instead of the
full vocabulary in bfloat16: an answer is a softmax over its options' log-probs,
so the vocabulary's log-sum-exp cancels (calibration only divides by a temperature).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Sequence

import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from . import graphs as G
from .lm import HybridLM, Plan, from_lfm2, from_lfm3

TAIL = os.environ.get("D1_TAIL", "1") != "0"


def _module(engine, name: str):
    """A sibling module of the checkpoint's own code (`prompt`, `slots`, ...)."""
    return sys.modules[type(engine).__module__.rsplit(".", 1)[0] + "." + name]


def _pad_rows(rows: list, step: int = 4) -> list:
    return rows + [rows[-1]] * (G.bucket(len(rows), step) - len(rows))


class _Fast:
    lm: HybridLM

    def option_logits(self, hidden: torch.Tensor, ids: list[int]) -> torch.Tensor:
        """(n, len(ids)) float32 logits of the tied embedding's rows `ids`."""
        w = self.lm.embed[torch.tensor(ids, device=hidden.device)]
        return hidden.float() @ w.float().T


# ---------------------------------------------------------------------- d1-2B


def sees_mask(lengths: torch.Tensor, state: torch.Tensor, question: torch.Tensor | None, t: int) -> torch.Tensor:
    """`attention.Sees` of the original as one boolean mask (B, 1, T, T): the state
    both ways, a question token its own question (`set`), causal otherwise; padding
    keys hidden, a padding query sees itself."""
    dev = lengths.device
    q = torch.arange(t, device=dev)[None, :, None]
    kv = torch.arange(t, device=dev)[None, None, :]
    s = state[:, None, None]
    if question is None:
        m = (kv <= q) | ((q < s) & (kv < s))
    else:
        m = (q < s) | (kv < s) | (question[:, :, None] == question[:, None, :])
    m = (m & (kv < lengths[:, None, None])) | (q == kv)
    return m[:, None]


def sliding_mask(lengths: torch.Tensor, t: int, window: int) -> torch.Tensor:
    dev = lengths.device
    q = torch.arange(t, device=dev)[None, :, None]
    kv = torch.arange(t, device=dev)[None, None, :]
    m = ((kv <= q) & (kv > q - window) & (kv < lengths[:, None, None])) | (q == kv)
    return m[:, None]


class Fast2B(_Fast):
    def __init__(self, model):
        self.model = model
        self.engine = engine = model.engine  # the original engine; it fuses the projections
        slots = _module(engine, "slots")
        self.prompt, self.slots = _module(engine, "prompt"), slots
        self.vl = slots.text_backbone(model)
        self.lang = getattr(self.vl, "language_model", self.vl)
        self.lm = from_lfm3(self.lang, self.lang.config)
        self.window = self.lang.config.sliding_window
        self.graphs = G.Graphs()
        self._ctx: dict | None = None
        self.lang.forward = self._lm_forward
        engine._slot_logits = self._slot_logits

    def compile(self, mode=None):
        """torch.compile the language model's pass (`HybridLM.compile`) and, as the
        original `SystemOne.compile` does, the media encoders the model names
        (`compiled_parts`); single requests keep replaying as CUDA graphs."""
        self.lm.compile(dynamic=True)
        self.graphs.cache.clear()
        body = self.vl
        for name in getattr(self.model, "compiled_parts", ()):
            part = getattr(body, name)
            part.forward = torch.compile(part.forward, dynamic=True)

    # the original `SystemOne._slot_logits`, on this package's forward
    def _slot_logits(self, seqs: list[tuple], batch: list[int], slots: list[tuple], pointer: dict | None):
        e = self.engine
        ids, lengths, media = e._rows(seqs, batch)
        width = ids.shape[1]
        dev = ids.device
        mask = (torch.arange(width, device=dev) < torch.tensor(lengths, device=dev)[:, None]).long()
        state = None if e.attention == "causal" else [seqs[b][3] for b in batch]
        question = None if e.attention != "set" else [seqs[b][4] + [-1] * (width - len(seqs[b][4])) for b in batch]
        tail = TAIL and len(batch) == 1 and pointer is None
        self._ctx = {"lengths": lengths, "state": state, "question": question,
                     "rows": [s[3] for s in slots] if tail else None}
        try:
            out = self.vl(input_ids=ids, attention_mask=mask, use_cache=False, **e._on_device(media))
        finally:
            self._ctx = None
        hidden = out.last_hidden_state
        if tail:
            at = hidden[: len(slots)]
        else:
            at = hidden[torch.tensor([s[0] for s in slots], device=dev), torch.tensor([s[3] for s in slots], device=dev)]
        states = hidden[pointer["rows"].to(dev), : pointer["width"]] if pointer is not None else None
        codes = sorted({i for s in slots if not isinstance(s[2], self.prompt.Position)
                        for i in self._tokens(s[2])})
        vocab = self.lm.embed.shape[0]

        def project(x):
            full = torch.zeros(x.shape[0], vocab, device=x.device)
            if codes:
                full[:, codes] = self.option_logits(x, codes)
            return full

        rows = self.slots.split(at, states, pointer, project)
        return rows.codes, rows.positions

    def _tokens(self, q) -> tuple[int, ...]:
        """The option token ids `q`'s readout reads (`prompt.readout_table`, cached there)."""
        return self.prompt.readout_table(self.engine.tokenizer, q)[0]

    # the language model's forward, under the multimodal wrapper
    def _lm_forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, **_):
        ctx = self._ctx
        h = self.lm.embed[input_ids] if inputs_embeds is None else inputs_embeds
        if ctx is None:  # called outside `system_one`: a plain causal pass
            b, t = h.shape[:2]
            plan = Plan(positions=torch.arange(t, device=h.device))
            return BaseModelOutputWithPast(last_hidden_state=self.lm.run(h, plan))
        b, t = h.shape[:2]
        dev = h.device
        rows = ctx["rows"]
        if b == 1 and G.ENABLED and t <= G.MAX_TOKENS:
            out = self._graph(h, ctx)
        else:
            out = self.lm.run(h, self._plan(ctx, t, dev, rows))
        if rows is None:
            out = out[:, :t]
        return BaseModelOutputWithPast(last_hidden_state=out)

    def _plan(self, ctx: dict, t: int, dev, rows: list[int] | None, lengths=None) -> Plan:
        given = lengths
        lengths = torch.tensor(ctx["lengths"] if lengths is None else lengths, device=dev)
        if ctx["state"] is None:
            state = torch.zeros_like(lengths)
            question = None
            full = None if bool((lengths == t).all()) else sees_mask(lengths, state, None, t)
        else:
            state = torch.tensor(ctx["state"], device=dev)
            question = None
            if ctx["question"] is not None:
                q = [row + [-1] * (t - len(row)) for row in ctx["question"]]
                question = torch.tensor(q, device=dev)
            full = sees_mask(lengths, state, question, t)
        plan = Plan(positions=torch.arange(t, device=dev), full=full,
                    sliding=sliding_mask(lengths, t, self.window))
        if given is None and len(ctx["lengths"]) == 1 and ctx["lengths"][0] == t:
            # one unpadded row (the eager path of a long prompt): a lone question under `set`
            # sees all of itself, so its full layers need no mask, and the sliding layers
            # attend block by block
            if ctx["question"] is not None and max(ctx["question"][0]) <= 1:
                plan.full, plan.full_all = None, True
            plan.sliding, plan.window = None, self.window
        if rows is not None:
            windows, valid = self.lm.windows(rows, list(range(-1, t - 1)))
            plan.windows = torch.tensor(windows, device=dev)
            plan.valid = torch.tensor(valid, device=dev)
        return plan

    def _graph(self, h: torch.Tensor, ctx: dict) -> torch.Tensor:
        t = h.shape[1]
        tb = G.bucket(t)
        rows = ctx["rows"]
        dev = h.device
        padded_rows = _pad_rows(rows) if rows is not None else None
        plan = self._plan(ctx, tb, dev, padded_rows, lengths=[t])
        key = ("2b", tb, len(padded_rows) if rows is not None else None, ctx["state"] is None)
        inputs = {"h": torch.nn.functional.pad(h, (0, 0, 0, tb - t)), "full": plan.full, "sliding": plan.sliding,
                  "windows": plan.windows, "valid": plan.valid}

        def forward(s):
            return self.lm.run(s["h"], Plan(positions=torch.arange(tb, device=dev), full=s["full"],
                                                sliding=s["sliding"], windows=s["windows"], valid=s["valid"]))

        out = self.graphs.run(key, inputs, forward)
        return out[: len(rows)] if rows is not None else out


# ---------------------------------------------------------------------- d1-3B


class Tree:
    """A trunk of `p` tokens and branches of `lengths` tokens packed on one axis, as
    the original `hybrid.Tree`: the trunk causal, each branch over the trunk and its
    own earlier tokens, positions continuing the trunk, convolutions continuing it."""

    def __init__(self, p: int, lengths: list[int], pad: int = 0):
        self.p, self.lengths, self.n = p, list(lengths), len(lengths)
        self.starts = []
        at = p
        for n in lengths:
            self.starts.append(at)
            at += n
        self.total = at
        self.leaves = [s + n - 1 for s, n in zip(self.starts, lengths)]
        self.pad = pad  # rows after the last branch (graph buckets): they continue it
        self.bucketed = False  # a graph bucket's tree: one dense mask (static shapes)

    def mask(self, dev) -> torch.Tensor:
        t = self.total + self.pad
        idx = torch.arange(t, device=dev)
        branch = torch.full((t,), -1, device=dev)
        for i, (s, n) in enumerate(zip(self.starts, self.lengths)):
            branch[s:s + n] = i
        if self.pad:
            branch[self.total:] = self.n  # padding: its own group
        q, kv = idx[:, None], idx[None, :]
        trunk_kv = kv < self.p
        m = (trunk_kv & (kv <= q)) | ((branch[:, None] == branch[None, :]) & (branch[:, None] >= 0) & (kv <= q))
        m |= (q >= self.p) & trunk_kv & (branch[:, None] < self.n)
        return (m | (q == kv))[None, None]

    def parts(self, dev) -> tuple:
        """`Plan.parts`: the branches as a padded batch over the packed axis."""
        length = max(self.lengths)
        rows = torch.zeros(self.n, length, dtype=torch.long)
        valid = torch.zeros(self.n, length, dtype=torch.bool)
        for i, (s, n) in enumerate(zip(self.starts, self.lengths)):
            rows[i, :n] = torch.arange(s, s + n)
            valid[i, :n] = True
        keep = valid.flatten().nonzero().squeeze(1)
        branch = torch.full((self.total,), -1, dtype=torch.long)
        for i, (s, n) in enumerate(zip(self.starts, self.lengths)):
            branch[s:s + n] = i
        return (self.p, rows.to(dev), valid.to(dev), keep.to(dev), branch.to(dev))

    def positions(self) -> list[int]:
        pos = list(range(self.p))
        for n in self.lengths:
            pos += range(self.p, self.p + n)
        return pos + [0] * self.pad

    def prev(self) -> list[int]:
        """Each row's predecessor in its convolution's order: a branch starts after
        the trunk's last row."""
        prev = [r - 1 for r in range(self.p)]
        for s, n in zip(self.starts, self.lengths):
            prev += [self.p - 1 if self.p else -1] + list(range(s, s + n - 1))
        return prev + list(range(self.total - 1, self.total + self.pad - 1))

    def conv_order(self, keep: int, dev) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather and keep indices for one causal conv over all branches, each led by
        the trunk's last `keep` inputs (the padding rows continue the last branch),
        into the input left-padded by `keep` zeros."""
        order = list(range(keep + self.p))
        kept = list(range(self.p))  # output j belongs to order index j + keep
        lengths = list(self.lengths)
        lengths[-1] += self.pad
        starts = self.starts
        for s, n in zip(starts, lengths):
            lead = [keep + r if r >= 0 else 0 for r in range(self.p - keep, self.p)]
            base = len(order)
            order += lead + [keep + s + i for i in range(n)]
            kept += [base + i for i in range(n)]
        return torch.tensor(order, device=dev), torch.tensor(kept, device=dev)


class Fast3B(_Fast):
    def __init__(self, model):
        self.model = model
        self.engine = engine = model.engine
        self.prompt = _module(engine, "prompt")
        self.runner = _module(engine, "runner")
        self.lang = model.model.language_model
        self.lm = from_lfm2(self.lang, model.config.text_config)
        self.taps = {l.taps for l in self.lm.layers if not l.attention}
        self.graphs = G.Graphs()
        engine.run = self.run

    def compile(self, mode=None):
        """torch.compile the language model's pass (`HybridLM.compile`) and the
        SigLIP2 tower; single requests keep replaying as CUDA graphs."""
        self.lm.compile(dynamic=True)
        self.graphs.cache.clear()
        for part in (self.model.model.vision_tower, self.model.model.multi_modal_projector):
            part.forward = torch.compile(part.forward, dynamic=True)

    def logz(self, hidden: torch.Tensor, qs: Sequence) -> list[dict]:
        """Each row's option log-probs up to a constant, as {token id: value}."""
        tok = self.engine.tokenizer
        union = sorted({i for q in qs for g in self.prompt.readout_ids(tok, q) for i in g})
        values = self.option_logits(hidden, union).cpu().tolist()  # one transfer for every row
        return [dict(zip(union, row)) for row in values]

    # ------------------------------------------------------------- passes

    def hidden(self, embeds: torch.Tensor, tree: Tree | None) -> torch.Tensor:
        """Final hidden states at each branch's last token (a chain: its last row)."""
        dev = embeds.device
        t = embeds.shape[1]
        rows = tree.leaves if tree is not None else [t - 1]
        if G.ENABLED and t <= G.MAX_TOKENS:
            return self._graph(embeds, tree, rows)
        plan = self._plan(tree, t, dev, rows if TAIL else None)
        out = self.lm.run(embeds, plan)
        return out if TAIL else out[0, rows]

    def _plan(self, tree: Tree | None, t: int, dev, rows: list[int] | None) -> Plan:
        if tree is None:
            plan = Plan(positions=torch.arange(t, device=dev))
            prev = list(range(-1, t - 1))
        else:
            plan = Plan(positions=torch.tensor(tree.positions(), device=dev))
            if tree.bucketed:  # a graph bucket: one dense mask, static shapes
                plan.full = tree.mask(dev)
            else:  # attended by parts: no T x T mask
                plan.parts = tree.parts(dev)
            keep = next(iter(self.taps)) - 1
            plan.conv = tree.conv_order(keep, dev)
            prev = tree.prev()
        if rows is not None:
            windows, valid = self.lm.windows(rows, prev)
            plan.windows = torch.tensor(windows, device=dev)
            plan.valid = torch.tensor(valid, device=dev)
        return plan

    def _graph(self, embeds: torch.Tensor, tree: Tree | None, rows: list[int]) -> torch.Tensor:
        dev = embeds.device
        t = embeds.shape[1]
        tb = G.bucket(t)
        n = len(rows)
        padded_rows = _pad_rows(rows)
        if tree is None:
            # a chain: causal attention needs no mask, the padding rows come after the real ones
            plan = self._plan(None, tb, dev, padded_rows)
            key = ("3b-chain", tb, len(padded_rows))
        else:
            tree = Tree(tree.p, tree.lengths, pad=tb - t)
            tree.bucketed = True
            plan = self._plan(tree, tb, dev, padded_rows)
            key = ("3b-tree", tb, len(padded_rows), tree.n)
        inputs = {"h": torch.nn.functional.pad(embeds, (0, 0, 0, tb - t)), "positions": plan.positions,
                  "full": plan.full, "windows": plan.windows, "valid": plan.valid,
                  "order": plan.conv[0] if plan.conv else None, "kept": plan.conv[1] if plan.conv else None}

        def forward(s):
            conv = (s["order"], s["kept"]) if s["order"] is not None else None
            return self.lm.run(s["h"], Plan(positions=s["positions"], full=s["full"], windows=s["windows"],
                                                valid=s["valid"], conv=conv))

        return self.graphs.run(key, inputs, forward)[:n]

    def _chain(self, ids: list[int] | None = None, embeds=None) -> torch.Tensor:
        if embeds is None:
            embeds = self.lm.embed[torch.tensor([ids], device=self.engine.device)]
        return self.hidden(embeds, None)

    def _tree(self, trunk_embeds: torch.Tensor, branches: list[list[int]]) -> torch.Tensor:
        dev = trunk_embeds.device
        packed = torch.tensor([t for b in branches for t in b], device=dev)
        h = torch.cat((trunk_embeds, self.lm.embed[packed][None]), dim=1)
        return self.hidden(h, Tree(trunk_embeds.shape[1], [len(b) for b in branches]))

    def _embed_vision(self, ids: torch.Tensor, vision: dict) -> torch.Tensor:
        """The trunk's embeddings with the pictures' features placed, as the original
        `Lfm2VlForConditionalGeneration.answer`."""
        m = self.model
        embeds = m.get_input_embeddings()(ids)
        features = torch.cat(m.model.get_image_features(**vision).pooler_output).to(embeds.dtype)
        slots = m.model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
        return embeds.masked_scatter(slots, features)

    # --------------------------------------------- the original `run`, on these passes

    @torch.inference_mode()
    def run(self, requests: Sequence[tuple[Any, list, Sequence]]) -> list[tuple[list[list[float]], int]]:
        e = self.engine
        out: list = [None] * len(requests)
        single = [i for i, (_, qs, images) in enumerate(requests) if len(qs) == 1 and not images]
        rows = [e.tokenizer.encode(e.render(requests[i][0], requests[i][1][0]), add_special_tokens=False)
                for i in single]
        for chunk in e._plan([len(r) for r in rows]):
            group = [rows[j] for j in chunk]
            qs = [requests[single[j]][1][0] for j in chunk]
            if len(group) == 1:
                h = self._chain(group[0])
            else:
                shared = 0
                while shared < min(map(len, group)) - 1 and len({r[shared] for r in group}) == 1:
                    shared += 1
                trunk = self.lm.embed[torch.tensor([group[0][:shared]], device=e.device)]
                h = self._tree(trunk, [r[shared:] for r in group])
            for j, q, z in zip(chunk, qs, self.logz(h, qs)):
                out[single[j]] = ([e._readout(q, z)], len(rows[j]))
        for i, (state, qs, images) in enumerate(requests):
            if out[i] is None:
                out[i] = self._request(state, qs, images)
        return out

    def _request(self, state: Any, qs: list, images: Sequence) -> tuple[list[list[float]], int]:
        e, p = self.engine, self.prompt
        pics = [self.runner.cap_pixels(im) for im in images]
        prefix = p.prefix_text(e.tokenizer, state, e.bos, e.state_style, e.system,
                               e._image_markup(len(pics)) if pics else "")
        suffixes = [p.suffix_text(e.tokenizer, q, e.lead, e.option_style) for q in qs]
        if not pics:
            trunk_ids = e.tokenizer.encode(prefix, add_special_tokens=False)
            trunk = self.lm.embed[torch.tensor([trunk_ids], device=e.device)]
        elif len(qs) == 1:  # the whole prompt in one plain pass
            inputs = e._image_inputs(prefix + suffixes[0], pics)
            ids = inputs.pop("input_ids")
            inputs.pop("attention_mask", None)
            h = self._chain(embeds=self._embed_vision(ids, inputs))
            return [e._readout(qs[0], self.logz(h, qs)[0])], int(ids.shape[1])
        else:
            vision = e._image_inputs(prefix, pics)
            ids = vision.pop("input_ids")
            vision.pop("attention_mask", None)
            trunk_ids = ids[0].tolist()
            trunk = self._embed_vision(ids, vision)
        branches = [e.tokenizer.encode(s, add_special_tokens=False) for s in suffixes]
        probs: list[list[float]] = []
        for chunk in e._plan([len(b) for b in branches]):
            h = self._tree(trunk, [branches[j] for j in chunk])
            cq = [qs[j] for j in chunk]
            probs += [e._readout(q, z) for q, z in zip(cq, self.logz(h, cq))]
        return probs, len(trunk_ids) + sum(map(len, branches))

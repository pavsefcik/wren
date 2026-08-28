"""Patched MoE forward for Qwen3.6-35B-A3B that streams experts from the cache.

The stock ``Qwen3_5MoeSparseMoeBlock`` computes over the full fused expert table
via ``switch_mlp``.  We replace it with an equivalent forward that:

1. runs the router (unchanged),
2. ensures only the selected experts are materialized in the ``ExpertCache``,
3. gathers those experts into small stacked tensors and runs ``gather_qmm``
   (the same quantized-gather kernel the stock path uses),
4. hands router logits to the predictor so future experts are prefetched.

Two predictor strategies are available: a trained MLP (``predictor``) and a
residual-stream *lookahead* that runs the real next-layer router on the current
hidden state (approximating ``h_{L+1} ~= h_L``) to forecast the next layer's
experts before the MoE output is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm.models.activations import swiglu

from .expert_cache import ExpertCache


@dataclass
class MoEContext:
    cache: ExpertCache
    layer: int
    predictor: object = None
    recorder: object = None
    num_layers: int = 0
    group_size: int = 64
    bits: int = 4
    mode: str = "affine"
    # residual-stream lookahead: list of (gate, norm_ratio) for layers L+1, L+2, ...
    ahead_gates: tuple = ()
    prefetch_top_k: int = 0


def _top_ids(arr, k: int):
    p = np.asarray(arr, dtype=np.float32)
    p = p.reshape(-1, p.shape[-1]).sum(axis=0)
    return p.argsort()[::-1][:k].tolist()


def _switch(
    ctx: MoEContext, x: mx.array, inds: mx.array, gates: mx.array, logits: mx.array
) -> mx.array:
    ids_flat = [int(i) for i in np.asarray(inds).reshape(-1).tolist()]
    distinct = list(dict.fromkeys(ids_flat))

    ctx.cache.ensure(ctx.layer, distinct)

    logits_f32 = logits.astype(mx.float32)

    if ctx.recorder is not None:
        ctx.recorder.observe(ctx.layer, ctx.num_layers, logits_f32, inds)

    _prefetch(ctx, x, logits_f32)

    local = {e: i for i, e in enumerate(distinct)}
    linds = mx.array([local[e] for e in ids_flat], dtype=mx.uint32).reshape(inds.shape)

    xr = mx.expand_dims(x, (-2, -3))

    def qmm(proj: str, inp: mx.array) -> mx.array:
        w = ctx.cache.stacked(ctx.layer, proj, "weight", distinct)
        s = ctx.cache.stacked(ctx.layer, proj, "scales", distinct)
        b = ctx.cache.stacked(ctx.layer, proj, "biases", distinct)
        return mx.gather_qmm(
            inp,
            w,
            s,
            b,
            rhs_indices=linds,
            transpose=True,
            group_size=ctx.group_size,
            bits=ctx.bits,
            mode=ctx.mode,
        )

    x_gate = qmm("gate_proj", xr)
    x_up = qmm("up_proj", xr)
    h = swiglu(x_gate, x_up)
    x = qmm("down_proj", h)

    return x.squeeze(-2)


def _prefetch(ctx: MoEContext, x: mx.array, logits_f32: mx.array) -> None:
    if ctx.cache.prefetcher is None:
        return

    top_k = ctx.prefetch_top_k
    if ctx.predictor is not None:
        # trained MLP handles same-layer warm + next-layer prediction
        hints = ctx.predictor.hints(ctx.layer, np.asarray(logits_f32), ctx.num_layers)
        ctx.cache.prefetcher.submit(hints)
        return

    if top_k <= 0:
        return

    if ctx.ahead_gates:
        # residual-stream lookahead: run the real future-layer routers on the
        # current hidden state (h_{L+d} ~= h_L), predicting several layers ahead
        # so the async prefetcher has time to finish the SSD reads.
        arrays = [logits_f32]
        for gate, ratio in ctx.ahead_gates:
            arrays.append(gate(x * ratio).astype(mx.float32))
        mx.eval(*arrays)
        hints = [(ctx.layer, e) for e in _top_ids(arrays[0], top_k)]
        for d, arr in enumerate(arrays[1:], start=1):
            hints += [(ctx.layer + d, e) for e in _top_ids(arr, top_k)]
        ctx.cache.prefetcher.submit(hints)
    else:
        cur = _top_ids(np.asarray(logits_f32), top_k)
        ctx.cache.prefetcher.submit([(ctx.layer, e) for e in cur])


def make_moe_forward():
    def _call(self, x):
        logits = self.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = _switch(self._loki, x, inds, gates, logits)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    return _call


def patch_moe(
    model,
    cache: ExpertCache,
    predictor: object = None,
    recorder: object = None,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
    prefetch_top_k: int = 0,
    prefetch_lookahead: int = 0,
) -> None:
    """Attach the streaming MoE forward to every MoE block in ``model``."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    lm = model.language_model
    num_layers = lm.model.args.num_hidden_layers
    layers = lm.model.layers

    Qwen3_5MoeSparseMoeBlock.__call__ = make_moe_forward()

    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if not isinstance(mlp, Qwen3_5MoeSparseMoeBlock):
            continue

        ahead = []
        for d in range(1, prefetch_lookahead + 1):
            if i + d >= num_layers:
                break
            gate = layers[i + d].mlp.gate
            ratio = (
                layers[i + d].post_attention_layernorm.weight
                / layers[i].post_attention_layernorm.weight
            )
            ahead.append((gate, ratio))

        mlp._loki = MoEContext(
            cache=cache,
            layer=i,
            predictor=predictor,
            recorder=recorder,
            num_layers=num_layers,
            group_size=group_size,
            bits=bits,
            mode=mode,
            ahead_gates=tuple(ahead),
            prefetch_top_k=prefetch_top_k,
        )

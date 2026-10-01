"""Patched MoE forward for Qwen3.6-35B-A3B that streams experts from the cache.

The stock ``Qwen3_5MoeSparseMoeBlock`` computes over the full fused expert
table via ``switch_mlp``. We replace it with an equivalent forward that:

1. runs the router (unchanged),
2. ensures only the selected experts are materialized in the ``ExpertCache``,
3. gathers those experts into small stacked tensors and runs ``gather_qmm``
   (the same quantized-gather kernel the stock path uses).
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
    group_size: int = 64
    bits: int = 4
    mode: str = "affine"


def _switch(ctx: MoEContext, x: mx.array, inds: mx.array) -> mx.array:
    ids_flat = [int(i) for i in np.asarray(inds).reshape(-1).tolist()]
    distinct = list(dict.fromkeys(ids_flat))

    ctx.cache.ensure(ctx.layer, distinct)

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


def make_moe_forward():
    def _call(self, x):
        logits = self.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)

        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        scores = scores / scores.sum(axis=-1, keepdims=True)

        y = _switch(self._wren, x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y

    return _call


def patch_moe(
    model,
    cache: ExpertCache,
    group_size: int = 64,
    bits: int = 4,
    mode: str = "affine",
) -> None:
    """Attach the streaming MoE forward to every MoE block in ``model``."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    lm = model.language_model
    layers = lm.model.layers

    Qwen3_5MoeSparseMoeBlock.__call__ = make_moe_forward()

    for i, layer in enumerate(layers):
        mlp = layer.mlp
        if not isinstance(mlp, Qwen3_5MoeSparseMoeBlock):
            continue
        mlp._wren = MoEContext(
            cache=cache,
            layer=i,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

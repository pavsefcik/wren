"""Validate the streaming MoE math against a stock (full-table) path.

Builds a synthetic safetensors file with ``switch_mlp``-shaped tensors, runs
the patched forward against a fake block, and compares to ``gather_qmm`` over
the full expert table.
"""

import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

from loki.engine.expert_cache import ExpertCache
from loki.engine.expert_store import ExpertStore
from loki.engine.moe import MoEContext


def _write_synthetic(path: Path, num_experts: int, hidden: int, inter: int):
    import safetensors.mlx as sm

    rng = np.random.default_rng(0)
    tensors = {}
    base = "language_model.model.layers.0.mlp.switch_mlp"
    for proj, (o, i) in {
        "gate_proj": (inter, hidden),
        "up_proj": (inter, hidden),
        "down_proj": (hidden, inter),
    }.items():
        w = mx.array(rng.normal(0, 0.02, size=(num_experts, o, i)).astype(np.float32))
        wq, sc, bs = mx.quantize(w, group_size=64, bits=4)
        tensors[f"{base}.{proj}.weight"] = wq
        tensors[f"{base}.{proj}.scales"] = sc
        tensors[f"{base}.{proj}.biases"] = bs

    sm.save_file(tensors, str(path))


def test_streaming_moe_matches_stock():
    num_experts, hidden, inter, top_k = 16, 128, 64, 4
    tmp = Path(tempfile.mkdtemp())
    shard = tmp / "model-00001-of-00001.safetensors"
    _write_synthetic(shard, num_experts, hidden, inter)

    store = ExpertStore(tmp)
    cache = ExpertCache(store, budget_bytes=1 << 30)

    rng = np.random.default_rng(1)
    x = mx.array(rng.normal(0, 1, size=(1, 5, hidden)).astype(np.float32))
    gate_w = mx.array(rng.normal(0, 0.02, size=(num_experts, hidden)).astype(np.float32))
    logits = x @ gate_w.T
    gates = mx.softmax(logits, axis=-1, precise=True)
    inds = mx.argpartition(gates, kth=-top_k, axis=-1)[..., -top_k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)

    # full-table tensors for the stock comparison
    stock = {}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        parts = {}
        for part in ("weight", "scales", "biases"):
            parts[part] = mx.stack([store.expert(0, proj, part, e) for e in range(num_experts)])
        stock[proj] = parts

    xr = mx.expand_dims(x, (-2, -3))
    from mlx_lm.models.activations import swiglu

    def stock_qmm(proj, inp):
        return mx.gather_qmm(
            inp,
            stock[proj]["weight"],
            stock[proj]["scales"],
            stock[proj]["biases"],
            rhs_indices=inds,
            transpose=True,
            group_size=64,
            bits=4,
        )

    xg = stock_qmm("gate_proj", xr)
    xu = stock_qmm("up_proj", xr)
    h = swiglu(xg, xu)
    xd = stock_qmm("down_proj", h).squeeze(-2)
    stock_y = (xd * scores[..., None]).sum(axis=-2)

    ctx = MoEContext(cache=cache, layer=0, num_layers=1, group_size=64, bits=4)
    from loki.engine.moe import _switch

    y = _switch(ctx, x, inds, gates, logits)
    patched_y = (y * scores[..., None]).sum(axis=-2)

    mx.eval(stock_y, patched_y)
    diff = mx.abs(stock_y - patched_y).max().item()
    print("max abs diff:", diff)
    assert diff < 1e-3, diff


if __name__ == "__main__":
    test_streaming_moe_matches_stock()
    print("PASS")

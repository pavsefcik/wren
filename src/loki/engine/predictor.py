"""Learned cross-layer expert predictor.

A small MLP (``256 -> hidden -> 256``) maps the router logits at layer ``L`` to
a score vector over experts at layer ``L+1``.  It is trained on routing traces
collected by :class:`~loki.engine.trace.TraceRecorder`, and at inference time its
top-K predictions are handed to the prefetcher so the next layer's experts are
already on their way while the current layer computes.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class LearnedPredictor:
    def __init__(self, path: str, top_k: int = 16, lookahead: int = 8):
        d = np.load(path)
        self.w1 = d["w1"].astype(np.float32)
        self.b1 = d["b1"].astype(np.float32)
        self.w2 = d["w2"].astype(np.float32)
        self.b2 = d["b2"].astype(np.float32)
        self.top_k = top_k
        self.lookahead = lookahead

    def forward(self, logits: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, logits @ self.w1 + self.b1)
        return h @ self.w2 + self.b2

    def hints(self, layer: int, logits_np, num_layers: int) -> List[Tuple[int, int]]:
        p = np.asarray(logits_np, dtype=np.float32)
        p = p.reshape(-1, p.shape[-1]).sum(axis=0)  # aggregate tokens

        # same-layer warm (actual routing, not predicted)
        top = p.argsort()[::-1][: self.top_k].tolist()
        hints = [(layer, e) for e in top]

        # learned cross-layer prediction
        if layer + 1 < num_layers and self.lookahead > 0:
            scores = self.forward(p)
            nxt = scores.argsort()[::-1][: self.lookahead].tolist()
            hints += [(layer + 1, e) for e in nxt]
        return hints


def train_predictor(
    trace_path: str,
    output_path: str,
    hidden: int = 256,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 3e-3,
    num_experts: int = 256,
    top_k: int = 8,
) -> dict:
    """Train the MLP on recorded traces and save weights to ``output_path``.

    Uses MLX (the same framework as inference) and soft-label cross-entropy
    against a uniform distribution over the true top-K experts.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from .trace import load_traces

    X, Y = load_traces(trace_path, top_k)
    n = X.shape[0]
    assert n > 0, "no traces found"

    target = np.zeros((n, num_experts), dtype=np.float32)
    for i in range(n):
        target[i, Y[i]] = 1.0 / top_k

    X_np = np.ascontiguousarray(X, dtype=np.float32)
    target_np = np.ascontiguousarray(target, dtype=np.float32)

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            scale = num_experts ** -0.5
            self.w1 = mx.random.normal(shape=(num_experts, hidden)) * scale
            self.b1 = mx.zeros((hidden,))
            self.w2 = mx.random.normal(shape=(hidden, num_experts)) * (hidden ** -0.5)
            self.b2 = mx.zeros((num_experts,))

        def __call__(self, x):
            h = mx.maximum(0.0, x @ self.w1 + self.b1)
            return h @ self.w2 + self.b2

    model = MLP()

    def loss_fn(m, x, y):
        logits = m(x)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        return -mx.sum(y * logp, axis=-1).mean()

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = mx.array(X_np[idx])
            yb = mx.array(target_np[idx])
            loss, grads = loss_and_grad(model, xb, yb)
            optimizer.update(model, grads)
            mx.eval(model.parameters())
            total += loss.item() * idx.shape[0]
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs}: loss={total / n:.4f}")

    w1n = np.asarray(model.w1)
    b1n = np.asarray(model.b1)
    w2n = np.asarray(model.w2)
    b2n = np.asarray(model.b2)
    np.savez(
        output_path,
        w1=w1n.astype(np.float32),
        b1=b1n.astype(np.float32),
        w2=w2n.astype(np.float32),
        b2=b2n.astype(np.float32),
    )

    # quick top-k recovery report on the training set
    p = LearnedPredictor(output_path, top_k=16, lookahead=8)
    hit = 0.0
    for i in range(0, n, 4096):
        scores = p.forward(X_np[i : i + 4096])
        pred = np.argpartition(scores, -top_k, axis=-1)[:, -top_k:]
        for j in range(pred.shape[0]):
            hit += len(set(pred[j].tolist()) & set(Y[i + j].tolist())) / top_k
    return {"samples": n, "topk_recovery": round(hit / n, 4)}

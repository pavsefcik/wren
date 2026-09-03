"""Routing-trace recorder for training the learned expert predictor.

During decode we record, for every layer transition ``L -> L+1``, the router
logits at layer ``L`` (the predictor input) and the top-K expert ids actually
selected at layer ``L+1`` (the predictor target).

Records are appended as fixed-size binary rows so a long session can stream to
disk without growing memory::

    float32[256] logits  (1024 bytes)
    int16[K] expert ids   (2*K bytes)
"""

from __future__ import annotations

import numpy as np

RECORD_BYTES = 256 * 4 + 8 * 2  # logits + top-8 ids


class TraceRecorder:
    def __init__(self, path: str, top_k: int = 8):
        self.path = path
        self.top_k = top_k
        self._f = open(path, "ab")
        self._buf: dict = {}
        self.count = 0

    def observe(self, layer: int, num_layers: int, logits, inds) -> None:
        """Record one layer's routing for the current token (decode only)."""
        if logits.shape[1] != 1:
            return  # only single-token (decode) steps
        logits_np = np.asarray(logits, dtype=np.float32).reshape(256)
        ids_np = np.asarray(inds, dtype=np.int16).reshape(-1)[: self.top_k]
        self._buf[layer] = (logits_np, ids_np)
        if layer == num_layers - 1:
            self._flush()

    def _flush(self) -> None:
        for layer in range(len(self._buf) - 1):
            if layer not in self._buf or (layer + 1) not in self._buf:
                continue
            x = self._buf[layer][0]
            y = self._buf[layer + 1][1]
            self._f.write(x.tobytes())
            self._f.write(y.astype(np.int16).tobytes())
            self.count += 1
        self._buf.clear()
        self._f.flush()

    def close(self) -> None:
        self._flush() if self._buf else None
        self._f.close()


def load_traces(path: str, top_k: int = 8):
    """Read a trace file into ``(X, Y)``: ``X`` [N, 256] logits, ``Y`` [N, K] ids."""
    data = np.fromfile(path, dtype=np.uint8)
    rows = data.shape[0] // RECORD_BYTES
    data = data[: rows * RECORD_BYTES]
    arr = data.reshape(rows, RECORD_BYTES)
    X = np.frombuffer(arr[:, :1024].tobytes(), dtype=np.float32).reshape(rows, 256)
    Y = np.frombuffer(arr[:, 1024:].tobytes(), dtype=np.int16).reshape(rows, top_k)
    return X, Y

"""Lazy partial reads of MoE expert weights from safetensors shards.

Each routed expert tensor is stored contiguously in a shard with shape
``[num_experts, ...]``.  We parse the safetensors header once to find byte
offsets, keep the shard file descriptors open, and read a single expert's bytes
with ``os.pread`` on demand.  This gives true partial reads (only the requested
expert touches the disk) without the per-slice overhead of safetensors' Python
backend.
"""

from __future__ import annotations

import glob
import json
import os
import re
import struct
from pathlib import Path
from typing import Dict, Tuple

import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
import mlx.core as mx
import numpy as np

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
PARTS = ("weight", "scales", "biases")

_NP_DTYPE = {
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.uint16,
    "I16": np.int16,
    "U32": np.uint32,
    "I32": np.int32,
    "U64": np.uint64,
    "I64": np.int64,
    "F16": np.float16,
    "BF16": ml_dtypes.bfloat16,
    "F32": np.float32,
    "F64": np.float64,
}

_KEY_RE = re.compile(
    r"language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.(\w+_proj)\.(\w+)"
)


class _Tensor:
    __slots__ = ("fd", "data_start", "per_expert_bytes", "inner_shape", "np_dtype")

    def __init__(self, fd: int, data_start: int, per_expert_bytes: int, inner_shape, np_dtype):
        self.fd = fd
        self.data_start = data_start
        self.per_expert_bytes = per_expert_bytes
        self.inner_shape = inner_shape
        self.np_dtype = np_dtype


class ExpertStore:
    def __init__(self, model_dir: Path):
        shard_paths = sorted(glob.glob(str(model_dir / "*.safetensors")))
        shard_paths = [p for p in shard_paths if not p.endswith("consolidated.safetensors")]
        if not shard_paths:
            raise FileNotFoundError(f"No safetensors shards found in {model_dir}")

        self._fds = []
        self._tensors: Dict[Tuple[int, str, str], _Tensor] = {}
        self.num_layers = 0
        self.num_experts = 0

        for path in shard_paths:
            fd = os.open(path, os.O_RDONLY)
            self._fds.append(fd)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                meta = json.loads(fh.read(n))
            base = 8 + n  # data_offsets are relative to the data section

            for key, info in meta.items():
                m = _KEY_RE.match(key)
                if not m:
                    continue
                layer = int(m.group(1))
                proj = m.group(2)
                part = m.group(3)
                shape = info["shape"]
                dtype = info["dtype"]
                start, end = info["data_offsets"]
                per = (end - start) // shape[0]
                np_dtype = _NP_DTYPE[dtype]
                self._tensors[(layer, proj, part)] = _Tensor(
                    fd, base + start, per, tuple(shape[1:]), np_dtype
                )
                self.num_layers = max(self.num_layers, layer + 1)
                self.num_experts = max(self.num_experts, shape[0])

    def total_expert_bytes(self) -> int:
        return self.num_experts * sum(
            self.expert_bytes(layer) for layer in range(self.num_layers)
        )

    def read_bytes(self, layer: int, proj: str, part: str, idx: int) -> bytes:
        t = self._tensors[(layer, proj, part)]
        return os.pread(t.fd, t.per_expert_bytes, t.data_start + idx * t.per_expert_bytes)

    def expert(self, layer: int, proj: str, part: str, idx: int) -> mx.array:
        t = self._tensors[(layer, proj, part)]
        data = self.read_bytes(layer, proj, part, idx)
        arr = np.frombuffer(data, dtype=t.np_dtype).reshape(t.inner_shape)
        return mx.array(arr)

    def dtype_of(self, layer: int, proj: str, part: str):
        return self._tensors[(layer, proj, part)].np_dtype

    def inner_shape(self, layer: int, proj: str, part: str):
        return self._tensors[(layer, proj, part)].inner_shape

    def expert_bytes(self, layer: int) -> int:
        total = 0
        for proj in PROJECTIONS:
            for part in PARTS:
                total += self._tensors[(layer, proj, part)].per_expert_bytes
        return total

"""Bounded, LFU/LRU cache of materialized MoE expert weights.

The cache stores *materialized* (``mx.eval``'d) expert tensors keyed by
``(layer, expert_id)``.  A byte budget bounds total residency; eviction follows
least-frequently-used with a least-recently-used tiebreak, mirroring the
TurboFieldfare slot policy but measured in bytes instead of slots.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .expert_store import PARTS, PROJECTIONS, ExpertStore


class ExpertCache:
    def __init__(
        self,
        store: ExpertStore,
        budget_bytes: int,
        eviction: str = "lfu",
        prefetch_policy: Optional["PrefetchPolicy"] = None,
        prefetcher=None,
    ):
        self.store = store
        self.budget = budget_bytes
        self.eviction = eviction
        self.prefetch_policy = prefetch_policy
        self.prefetcher = prefetcher

        # (layer, expert) -> {proj: {part: mx.array}}
        self._entries: Dict[Tuple[int, int], Dict[str, Dict[str, mx.array]]] = {}
        self._freq: Dict[Tuple[int, int], int] = {}
        self._last: Dict[Tuple[int, int], float] = {}
        self._bytes: Dict[Tuple[int, int], int] = {}

        self.total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._clock = 0.0
        self._lock = threading.Lock()

    # -- core ----------------------------------------------------------------

    def ensure(self, layer: int, expert_ids: List[int]) -> None:
        """Materialize ``expert_ids`` for ``layer``, fetching cold misses.

        Room for cold experts is freed *before* insertion so the experts about
        to be used are never evicted by their own fetch.
        """
        cold = []
        now = time.monotonic()
        with self._lock:
            for e in expert_ids:
                key = (layer, e)
                if key in self._entries:
                    self.hits += 1
                    self._freq[key] += 1
                    self._last[key] = now
                else:
                    self.misses += 1
                    cold.append(e)

        if cold:
            need = self.store.expert_bytes(layer) * len(cold)
            self._evict_to(self.budget - need)
            with self._lock:
                for e in cold:
                    self._insert(layer, e, now)

    def _insert(self, layer: int, e: int, now: float) -> None:
        import numpy as np

        entry: Dict[str, Dict[str, mx.array]] = {}
        nbytes = 0

        ready = self.prefetcher.take(layer, e) if self.prefetcher is not None else None

        for proj in PROJECTIONS:
            pd: Dict[str, mx.array] = {}
            for part in PARTS:
                if ready is not None:
                    dtype = self.store.dtype_of(layer, proj, part)
                    shape = self.store.inner_shape(layer, proj, part)
                    arr = mx.array(np.frombuffer(ready[proj][part], dtype=dtype).reshape(shape))
                else:
                    arr = self.store.expert(layer, proj, part, e)
                pd[part] = arr
                nbytes += arr.nbytes
            entry[proj] = pd
        key = (layer, e)
        self._entries[key] = entry
        self._freq[key] = 1
        self._last[key] = now
        self._bytes[key] = nbytes
        self.total_bytes += nbytes

    # -- gather --------------------------------------------------------------

    def stacked(self, layer: int, proj: str, part: str, expert_ids: List[int]) -> mx.array:
        """Return a ``[len(expert_ids), ...]`` tensor of a projection part."""
        arrays = [self._entries[(layer, e)][proj][part] for e in expert_ids]
        return mx.stack(arrays)

    # -- stats ---------------------------------------------------------------

    def stats(self) -> dict:
        with self._lock:
            total = self.hits + self.misses
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
                "resident_bytes": self.total_bytes,
                "budget_bytes": self.budget,
                "evictions": self.evictions,
                "resident_experts": len(self._entries),
            }

    # -- eviction ------------------------------------------------------------

    def _evict_to(self, target: int) -> None:
        if self.total_bytes <= target:
            return
        with self._lock:
            if self.total_bytes <= target:
                return
            items = sorted(
                self._entries.keys(),
                key=lambda k: (self._freq[k], self._last[k]),
            )
            for key in items:
                if self.total_bytes <= target:
                    break
                self.total_bytes -= self._bytes[key]
                del self._entries[key]
                del self._freq[key]
                del self._last[key]
                del self._bytes[key]
                self.evictions += 1


class PrefetchPolicy:
    """Decides which experts to prefetch from router logits.

    V1 heuristics (no training):

    * ``top_k``       -- keep the top-K experts per layer warm (>= num_experts_per_tok).
    * ``lookahead``   -- prefetch the current layer's top-K experts for the
                         *next* layer as a cross-layer co-activation guess.
    """

    def __init__(self, num_experts_per_tok: int, top_k: int = 16, lookahead: int = 8):
        self.num_experts_per_tok = num_experts_per_tok
        self.top_k = top_k
        self.lookahead = lookahead

    def hints(self, layer: int, gate_probs, num_layers: int) -> List[Tuple[int, int]]:
        """Return a list of ``(layer, expert_id)`` to prefetch.

        ``gate_probs`` is the router softmax for the current layer, shape
        ``[..., num_experts]``.  Across all tokens in the current step we
        aggregate to a per-expert score and keep the top-K.
        """
        p = np.asarray(gate_probs)
        probs = p.reshape(-1, p.shape[-1]).sum(axis=0)
        top = probs.argsort()[::-1][: self.top_k].tolist()

        hints = [(layer, e) for e in top]
        if layer + 1 < num_layers and self.lookahead > 0:
            hints += [(layer + 1, e) for e in top[: self.lookahead]]
        return hints

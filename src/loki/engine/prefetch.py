"""Background expert prefetching.

The predictor hands us a set of ``(layer, expert)`` hints.  A worker thread
reads those experts' bytes from the checkpoint with ``os.pread`` (thread-safe
I/O) while the decode thread keeps computing, and stores them in a bounded
"ready" buffer of raw bytes.  The decode thread converts bytes to ``mx.array``
on demand — MLX arrays are never constructed on the worker thread.
"""

from __future__ import annotations

import queue
import threading
from typing import Dict, Optional, Tuple

from .expert_store import PARTS, PROJECTIONS, ExpertStore


class Prefetcher:
    def __init__(self, store: ExpertStore, max_pending_bytes: int = 256 * (1 << 20)):
        self.store = store
        self.max_pending_bytes = max_pending_bytes
        self._queue: "queue.Queue[Tuple[int, int]]" = queue.Queue()
        self._ready: Dict[Tuple[int, int], Dict[str, Dict[str, bytes]]] = {}
        self._ready_bytes = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, pairs) -> None:
        for layer, e in pairs:
            self._queue.put((layer, e))

    def take(self, layer: int, e: int) -> Optional[Dict[str, Dict[str, bytes]]]:
        with self._lock:
            entry = self._ready.pop((layer, e), None)
            if entry is not None:
                n = sum(len(b) for pd in entry.values() for b in pd.values())
                self._ready_bytes -= n
        return entry

    def close(self) -> None:
        self._stop.set()

    # -- internals -----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                layer, e = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._ready_bytes >= self.max_pending_bytes:
                continue

            try:
                entry: Dict[str, Dict[str, bytes]] = {}
                nbytes = 0
                for proj in PROJECTIONS:
                    pd = {}
                    for part in PARTS:
                        data = self.store.read_bytes(layer, proj, part, e)
                        pd[part] = data
                        nbytes += len(data)
                    entry[proj] = pd

                with self._lock:
                    if (layer, e) not in self._ready:
                        self._ready[(layer, e)] = entry
                        self._ready_bytes += nbytes
            except Exception:
                pass

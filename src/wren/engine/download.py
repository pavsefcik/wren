"""One smooth Rich progress bar in place of huggingface_hub's raw tqdm flood.

The first ``wren`` run downloads ~20 GB of model weights via
``huggingface_hub.snapshot_download``, which normally prints a whole stack of
raw tqdm bars straight to stderr ("Fetching N files", "Downloading bytes",
"Reconstructing (incomplete total...)..."). We hand ``snapshot_download`` a
custom ``tqdm_class`` that swallows the native output and instead drives a
single Rich progress widget. Size / speed / ETA are computed ourselves from the
byte counts huggingface reports, so they're correct from the first frame (Rich's
own speed/ETA columns need several seconds of history and a known total).
"""

from __future__ import annotations

import threading
import time

from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

# snapshot_download's disk-write bar. It carries the true ~20 GB denominator
# (the network "Downloading bytes" bar only keeps a growing estimate), so it's
# the one worth showing.
_PRIMARY_DESC = "Reconstructing (incomplete total...)"

# Rate at which the on-screen stats are recomputed (smooth, not per byte).
_REPORT_EVERY = 0.25

_lock = threading.RLock()
_progress: Progress | None = None
_task_id: int | None = None
_state: dict = {"t0": 0.0, "last": 0.0, "n": 0.0, "total": None}


def _units(bytes_: float) -> str:
    """Render a byte count as a compact, human-readable string."""
    for divisor, suffix in ((1 << 30, "GB"), (1 << 20, "MB"), (1 << 10, "KB")):
        if bytes_ >= divisor:
            return f"{bytes_ / divisor:.1f} {suffix}"
    return f"{bytes_:.0f} B"


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _ensure() -> tuple[Progress, int]:
    """Create (once) the shared Rich widget returning (progress, task_id)."""
    global _progress, _task_id
    with _lock:
        if _progress is None:
            _progress = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(bar_width=28),
                TextColumn("{task.fields[pct]}", justify="right"),
                TextColumn("{task.fields[stat]}"),
                refresh_per_second=8,
            )
            _progress.start()
        if _task_id is None:
            _task_id = _progress.add_task(
                "Downloading model…",
                total=None,
                pct="0%",
                stat="0.0 B / ?  •  ?",
            )
        return _progress, _task_id


def _report(progress: Progress, task: int) -> None:
    """Recompute and push the percent/speed/ETA fields (throttled)."""
    now = time.monotonic()
    if _state["t0"] and now - _state.get("last", 0) < _REPORT_EVERY:
        return
    _state["last"] = now

    n = _state.get("n", 0.0)
    total = _state.get("total")
    elapsed = (now - _state["t0"]) if _state["t0"] else 1e-9
    rate = n / elapsed if elapsed > 0 else 0.0

    if total:
        pct = f"{min(100.0, 100.0 * n / total):.0f}%"
        size = f"{_units(n)} / {_units(total)}"
    else:
        pct = f"{_units(n)}"
        size = f"{_units(n)} / ?"

    stat = f"{size}  •  {_units(rate)}/s"
    if total and rate > 0:
        stat += f"  •  ETA {_format_duration((total - n) / rate)}"
    progress.update(task, pct=pct, stat=stat)


class rich_tqdm_class:
    """tqdm look-alike that proxies ``snapshot_download`` into one Rich bar.

    Only the "Reconstructing…" bar (which carries the real total) is rendered;
    every other internal bar ("Fetching…", "Downloading bytes…") is an inert
    stub so nothing leaks to the terminal. The class is intentionally *not* a
    ``huggingface_hub.utils.tqdm`` subclass — that takes the custom-class path
    in ``_create_progress_bar`` (no ``name``/``disable`` injected).
    """

    def __init__(self, desc="", total=None, initial=0, **kwargs):
        self.n = 0.0
        self._total = 0.0 if total is None else float(total or 0)
        self.format_dict = {"rate": None}  # read by hf's _set_aggregate_rate_postfix
        self._active = desc.startswith(_PRIMARY_DESC)
        if self._active:
            progress, task = _ensure()
            now = time.monotonic()
            if not _state["t0"]:
                _state["t0"] = now
            if self._total > 0:
                _state["total"] = self._total
                progress.update(task, total=self._total)
            if initial:
                self.n = float(initial)
                _state["n"] = self.n

    # -- the small surface huggingface_hub pokes at -------------------------
    @property
    def total(self) -> float:
        return self._total

    @total.setter
    def total(self, value: int | float | None) -> None:
        if not self._active:
            return
        self._total = float(value or 0)
        _state["total"] = self._total
        progress, task = _ensure()
        progress.update(task, total=self._total, pct=_state.get("pct", "0%"))

    def update(self, n: int | float = 1) -> None:
        if not self._active or not n:
            return
        progress, task = _ensure()
        self.n += float(n)
        _state["n"] = self.n
        progress.update(task, advance=float(n))
        _report(progress, task)

    def refresh(self) -> None:
        if self._active:
            progress, task = _ensure()
            _report(progress, task)
            progress.refresh()

    def close(self) -> None:
        pass

    def set_description(self, value: str, refresh: bool = False) -> None:
        self.set_description_str(value, refresh)

    def set_description_str(self, value: str, refresh: bool = False) -> None:
        if self._active:
            progress, task = _ensure()
            progress.update(task, description=value)

    def set_postfix_str(self, value: str, refresh: bool = False) -> None:
        pass

    def set_transfer_postfix_str(self, value: str, refresh: bool = False) -> None:
        pass

    def __enter__(self) -> "rich_tqdm_class":
        return self

    def __exit__(self, *exc) -> None:
        pass


def finish_download() -> None:
    """Stop the Rich widget after the download returns (idempotent)."""
    global _progress, _task_id
    with _lock:
        if _progress is not None:
            _progress.stop()
            _progress = None
            _task_id = None
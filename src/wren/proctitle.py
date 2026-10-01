"""Rename the running process to the model id in Activity Monitor / ps.

Mirrors ymlx's process-title hook (lib/sitecustomize.py): when a model is
being loaded, set the process name to the model id so Activity Monitor and
``ps`` show e.g. ``mlx-community/Qwen3.6-35B-A3B-4bit`` instead of a generic
``Python 3.1x``. Best-effort: if ``setproctitle`` isn't importable (some
slimmed environments omit it), we no-op and leave the process named as-is.
"""

from __future__ import annotations


def set_process_title(model_id: str) -> None:
    title = (model_id or "wren").strip()
    if not title:
        return
    try:
        from setproctitle import setproctitle
    except Exception:
        return
    setproctitle(title)

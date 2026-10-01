#!/usr/bin/env bash
#
# scripts/bootstrap.sh — build wren's LEAN virtualenv.
#
# wren is text-only (Qwen MoE). mlx-vlm ==0.4.4 declares `datasets` (→ pandas,
# ~40MB + pyarrow ~122MB) and `opencv-python` (→ cv2 ~119MB) as *unconditional*
# dependencies, but none of wren's imports touch them — they are ~280MB of dead
# weight for text inference. A plain `uv sync` therefore re-adds them, so wren
# installs mlx-vlm `--no-deps` and brings in only the core deps it actually
# imports. Idempotent: also prunes the bloat from a previously-`uv sync`'d env.
#
#   scripts/bootstrap.sh            # lean runtime env
#   scripts/bootstrap.sh WREN_DEV=0 # skip dev tooling
#
# Runs via:  ./.venv/bin/wren chat --cache-gb 6

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$ROOT/.venv}"
PY="${VENV%/}/bin/python"
DEV="${WREN_DEV:-1}"

say() { printf '==> %s\n' "$1"; }

if [ ! -x "$PY" ]; then
  say "Creating $VENV (python 3.12)…"
  uv venv "$VENV" --python 3.12
fi

# mlx-vlm itself, with NO transitive deps (avoid datasets/opencv/pandas/pyarrow).
say "Installing mlx-vlm==0.4.4 (--no-deps)…"
uv pip install --python "$PY" --no-deps "mlx-vlm==0.4.4"

# The core packages mlx-vlm's text path + wren actually import.
say "Installing core deps…"
uv pip install --python "$PY" \
  "mlx>=0.30" \
  "transformers" \
  "tqdm" \
  "Pillow" \
  "requests" \
  "mlx-lm>=0.31" \
  "fastapi" \
  "uvicorn" \
  "numpy" \
  "miniaudio" \
  "typer" \
  "rich" \
  "safetensors" \
  "huggingface-hub" \
  "ml-dtypes>=0.6" \
  "setproctitle"

# wren itself (console script `wren`) — no transitive deps pulled.
say "Installing wren (editable)…"
uv pip install --python "$PY" --no-deps -e "$ROOT"

if [ "$DEV" = "1" ]; then
  say "Installing dev tooling (pytest, ruff)…"
  uv pip install --python "$PY" pytest ruff
fi

# Prune the data-science / media bloat a normal `uv sync` would have pulled in.
say "Pruning unused heavy deps (datasets/pandas/pyarrow/opencv)…"
uv pip uninstall --python "$PY" -q \
  datasets pandas pyarrow opencv-python cv2 2>/dev/null || true

say "lean env ready: $(du -sh "$VENV" | awk '{print $1}') → run: ./.venv/bin/wren"
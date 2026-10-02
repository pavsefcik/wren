#!/usr/bin/env zsh
# wren-launcher.zsh — self-healing launcher for the wren MoE CLI/server.
#
# Runs `./.venv/bin/wren` DIRECTLY, never `uv run` / `uv sync` — the wren
# pyproject.toml explicitly forbids uv-managed installs because it declares
# ZERO runtime dependencies (the lean env is built solely by scripts/bootstrap.sh).
# A bare `uv run` therefore recreates an empty venv with no `typer` → the crash
# you hit. This shim detects that state and rebuilds via bootstrap.sh.
#
# Usage (in zsh):   wren [args...]
#   wren                          -> serve (default) on :8080
#   wren chat                     -> interactive REPL
#   wren --help                   -> help
#   wren serve --port 9000        -> any extra args forwarded

_WREN_ROOT="${0:A:h}"                     # this file's real directory (symlink-safe)
_WREN_VENV="$_WREN_ROOT/.venv"

wren() {
  # Rebuild only if the env is missing or is a bare `uv`-synced env (no typer).
  if [[ ! -x "$_WREN_VENV/bin/wren" ]] \
     || ! "$_WREN_VENV/bin/python" -c 'import typer' 2>/dev/null; then
    echo "==> rebuilding wren lean env…" >&2
    rm -rf "$_WREN_VENV"
    "$_WREN_ROOT/scripts/bootstrap.sh" || {
      echo "==> wren env build FAILED. Fix manually:" >&2
      echo "    $_WREN_ROOT/scripts/bootstrap.sh" >&2
      return 1
    }
  fi

  builtin cd "$_WREN_ROOT" || return 1
  "$_WREN_VENV/bin/wren" serve --cache-gb 6 --port 8080 "$@"
}
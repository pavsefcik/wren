# Changelog

All notable changes to wren are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/). Version numbers are kept in
sync across `VERSION`, `pyproject.toml`, and `src/wren/__init__.py` via
`scripts/version.py`.

## [0.2.0] - 2026-10-01

### Removed
- **Predictive expert prefetching.** Removed the learned expert predictor
  (`predictor.py`), the background prefetch worker pool (`prefetch.py`), the
  routing-trace recorder/training pipeline (`trace.py`, `collect_traces.py`,
  `train-predictor`), and all related CLI/server flags. Benchmarking showed
  prefetch never reduced cache misses (hit rate and misses were identical to
  the streaming baseline at every `--cache-gb` and lookahead) while adding
  roughly 7–12% decode overhead and a residual-stream-lookahead path that could
  crash the GPU. The win that *does* move decode speed is the expert-cache
  budget alone, so the tool now ships just the streaming cache.

### Added
- **Lean virtualenv build (`scripts/bootstrap.sh`).** Builds a ~400 MB `.venv`
  (mlx-vlm `--no-deps` + only the core deps), skipping the
  `datasets`/`pandas`/`pyarrow`/`opencv` tree (~280 MB) that mlx-vlm declares
  but text-only wren never imports. `pyproject.toml` is packaging metadata only.
- **Process title (`proctitle.py`).** The running process renames itself to the
  model id in Activity Monitor / `ps` (via `setproctitle`), like ymlx.
- **Versioning.** New `VERSION` source-of-truth, this `CHANGELOG.md`, and a
  `scripts/version.py` sync helper.
- **`scripts/bench.py`** now measures single-config decode speed / cache behavior
  for tuning `--cache-gb` (no prefetch comparison).

## [0.1.0] - 2026-08-28
- Initial release: streaming MoE expert cache for Qwen3.6-35B-A3B on Apple
  Silicon, CLI + OpenAI-compatible server, and (now removed) predictive
  prefetch.
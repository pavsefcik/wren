# Changelog

All notable changes to wren are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/). Version numbers are kept in
sync across `VERSION`, `pyproject.toml`, and `src/wren/__init__.py` via
`scripts/version.py`.

## [0.4.0] - 2026-10-02

### Added
- **`wren-launcher.zsh` — self-healing launcher.** Sourcing it defines a `wren`
  shell function that runs `./.venv/bin/wren` directly (never `uv run`/`uv sync`,
  which wren's dependency-free `pyproject.toml` forbids). If the lean env is
  missing or is a bare uv-built env without `typer`, it rebuilds via
  `scripts/bootstrap.sh` automatically and then launches.

### Changed
- **Smooth download progress.** First-run model fetches previously flooded
  stderr with huggingface_hub's raw tqdm bars ("Fetching N files" / "Downloading
  bytes" / "Reconstructing…"). `resolve_model_path` now hands `snapshot_download`
  a custom `tqdm_class` (`engine/download.py`) that swallows the native output
  and renders a single Rich widget — spinner plus live size, percent, speed, and
  ETA computed from the reported byte counts (correct from the first frame, and
  throttled to ~8 redraws/sec for smoothness). `wren serve` also downloads the
  weights *before* entering its `Loading…` status spinner, so the two live
  widgets no longer fight for the terminal.

## [0.3.1] - 2026-10-01

### Changed
- **Deterministic start/exit framing for `wren serve`.** On launch the screen
  is cleared so the WREN banner appears at the top of the terminal. On exit it
  wipes itself and leaves a short note (`wren stopped · served N req · ran …`)
  only. This avoids Rich Live's bottom-edge off-by-one that could leave a
  stray/duplicated traffic-box border under uvicorn's Ctrl+C shutdown; the
  live traffic box is now erased on stop rather than re-printed, and the
  idle placeholder renders dimmed without leaking markup text.

## [0.3.0] - 2026-10-01

### Added
- **Polished `wren serve` output.** The server now shows a bounded-resource
  banner (model, experts/layers, expert bytes on disk, streaming-cache budget,
  endpoint) after loading, replaces the model-loading wait with a spinner, and
  silences uvicorn's startup/access log noise. A live **traffic box** grows as
  requests arrive (last 20, then scrolls), reporting each request's timestamp,
  full-stream latency, token count, and throughput — or the status code and
  reason on errors. The box only tracks inference (`POST /v1/chat/completions`);
  health/model-listing routes are ignored. Loader INFO and mlx-lm's raw-stdout
  "maximum recommended size" warnings are muted, keeping the display clean.
  Controlled entirely from rich; no new dependencies.

## [0.2.0] - 2026-10-

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
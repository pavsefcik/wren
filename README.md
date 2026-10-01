# WREN

a Python CLI + OpenAI-compatible server that runs
**Qwen3.6-35B-A3B** (a 35B-parameter Mixture-of-Experts, 3B active/token) on a memory-constrained
Apple Silicon Mac by streaming only the experts the router actually picks, from SSD.

Inspired by [TurboFieldfare](https://github.com/drumih/turbo-fieldfare): WREN keeps the shared core
resident and streams the routed experts into a bounded LFU RAM cache with ``os.pread`` partial reads,
so a ~19 GB checkpoint runs in a few GB of RAM.

## How it works

The 4-bit checkpoint is ~18–20 GB, but only ~3 GB of parameters are active per token. WREN keeps the
shared core resident and streams the routed experts:

| Tier | Contents | Resident |
|---|---|---|
| Resident core | embeddings, attention, norms, router, shared expert, DeltaNet state | ~2–3 GB |
| Expert cache | recently-used window of routed experts (`switch_mlp`) | bounded (default 6 GB) |
| Disk | the full `[256 experts × 40 layers]` expert table | ~17 GB on SSD |

- **Lazy load** — `mlx_vlm.load(lazy=True)` so the full expert table never materializes.
- **Expert index** — parse the safetensors header once; read a single expert's bytes with `os.pread`.
- **Bounded cache** — LFU eviction with a byte budget; bigger budget = more experts resident,
  fewer SSD reads.

Measured on an 18 GB M3 Pro (4-bit): ~9–12 tok/s steady-state, ~5.5–9.5 GB peak, ~70–92% expert-cache
hit rate as `--cache-gb` goes 4→8.

### Why no prefetch

An earlier version shipped a learned expert-predictor plus a residual-stream *lookahead* that ran
future-layer routers to pre-read experts. It was removed after benchmarking: on a fast Apple SSD the
on-demand `os.pread` miss stall is short (~µs–200 µs), so prefetching never reduced cache misses
(hit rate and misses were identical to the streaming baseline) while adding overhead and a GPU-timeout
crash path. The LFU expert cache already captures local expert reuse; the lever that actually moves
decode speed is simply the `--cache-gb` budget.

## Requirements

- Apple Silicon Mac (Apple GPU; MLX needs Metal)
- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- ~20 GB free disk for the model

## Install

```shell
git clone https://github.com/pavsefcik/wren && cd wren
scripts/bootstrap.sh            # builds a LEAN .venv (≈400 MB)
```

The lean env installs `mlx-vlm` with `--no-deps` and skips the
`datasets`/`pandas`/`pyarrow`/`opencv` packages it declares but wren's
text-only path never imports (~280 MB of bloat). Run everything through
`./.venv/bin/wren`.

`pyproject.toml` is packaging metadata only — do **not** use `uv sync`/`uv run`
to manage this env. Because `mlx-vlm` is only installable via a package manager
together with its media/data-science tree, a `uv` sync would re-add the bloat
(or strip the bootstrap packages). `scripts/bootstrap.sh` is the sole way to
build the env. Optionally pass `WREN_DEV=0` to skip pytest/ruff.

The first run downloads `mlx-community/Qwen3.6-35B-A3B-4bit` (~19 GB) into the Hugging Face cache.

## Usage

### Chat REPL

```shell
./.venv/bin/wren chat --cache-gb 6
```

### One-shot generation

```shell
./.venv/bin/wren run "Name three planets in our solar system." --max-tokens 64
```

### OpenAI-compatible server (loopback)

```shell
./.venv/bin/wren serve --cache-gb 6 --port 8080
```

```shell
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"stream":true}'
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--cache-gb` | 6.0 | Expert cache budget (GiB). Higher = more experts resident, fewer SSD reads. |
| `--max-tokens` | 1024 | Max generated tokens. |
| `--temperature` | 0.2 | Sampling temperature (0 = greedy). |
| `--enable-thinking` | off | Emit Qwen3.6 reasoning before the answer. |

## Architecture

```
src/wren/
  cli.py                 # typer CLI: chat + run + serve
  server.py              # FastAPI /v1/chat/completions (streaming + JSON)
  engine/
    engine.py            # load_engine, generate, stream, TextProcessor
    expert_store.py      # safetensors header parse + os.pread partial reads
    expert_cache.py      # bounded LFU cache
    moe.py               # patched MoE forward (gather_qmm over gathered experts)
```

## Current status

Working v1: streaming expert-cache inference (runs the 35B MoE in a few GB of RAM), a CLI +
server, and a `scripts/bench.py` benchmark for tuning `--cache-gb`. The prefetch/predictor tier was
removed after benchmarking showed it never reduced cache misses and only added overhead.

## License

MIT. Model weights are governed by their own license (`Qwen3.6-35B-A3B` is Apache-2.0).

# WREN

a Python CLI + OpenAI-compatible server that runs
**Qwen3.6-35B-A3B** (a 35B-parameter Mixture-of-Experts, 3B active/token) on a memory-constrained
Apple Silicon Mac by streaming only the experts the router actually picks, from SSD.

Inspired by [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), but with a twist: instead
of loading experts cold on every miss, WREN **predicts** which experts are likely next and prefetches
them into a bounded RAM cache so decode doesn't stall on SSD reads.

## How it works

The 4-bit checkpoint is ~18–20 GB, but only ~3 GB of parameters are active per token. WREN keeps the
shared core resident and streams the routed experts:

| Tier | Contents | Resident |
|---|---|---|
| Resident core | embeddings, attention, norms, router, shared expert, DeltaNet state | ~2–3 GB |
| Expert cache | predicted window of routed experts (`switch_mlp`) | bounded (default 6 GB) |
| Disk | the full `[256 experts × 40 layers]` expert table | ~17 GB on SSD |

- **Lazy load** — `mlx_vlm.load(lazy=True)` so the full expert table never materializes.
- **Expert index** — parse the safetensors header once; read a single expert's bytes with `os.pread`.
- **Bounded cache** — LFU eviction with a byte budget.
- **Predictive prefetch** — a background thread pool pre-reads experts the router is likely to need
  next, hiding SSD latency behind GPU compute. Two predictor tiers ship:

  1. **Residual-stream lookahead** (no training) — runs the *real* next-layer router on the current
     hidden state (`h_{L+1} ≈ h_L`) to forecast the next layer's experts. Predicts the next layer's
     top-8 experts with **~80% recovery** (vs ~3% random).
  2. **Learned MLP** — a small 256→256→256 network trained on routing traces
     (`wren train-predictor`) that maps layer-`L` router logits to layer-`L+1` expert scores.

Measured on an 18 GB M3 Pro (4-bit): ~9 tok/s steady-state, ~7.5 GB peak, ~80% expert-cache hit rate.

### Honest evaluation of prefetch

On this hardware the story is nuanced and worth reading before enabling `--prefetch`:

- The **LFU cache alone** already captures expert temporal locality well: a 6 GB cache reaches ~81%
  hit rate and ~9 tok/s. A 2 GB cache drops to ~47% hit rate and ~7.5 tok/s.
- **Residual-stream lookahead** is a strong *predictor* (~80% next-layer top-8 recovery, vs 5% for
  the logits→logits MLP, which is weak because layer-`L` logits don't contain the MoE output that
  drives layer-`L+1` routing), but on a fast Apple SSD the prefetch I/O contends with the on-demand
  reads and the miss stall is already only ~200 µs/expert, so it currently does **not** improve
  tok/s on this machine.
- Prediction becomes valuable when storage is slower relative to compute, or when the cache is much
  smaller than the working set. The infrastructure is in place; the lever to pull next is a
  *cross-token* (temporal) predictor with a longer horizon rather than a 1–3 layer look-ahead.

## Requirements

- Apple Silicon Mac (Apple GPU; MLX needs Metal)
- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- ~20 GB free disk for the model

## Install

```shell
git clone https://github.com/pavsefcik/wren && cd wren
uv sync
```

The first run downloads `mlx-community/Qwen3.6-35B-A3B-4bit` (~19 GB) into the Hugging Face cache.

## Usage

### Chat REPL

```shell
uv run wren chat --cache-gb 6
```

### One-shot generation

```shell
uv run wren run "Name three planets in our solar system." --max-tokens 64
```

### OpenAI-compatible server (loopback)

```shell
uv run wren serve --cache-gb 6 --port 8080
```

```shell
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"stream":true}'
```

### Training the learned predictor

Record routing traces while you use the model, then train a small cross-layer expert predictor:

```shell
uv run wren run "..." --record-trace traces.jsonl
uv run wren train-predictor --traces traces.jsonl --output predictor.npz
uv run wren chat --predictor predictor.npz --prefetch
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--cache-gb` | 6.0 | Expert cache budget (GiB). Higher = more experts resident, fewer SSD reads. |
| `--prefetch` | off | Enable predictive prefetch (experimental on fast SSD). |
| `--prefetch-top-k` | 16 | Top-K experts per layer to prefetch. |
| `--prefetch-lookahead` | 0 | Number of layers to predict ahead (residual-stream lookahead). |
| `--predictor` | none | Path to a trained learned predictor (`.npz`). |
| `--record-trace` | none | File to append routing traces to during generation. |
| `--max-tokens` | 1024 | Max generated tokens. |
| `--temperature` | 0.2 | Sampling temperature (0 = greedy). |
| `--enable-thinking` | off | Emit Qwen3.6 reasoning before the answer. |

## Architecture

```
src/wren/
  cli.py                 # typer CLI: chat + run + serve + train-predictor
  server.py              # FastAPI /v1/chat/completions (streaming + JSON)
  engine/
    engine.py            # load_engine, generate, stream, TextProcessor
    expert_store.py      # safetensors header parse + os.pread partial reads
    expert_cache.py      # bounded LFU cache
    prefetch.py          # background prefetch worker pool
    moe.py               # patched MoE forward (gather_qmm over gathered experts)
    predictor.py         # learned cross-layer expert predictor (train + inference)
    trace.py             # routing-trace recorder
```

## Current status

Working v1: correct on-demand expert streaming, bounded memory, predictive prefetch (residual-stream
lookahead and a learned MLP), CLI + server, plus a trace→train→evaluate pipeline. See the honest
evaluation note above: the streaming + cache is the real win today; the predictor tier is built and
accurate but does not yet translate to throughput on fast local SSDs.

## License

MIT. Model weights are governed by their own license (`Qwen3.6-35B-A3B` is Apache-2.0).

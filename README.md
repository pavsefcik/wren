# LOKI

**LO**kale **K**ünstliche **I**ntelligenz — a Python CLI + OpenAI-compatible server that runs
**Qwen3.6-35B-A3B** (a 35B-parameter Mixture-of-Experts, 3B active/token) on a memory-constrained
Apple Silicon Mac by streaming only the experts the router actually picks, from SSD.

Inspired by [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), but with a twist: instead
of loading experts cold on every miss, LOKI **predicts** which experts are likely next and prefetches
them into a bounded RAM cache so decode doesn't stall on SSD reads.

## How it works

The 4-bit checkpoint is ~18–20 GB, but only ~3 GB of parameters are active per token. LOKI keeps the
shared core resident and streams the routed experts:

| Tier | Contents | Resident |
|---|---|---|
| Resident core | embeddings, attention, norms, router, shared expert, DeltaNet state | ~2–3 GB |
| Expert cache | predicted window of routed experts (`switch_mlp`) | bounded (default 6 GB) |
| Disk | the full `[256 experts × 40 layers]` expert table | ~17 GB on SSD |

- **Lazy load** — `mlx_vlm.load(lazy=True)` so the full expert table never materializes.
- **Expert index** — parse the safetensors header once; read a single expert's bytes with `os.pread`.
- **Bounded cache** — LFU eviction with a byte budget.
- **Predictive prefetch** — a background thread pre-reads experts the router is likely to need next,
  hiding SSD latency behind GPU compute. Ships with a router-logit heuristic plus a **learned
  cross-layer predictor** trained on routing traces (`loki train-predictor`).

Measured on an 18 GB M3 Pro (4-bit): ~9 tok/s steady-state, ~7.5 GB peak, ~80% expert-cache hit rate.

## Requirements

- Apple Silicon Mac (Apple GPU; MLX needs Metal)
- Python 3.12+, [uv](https://docs.astral.sh/uv/)
- ~20 GB free disk for the model

## Install

```shell
git clone https://github.com/pavsefcik/loki && cd loki
uv sync
```

The first run downloads `mlx-community/Qwen3.6-35B-A3B-4bit` (~19 GB) into the Hugging Face cache.

## Usage

### Chat REPL

```shell
uv run loki chat --cache-gb 6
```

### One-shot generation

```shell
uv run loki run "Name three planets in our solar system." --max-tokens 64
```

### OpenAI-compatible server (loopback)

```shell
uv run loki serve --cache-gb 6 --port 8080
```

```shell
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello!"}],"max_tokens":64,"stream":true}'
```

### Training the learned predictor

Record routing traces while you use the model, then train a small cross-layer expert predictor:

```shell
uv run loki run "..." --record-trace traces.jsonl
uv run loki train-predictor --traces traces.jsonl --output predictor.npz
uv run loki chat --predictor predictor.npz --prefetch
```

### Key options

| Flag | Default | Meaning |
|---|---|---|
| `--cache-gb` | 6.0 | Expert cache budget (GiB). Higher = more experts resident, fewer SSD reads. |
| `--prefetch` | off | Enable predictive prefetch. |
| `--prefetch-top-k` | 16 | Top-K experts per layer to prefetch. |
| `--prefetch-lookahead` | 0 | Prefetch the current layer's top-K for the *next* layer. |
| `--predictor` | none | Path to a trained learned predictor (`.npz`). |
| `--record-trace` | none | File to append routing traces to during generation. |
| `--max-tokens` | 1024 | Max generated tokens. |
| `--temperature` | 0.2 | Sampling temperature (0 = greedy). |
| `--enable-thinking` | off | Emit Qwen3.6 reasoning before the answer. |

## Architecture

```
src/loki/
  cli.py                 # typer CLI: chat + run + serve + train-predictor
  server.py              # FastAPI /v1/chat/completions (streaming + JSON)
  engine/
    engine.py            # load_engine, generate, stream, TextProcessor
    expert_store.py      # safetensors header parse + os.pread partial reads
    expert_cache.py      # bounded LFU cache + PrefetchPolicy
    prefetch.py          # background prefetch worker
    moe.py               # patched MoE forward (gather_qmm over gathered experts)
    predictor.py         # learned cross-layer expert predictor (train + inference)
    trace.py             # routing-trace recorder
```

## Current status

Working v1: correct on-demand expert streaming, bounded memory, predictive prefetch (heuristic and
learned), CLI + server. Natural next steps: co-activation profiles from offline traces, a
learned cross-token (temporal) predictor, and finer SSD-layout repacking.

## License

MIT. Model weights are governed by their own license (`Qwen3.6-35B-A3B` is Apache-2.0).

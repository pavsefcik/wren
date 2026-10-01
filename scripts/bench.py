"""Benchmark warm decode speed and MoE expert-cache behavior for WREN.

Measures decode tok/s, expert cache hit rate, evictions, resident experts and
peak memory for a given RAM budget / cache size, averaged over several fresh
prompts. Built to tune ``--cache-gb`` and validate the streaming-expert
proposition (a 19 GB MoE running in a small fraction of that memory).

Quick start
-----------
    ./.venv/bin/python scripts/bench.py --cache-gb 6 --repeats 3
    ./.venv/bin/python scripts/bench.py --cache-gb 10 --json   # machine-readable

Methodology
-----------
* temperature=0 and thinking off, so generation is deterministic.
* A warmup generation (--warmup-tokens) populates the expert cache and JIT
  compiles the graph before anything is measured.
* --fresh (default) rotates a distinct prompt each epoch so the cache keeps
  missing and hit_rate/evictions are measured under genuine cold traffic rather
  than replay-induced ~100% hits.
* Cache counters are snapshotted per-epoch and reported as deltas, so what gets
  reported is steady-state behavior, not cumulative noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List

import mlx.core as mx

from wren.engine import EngineConfig, load_engine

# A pool of varied factual prompts so each epoch exercises different router
# activity, forcing the expert cache to miss and evict under a bounded budget.
PROMPT_POOL = [
    "Write a short paragraph about why rivers flow to the sea. Keep it factual.",
    "Explain how a solar eclipse happens, in a few accurate sentences.",
    "Describe the difference between stalactites and stalagmites.",
    "Summarize how bees pollinate flowers and why it matters.",
    "What causes the four seasons? Answer in plain, correct terms.",
    "Explain what a magnet is and how it attracts metal.",
    "Describe the water cycle in a concise, accurate way.",
    "Why does the sky appear blue during the day?",
    "Briefly explain how submarines stay underwater without sinking.",
    "What is photosynthesis? Give a short factual summary.",
]


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="mlx-community/Qwen3.6-35B-A3B-4bit")
    p.add_argument("--cache-gb", type=float, default=6.0, help="Expert cache RAM budget (GiB).")
    p.add_argument("--max-tokens", type=int, default=64, help="Output tokens per generation.")
    p.add_argument("--warmup-tokens", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3, help="Epochs to average over.")
    p.add_argument("--fresh", action="store_true",
                   help="Use a distinct prompt each epoch so the cache keeps missing.")
    p.add_argument("--no-fresh", dest="fresh", action="store_false")
    p.add_argument("--json", action="store_true", help="Print machine-readable JSON results.")
    p.set_defaults(fresh=True)
    return p.parse_args(argv)


def _cfg(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model_id=args.model,
        cache_gb=args.cache_gb,
        max_tokens=args.max_tokens,
        temperature=0.0,
        enable_thinking=False,
    )


def _prompt(eng, text: str) -> str:
    return eng.chat_prompt([{"role": "user", "content": [{"type": "text", "text": text}]}])


def _warmup(eng, text: str, tokens: int) -> None:
    from mlx_vlm import stream_generate

    prompt = _prompt(eng, text)
    for _ in stream_generate(
        eng.model, eng.processor, prompt, max_tokens=tokens, temperature=0.0
    ):
        pass


def _measure(eng, text: str, max_tokens: int) -> Dict[str, float]:
    """Generate one response; return per-epoch metrics from cache-counter deltas."""
    from mlx_vlm import stream_generate

    prompt = _prompt(eng, text)
    before = eng.stats()
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    last = None
    for r in stream_generate(
        eng.model, eng.processor, prompt, max_tokens=max_tokens, temperature=0.0
    ):
        last = r
    dt = time.perf_counter() - t0
    after = eng.stats()

    d_miss = after["misses"] - before["misses"]
    d_hit = after["hits"] - before["hits"]
    denom = d_hit + d_miss
    return {
        "prompt_tps": last.prompt_tps,
        "gen_tps": last.generation_tps,
        "wall_s": dt,
        "peak_gb": after["peak_memory_bytes"] / 2**30,
        "hit_rate": (d_hit / denom) if denom else 0.0,
        "misses": float(d_miss),
        "evictions": float(after["evictions"] - before["evictions"]),
        "resident_experts": float(after["resident_experts"]),
    }


def run_case(args: argparse.Namespace) -> Dict[str, float]:
    cfg = _cfg(args)
    print(f"--- running cache_gb={args.cache_gb} ---")
    eng = load_engine(cfg)
    try:
        _warmup(eng, PROMPT_POOL[0], args.warmup_tokens)
        print(f"warmup cache: {eng.stats()}")

        epochs: List[Dict[str, float]] = []
        for k in range(args.repeats):
            prompt = PROMPT_POOL[0] if not args.fresh else PROMPT_POOL[k % len(PROMPT_POOL)]
            m = _measure(eng, prompt, args.max_tokens)
            epochs.append(m)
            print(f"  epoch {k + 1}: gen_tps={m['gen_tps']:.1f} "
                  f"hit_rate={m['hit_rate']*100:4.1f}% misses={int(m['misses'])} "
                  f"evict={int(m['evictions'])}")

        def mean(key: str) -> float:
            return sum(e[key] for e in epochs) / len(epochs)

        def std(key: str) -> float:
            mu = mean(key)
            return (sum((e[key] - mu) ** 2 for e in epochs) / len(epochs)) ** 0.5

        return {
            "gen_tps": mean("gen_tps"),
            "gen_tps_std": std("gen_tps"),
            "wall_s": mean("wall_s"),
            "hit_rate": mean("hit_rate"),
            "misses": mean("misses"),
            "evictions": mean("evictions"),
            "peak_gb": max(e["peak_gb"] for e in epochs),
            "resident_experts": mean("resident_experts"),
        }
    finally:
        eng.close()


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    res = run_case(args)
    print("\n=== RESULT ===")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    if args.json:
        import json

        print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

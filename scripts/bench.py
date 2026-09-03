"""Benchmark warm decode speed and cache hit rate for WREN."""

import sys
import time

import mlx.core as mx

from wren.engine import EngineConfig, load_engine


def main():
    prefetch = "--prefetch" in sys.argv
    cache_gb = (
        float(sys.argv[sys.argv.index("--cache-gb") + 1]) if "--cache-gb" in sys.argv else 6.0
    )
    lookahead = (
        int(sys.argv[sys.argv.index("--lookahead") + 1]) if "--lookahead" in sys.argv else 0
    )
    max_tokens = 128

    cfg = EngineConfig(
        model_id="mlx-community/Qwen3.6-35B-A3B-4bit",
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=16,
        prefetch_lookahead=lookahead,
        max_tokens=max_tokens,
        temperature=0.0,
        enable_thinking=False,
    )
    print(f"prefetch={prefetch} cache_gb={cache_gb} lookahead={lookahead}")
    eng = load_engine(cfg)

    prompt = eng.chat_prompt(
        [{"role": "user", "content": [{"type": "text", "text": "Say hello."}]}]
    )

    # warmup (JIT compile + cold expert fill)
    from mlx_vlm import stream_generate

    print("warmup...")
    for r in stream_generate(eng.model, eng.processor, prompt, max_tokens=8, temperature=0.0):
        pass

    print("warmup stats:", eng.stats())

    # benchmark: fresh conversation, steady decode
    bench_prompt = eng.chat_prompt(
        [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Write a short paragraph about why rivers flow to the sea. "
                            "Keep it factual and concise."
                        ),
                    }
                ],
            }
        ]
    )

    mx.reset_peak_memory()
    t0 = time.perf_counter()
    last = None
    for r in stream_generate(
        eng.model, eng.processor, bench_prompt, max_tokens=max_tokens, temperature=0.0
    ):
        last = r
    dt = time.perf_counter() - t0

    print("=== RESULT ===")
    print("generation_tokens:", last.generation_tokens)
    print("generation_tps:", round(last.generation_tps, 2))
    print("prompt_tps:", round(last.prompt_tps, 2))
    print("peak_memory_gb:", round(mx.get_peak_memory() / 2**30, 2))
    print("wall_seconds:", round(dt, 2))
    print("cache:", eng.stats())


if __name__ == "__main__":
    main()

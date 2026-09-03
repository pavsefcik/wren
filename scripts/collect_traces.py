"""Collect routing traces for training the learned predictor.

Runs a set of diverse prompts through the model with ``--record-trace``
semantics and appends every ``L -> L+1`` layer transition to the trace file.
"""

import sys

from wren.engine import EngineConfig, generate, load_engine

PROMPTS = [
    "Explain how a computer compiles source code into machine code.",
    "Write a short story about a lighthouse keeper and a storm.",
    "Describe the steps of photosynthesis in simple terms.",
    "What are the main differences between TCP and UDP?",
    "Give me a recipe for sourdough bread.",
    "Summarize the plot of Hamlet in three sentences.",
    "Explain the theory of evolution for a high school student.",
    "What causes earthquakes and how are they measured?",
    "Compare and contrast capitalism and socialism.",
    "Describe how neural networks are trained.",
    "Explain the water cycle and why it matters.",
    "What is the history of the printing press?",
]


def main():
    trace_path = sys.argv[1] if len(sys.argv) > 1 else "traces.bin"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 96

    cfg = EngineConfig(
        cache_gb=6.0,
        prefetch=False,
        record_trace=trace_path,
        max_tokens=max_tokens,
        temperature=0.7,
        enable_thinking=False,
    )
    engine = load_engine(cfg)

    for i, p in enumerate(PROMPTS):
        prompt = engine.chat_prompt(
            [{"role": "user", "content": [{"type": "text", "text": p}]}]
        )
        text = generate(engine, prompt)
        n = engine.recorder.count
        print(f"[{i + 1}/{len(PROMPTS)}] {len(text)} chars, cumulative traces: {n}")

    engine.close()
    print(f"done -> {trace_path}")


if __name__ == "__main__":
    main()

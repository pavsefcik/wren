"""LOKI command-line interface: chat REPL, one-shot generation, server, training."""

from __future__ import annotations

import time

import typer
from rich.console import Console

from .engine import Engine, EngineConfig, generate, load_engine, stream
from .server import serve

app = typer.Typer(
    name="loki",
    help="Run Qwen3.6-35B-A3B MoE with predictive expert prefetching on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()

_COMMON = {
    "model": typer.Option("mlx-community/Qwen3.6-35B-A3B-4bit", "--model"),
    "cache_gb": typer.Option(6.0, "--cache-gb"),
    "prefetch": typer.Option(False, "--prefetch", help="Enable predictive prefetch."),
    "prefetch_top_k": typer.Option(16, "--prefetch-top-k"),
    "prefetch_lookahead": typer.Option(0, "--prefetch-lookahead"),
    "predictor": typer.Option(None, "--predictor", help="Path to a trained predictor .npz."),
    "record_trace": typer.Option(None, "--record-trace", help="File to append routing traces to."),
}


def _cfg(
    model, cache_gb, prefetch, prefetch_top_k, prefetch_lookahead, predictor, record_trace, **kw
) -> EngineConfig:
    return EngineConfig(
        model_id=model,
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=prefetch_top_k,
        prefetch_lookahead=prefetch_lookahead,
        predictor=predictor,
        record_trace=record_trace,
        **kw,
    )


@app.command()
def chat(
    model: str = _COMMON["model"],
    cache_gb: float = _COMMON["cache_gb"],
    prefetch: bool = _COMMON["prefetch"],
    prefetch_top_k: int = _COMMON["prefetch_top_k"],
    prefetch_lookahead: int = _COMMON["prefetch_lookahead"],
    predictor: str = _COMMON["predictor"],
    record_trace: str = _COMMON["record_trace"],
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
    stats: bool = typer.Option(False, "--stats", help="Print expert-cache stats after each turn."),
):
    """Interactive multi-turn chat."""
    cfg = _cfg(
        model, cache_gb, prefetch, prefetch_top_k, prefetch_lookahead, predictor,
        record_trace, max_tokens=max_tokens, temperature=temperature,
    )
    engine = _load(cfg)

    history = []
    console.print(
        "[bold]LOKI[/bold] — type [cyan]/exit[/cyan] to quit, [cyan]/clear[/cyan] to reset."
    )
    try:
        while True:
            try:
                user = console.input("[bold green]You>[/bold green] ")
            except (EOFError, KeyboardInterrupt):
                console.print("\nBye.")
                break

            user = user.strip()
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/clear":
                history.clear()
                console.print("History cleared.")
                continue

            history.append({"role": "user", "content": [{"type": "text", "text": user}]})
            prompt = engine.chat_prompt(history)

            t0 = time.monotonic()
            console.print("[bold blue]Loki>[/bold blue] ", end="")
            text = ""
            for chunk in stream(engine, prompt):
                text += chunk
                console.print(chunk, end="")
            console.print()
            dt = time.monotonic() - t0

            history.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            n_tok = len(engine.processor.tokenizer.encode(text)) if text else 0
            console.print(f"[dim]({n_tok} tok in {dt:.1f}s, {n_tok/dt:.1f} tok/s)[/dim]")
            if stats:
                console.print(f"[dim]{engine.stats()}[/dim]")
    finally:
        engine.close()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt text."),
    model: str = _COMMON["model"],
    cache_gb: float = _COMMON["cache_gb"],
    prefetch: bool = _COMMON["prefetch"],
    prefetch_top_k: int = _COMMON["prefetch_top_k"],
    prefetch_lookahead: int = _COMMON["prefetch_lookahead"],
    predictor: str = _COMMON["predictor"],
    record_trace: str = _COMMON["record_trace"],
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
    show_stats: bool = typer.Option(False, "--stats"),
):
    """One-shot generation from a single prompt."""
    cfg = _cfg(
        model, cache_gb, prefetch, prefetch_top_k, prefetch_lookahead, predictor,
        record_trace, max_tokens=max_tokens, temperature=temperature,
    )
    engine = _load(cfg)
    try:
        prompt = engine.chat_prompt(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )

        t0 = time.monotonic()
        text = generate(engine, prompt)
        dt = time.monotonic() - t0

        console.print(text)
        if show_stats:
            console.print(f"[dim]{engine.stats()}[/dim]")
            console.print(f"[dim]elapsed {dt:.1f}s[/dim]")
    finally:
        engine.close()


@app.command("serve")
def serve_command(
    model: str = _COMMON["model"],
    cache_gb: float = _COMMON["cache_gb"],
    prefetch: bool = _COMMON["prefetch"],
    prefetch_top_k: int = _COMMON["prefetch_top_k"],
    prefetch_lookahead: int = _COMMON["prefetch_lookahead"],
    predictor: str = _COMMON["predictor"],
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8080, "--port"),
):
    """Serve the model on an OpenAI-compatible /v1 endpoint."""
    serve(
        model=model,
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=prefetch_top_k,
        prefetch_lookahead=prefetch_lookahead,
        predictor=predictor,
        host=host,
        port=port,
    )


@app.command("train-predictor")
def train_predictor(
    traces: str = typer.Option(..., "--traces", help="Trace file from --record-trace."),
    output: str = typer.Option("predictor.npz", "--output"),
    hidden: int = typer.Option(256, "--hidden"),
    epochs: int = typer.Option(20, "--epochs"),
    lr: float = typer.Option(3e-3, "--lr"),
):
    """Train the learned cross-layer expert predictor from routing traces."""
    from .engine.predictor import train_predictor as _train

    console.print(f"Training predictor on [cyan]{traces}[/cyan]...")
    result = _train(traces, output, hidden=hidden, epochs=epochs, lr=lr)
    console.print(f"Saved [cyan]{output}[/cyan]: {result}")


def _load(cfg: EngineConfig) -> Engine:
    console.print(f"[dim]Loading {cfg.model_id} (cache {cfg.cache_gb} GiB)...[/dim]")
    engine = load_engine(cfg)
    console.print(
        f"[dim]Loaded. {engine.store.num_layers} layers, "
        f"{engine.store.num_experts} experts/layer, "
        f"{engine.store.total_expert_bytes() / 2**30:.1f} GiB experts on disk.[/dim]"
    )
    return engine


if __name__ == "__main__":
    app()

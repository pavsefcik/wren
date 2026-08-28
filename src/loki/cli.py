"""LOKI command-line interface: chat REPL and one-shot generation."""

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


@app.command()
def chat(
    model: str = typer.Option("mlx-community/Qwen3.6-35B-A3B-4bit", "--model"),
    cache_gb: float = typer.Option(6.0, "--cache-gb", help="Expert cache budget in GiB."),
    prefetch: bool = typer.Option(False, "--prefetch", help="Enable predictive prefetch."),
    prefetch_top_k: int = typer.Option(16, "--prefetch-top-k"),
    prefetch_lookahead: int = typer.Option(0, "--prefetch-lookahead"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
    stats: bool = typer.Option(
        False, "--stats", help="Print expert-cache stats after each turn."
    ),
):
    """Interactive multi-turn chat."""
    cfg = EngineConfig(
        model_id=model,
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=prefetch_top_k,
        prefetch_lookahead=prefetch_lookahead,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    engine = _load(cfg)

    history = []
    console.print(
        "[bold]LOKI[/bold] — type [cyan]/exit[/cyan] to quit, [cyan]/clear[/cyan] to reset."
    )
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


@app.command()
def run(
    prompt: str = typer.Argument(..., help="Prompt text."),
    model: str = typer.Option("mlx-community/Qwen3.6-35B-A3B-4bit", "--model"),
    cache_gb: float = typer.Option(6.0, "--cache-gb"),
    prefetch: bool = typer.Option(False, "--prefetch"),
    prefetch_top_k: int = typer.Option(16, "--prefetch-top-k"),
    prefetch_lookahead: int = typer.Option(0, "--prefetch-lookahead"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
    show_stats: bool = typer.Option(False, "--stats"),
):
    """One-shot generation from a single prompt."""
    cfg = EngineConfig(
        model_id=model,
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=prefetch_top_k,
        prefetch_lookahead=prefetch_lookahead,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    engine = _load(cfg)

    prompt = engine.chat_prompt([{"role": "user", "content": [{"type": "text", "text": prompt}]}])

    t0 = time.monotonic()
    text = generate(engine, prompt)
    dt = time.monotonic() - t0

    console.print(text)
    if show_stats:
        console.print(f"[dim]{engine.stats()}[/dim]")
        console.print(f"[dim]elapsed {dt:.1f}s[/dim]")


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


@app.command("serve")
def serve_command(
    model: str = typer.Option("mlx-community/Qwen3.6-35B-A3B-4bit", "--model"),
    cache_gb: float = typer.Option(6.0, "--cache-gb"),
    prefetch: bool = typer.Option(False, "--prefetch"),
    prefetch_top_k: int = typer.Option(16, "--prefetch-top-k"),
    prefetch_lookahead: int = typer.Option(0, "--prefetch-lookahead"),
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
        host=host,
        port=port,
    )

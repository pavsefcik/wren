"""WREN command-line interface: chat REPL, one-shot generation, server, training."""

from __future__ import annotations

import time
from typing import Iterable, Iterator, Optional

import typer
from rich.console import Console
from rich.text import Text

from .engine import Engine, EngineConfig, generate, load_engine, stream
from .server import serve

app = typer.Typer(
    name="wren",
    help="Run Qwen3.6-35B-A3B MoE with predictive expert prefetching on Apple Silicon.",
    no_args_is_help=True,
)
console = Console()

# Qwen3.6 wraps its reasoning in the literal "<think>" / "</think>" tokens.
# They are added tokens (not special), so they decode as plain text and
# arrive in the streamed output like any other word. The chat template
# emits "<think>" as part of the prompt, so the stream is: reasoning,
# then the "</think>" marker, then the answer.
_THINK_START = "<think>"
_THINK_END = "</think>"


def _styled_chunks(
    chunks: Iterable[str], thinking: bool
) -> Iterator[tuple[str, Optional[str]]]:
    """Yield ``(text, style)`` pairs for streaming model output.

    With ``thinking`` enabled, everything before the model's ``</think>``
    marker is emitted with the ``dim`` style (gray) and the marker tokens
    themselves (``<think>`` and ``</think>``) are dropped. Without it,
    text passes through untouched.
    """
    style: Optional[str] = "dim" if thinking else None
    buf = ""
    started = not thinking

    for chunk in chunks:
        buf += chunk
        while buf:
            if style is None:
                yield buf, None
                buf = ""
                break

            if not started:
                # Drop a leading "<think>" marker (the chat template usually
                # already emits it, but the model may too).
                if _THINK_START.startswith(buf):
                    break  # partial marker: wait for the rest
                if buf.startswith(_THINK_START):
                    buf = buf[len(_THINK_START):]
                    if not buf:
                        break
                started = True
                continue

            idx = buf.find(_THINK_END)
            if idx >= 0:
                if idx:
                    yield buf[:idx], "dim"
                buf = buf[idx + len(_THINK_END):]
                style = None
                continue

            # No marker yet: don't split a trailing partial marker across
            # chunk boundaries.
            keep = next(
                (
                    k
                    for k in range(len(_THINK_END) - 1, 0, -1)
                    if buf.endswith(_THINK_END[:k])
                ),
                0,
            )
            if keep == 0:
                yield buf, "dim"
                buf = ""
            elif len(buf) > keep:
                yield buf[:-keep], "dim"
                buf = buf[-keep:]
            break

    if buf:
        yield buf, style


def _styled_final_text(text: str, thinking: bool) -> Text:
    """Style one-shot output: reasoning dimmed, marker tokens dropped."""
    if not thinking:
        return Text(text)
    if text.startswith(_THINK_START):
        text = text[len(_THINK_START):]
    idx = text.find(_THINK_END)
    if idx < 0:
        return Text(text, style="dim")
    out = Text()
    if idx:
        out.append(text[:idx], style="dim")
    out.append(text[idx + len(_THINK_END):])
    return out


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
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking",
        help=(
            "Emit Qwen3.6 reasoning before the answer (rendered dim; the "
            "'</think>' marker is hidden)."
        ),
    ),
    stats: bool = typer.Option(False, "--stats", help="Print expert-cache stats after each turn."),
):
    """Interactive multi-turn chat."""
    cfg = _cfg(
        model, cache_gb, prefetch, prefetch_top_k, prefetch_lookahead, predictor,
        record_trace, max_tokens=max_tokens, temperature=temperature,
        enable_thinking=enable_thinking,
    )
    engine = _load(cfg)

    history = []
    console.print(
        "[bold]WREN[/bold] — type [cyan]/exit[/cyan] to quit, [cyan]/clear[/cyan] to reset."
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
            console.print("[bold blue]Wren>[/bold blue] ", end="")
            text = ""
            for piece, style in _styled_chunks(
                stream(engine, prompt), cfg.enable_thinking
            ):
                text += piece
                console.print(piece, end="", style=style, markup=False)
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
    enable_thinking: bool = typer.Option(
        False,
        "--enable-thinking",
        help=(
            "Emit Qwen3.6 reasoning before the answer (rendered dim; the "
            "'</think>' marker is hidden)."
        ),
    ),
    show_stats: bool = typer.Option(False, "--stats"),
):
    """One-shot generation from a single prompt."""
    cfg = _cfg(
        model, cache_gb, prefetch, prefetch_top_k, prefetch_lookahead, predictor,
        record_trace, max_tokens=max_tokens, temperature=temperature,
        enable_thinking=enable_thinking,
    )
    engine = _load(cfg)
    try:
        prompt = engine.chat_prompt(
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        )

        t0 = time.monotonic()
        text = generate(engine, prompt)
        dt = time.monotonic() - t0

        console.print(_styled_final_text(text, cfg.enable_thinking))
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

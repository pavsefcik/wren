"""OpenAI-compatible loopback server for WREN."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any, List, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import Engine, EngineConfig, load_engine, stream


class ChatMessage(BaseModel):
    role: str
    # OpenAI clients send content as a plain string, a list of parts
    # (e.g. [{"type": "text", "text": "hi"}]), or null (tool-call messages).
    content: Optional[Union[str, List[Any]]] = None


class ChatRequest(BaseModel):
    model: str = "wren"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, alias="max_completion_tokens")
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = None
    stream: Optional[bool] = False
    enable_thinking: Optional[bool] = Field(
        default=None,
        description=(
            "Override thinking mode for this request. True emits Qwen3 reasoning, "
            "False skips it. Omit to use the engine default."
        ),
    )

    model_config = {"populate_by_name": True}


def _text_of(content) -> str:
    """Flatten an OpenAI content field (string, list of parts, or None) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def build_app(engine: Engine, model_name: str) -> FastAPI:
    api = FastAPI(title="WREN", version="0.1.0")

    @api.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @api.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        history = [
            {"role": m.role, "content": [{"type": "text", "text": _text_of(m.content)}]}
            for m in req.messages
        ]
        prompt = engine.chat_prompt(history, enable_thinking=req.enable_thinking)
        kwargs = {
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.enable_thinking is not None:
            kwargs["enable_thinking"] = req.enable_thinking
        resp_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            async def gen():
                yield "data: " + _chunk(resp_id, model_name, "", "delta") + "\n\n"
                for text in stream(engine, prompt, **kwargs):
                    yield "data: " + _chunk(resp_id, model_name, text, "delta") + "\n\n"
                yield "data: " + _chunk(resp_id, model_name, "", "stop") + "\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

        text = ""
        for chunk in stream(engine, prompt, **kwargs):
            text += chunk
        return {
            "id": resp_id,
            "object": "chat.completion",
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return api


def _chunk(resp_id: str, model: str, text: str, finish: str) -> str:
    import json

    delta = {"content": text} if finish == "delta" else {}
    return json.dumps(
        {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish if finish == "stop" else None,
                }
            ],
        }
    )


def _fmt_duration(secs: float) -> str:
    """Format a duration for the exit note (e.g. '1m 05s')."""
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    return f"{secs // 60}m {secs % 60:02d}s"


def _print_banner(console: Console, engine, cfg, host: str, port: int) -> None:
    """Render the static WREN header once the model is loaded."""
    from math import ceil

    store = engine.store
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold cyan")
    grid.add_column(overflow="fold")
    grid.add_row("model   ", cfg.model_id)
    grid.add_row(
        "experts ", f"[bold]{store.num_experts}[/bold] / layer × {store.num_layers} layers"
    )
    grid.add_row(
        "expert  ",
        f"[bold]{store.total_expert_bytes() / 2**30:.1f} GiB[/bold] on disk · "
        f"[bold]{ceil(cfg.cache_gb):d} GiB[/bold] streaming cache",
    )
    grid.add_row("endpoint", f"{host}:{port}  [dim]OpenAI /v1/chat/completions[/dim]")
    panel = Panel(
        grid,
        title="[bold]WREN[/bold]",
        title_align="left",
        subtitle="[dim]Ctrl+C to quit[/dim]",
        subtitle_align="right",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(panel)


class _TrafficMonitor:
    """A growing, bounded window of recent requests, redrawn in place."""

    def __init__(self, console: Console, engine, max_rows: int = 20):
        self.console = console
        self.tokenizer = engine.processor.tokenizer
        self.max_rows = max_rows
        self.rows: list[Text] = []
        self._live: Optional[Live] = None

    def start(self) -> None:
        # ``transient=True``: on exit the box is erased (clear-line sequences)
        # rather than re-printed. Left ``False``, Rich re-renders the final
        # frame on stop, and in a scrolled terminal that re-print lands a row
        # off and leaves a duplicated top border under the banner.
        self._live = Live(
            self._render(), console=self.console, refresh_per_second=8, transient=True
        )
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def _render(self) -> Panel:
        group = Group(*self.rows) if self.rows else Text("waiting for first request…", style="dim")
        return Panel(
            group,
            title=f"traffic ({len(self.rows)} req)",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def _record(self, row: Text) -> None:
        self.rows.append(row)
        if len(self.rows) > self.max_rows:
            self.rows.pop(0)
        if self._live is not None:
            self._live.update(self._render(), refresh=True)

    def add_stream(self, dt: float, text: str) -> None:
        n_tok = len(self.tokenizer.encode(text)) if text else 0
        rate = f"{n_tok / dt:.0f} tok/s" if dt > 0 else "—"
        stamp = datetime.now().strftime("%H:%M:%S")
        n_tok_str = f"{n_tok:,}"
        self._record(
            Text.assemble(
                (f"[{stamp}] ", "dim"),
                (f"· {dt:.1f}s · ", "dim"),
                (n_tok_str, "bold"),
                (f" tok · {rate}", "dim"),
            )
        )

    def add_error(self, dt: float, status: int, reason: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        detail = f" · {reason}" if reason else ""
        self._record(
            Text(
                f"[{stamp}] · {status}{detail}",
                style="bold red",
            )
        )


def _sse_text(full: str) -> str:
    """Concatenate the delta.content across all SSE chunks."""
    out: list[str] = []
    for line in full.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        try:
            content = obj["choices"][0]["delta"]["content"]
        except Exception:
            continue
        if content:
            out.append(content)
    return "".join(out)


def _json_text(full: str) -> str:
    try:
        return json.loads(full)["choices"][0]["message"]["content"]
    except Exception:
        return ""


def _error_reason(full: str) -> str:
    try:
        detail = json.loads(full).get("detail") or json.loads(full).get("message")
    except Exception:
        return ""
    if isinstance(detail, list):
        detail = "; ".join(
            str(d.get("msg", d)) for d in detail[:2] if isinstance(d, dict)
        )
    return str(detail) if detail else ""


# mlx-lm dumps a load-time model-size warning straight to stdout via bare
# ``print()`` (not the logging module), so logger levels can't silence it. We
# discard it by redirecting stdout while a request is being served.
_DEVNULL = open(os.devnull, "w")


def _traffic_middleware(app, monitor: _TrafficMonitor):
    """Raw ASGI middleware that measures each request to first/last byte."""

    async def traffic(scope, receive, send):
        if scope["type"] != "http":
            return await app(scope, receive, send)
        # Only inference calls show up in the traffic box; leave health/model
        # listing and any other routes untouched (no body buffering).
        if scope.get("method") != "POST" or not scope.get("path", "").endswith(
            "/chat/completions"
        ):
            return await app(scope, receive, send)
        started = time.monotonic()
        status = 500
        body: list[bytes] = []

        async def send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            elif message["type"] == "http.response.body" and message.get("body"):
                body.append(message["body"])
            await send(message)

        try:
            with contextlib.redirect_stdout(_DEVNULL):
                await app(scope, receive, send_wrapper)
        except Exception:
            monitor.add_error(time.monotonic() - started, 500, "exception")
            raise

        dt = time.monotonic() - started
        full = b"".join(body).decode("utf-8", "replace")
        if status >= 400:
            monitor.add_error(dt, status, _error_reason(full))
        elif "[DONE]" in full:
            monitor.add_stream(dt, _sse_text(full))
        else:
            monitor.add_stream(dt, _json_text(full))

    return traffic


def serve(
    model: str = "mlx-community/Qwen3.6-35B-A3B-4bit",
    cache_gb: float = 6.0,
    host: str = "127.0.0.1",
    port: int = 8080,
):
    """Serve the model on an OpenAI-compatible /v1 endpoint."""
    from .proctitle import set_process_title

    set_process_title(model)
    cfg = EngineConfig(
        model_id=model,
        cache_gb=cache_gb,
    )
    console = Console()
    # Fresh top-of-terminal: clear whatever preceded us before drawing.
    console.clear()
    started = time.monotonic()

    # Mute everything the model/uvicorn would otherwise spew for the whole run:
    # the loader's INFO chatter ("Local model ready") *and* the per-request
    # WARNING from mlx-lm ("model close to maximum recommended size"). The live
    # traffic box is deliberately the only activity on screen.
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(logging.ERROR)
    engine = None
    monitor = None
    try:
        spinner = f"[bold]Loading {cfg.model_id}[/bold] … [dim]cache {cfg.cache_gb:.0f} GiB[/dim]"
        with console.status(spinner):
            engine = load_engine(cfg)

        _print_banner(console, engine, cfg, host, port)
        monitor = _TrafficMonitor(console, engine, max_rows=20)
        monitor.start()

        api = _traffic_middleware(build_app(engine, model), monitor)
        # uvicorn's INFO startup lines + access logs are (also) turned off below.
        uvicorn.run(api, host=host, port=port, log_level="warning")
    finally:
        if monitor is not None:
            monitor.stop()
        if engine is not None:
            engine.close()
        root.setLevel(prev_level)
        # Deterministic clean exit: wipe the screen and leave only a short note
        # that wren stopped. (Rich Live's cursor accounting at the bottom edge
        # is off by one row under uvicorn's Ctrl+C shutdown, so we don't rely
        # on its erase.)
        served = len(monitor.rows) if monitor is not None else 0
        ran = _fmt_duration(time.monotonic() - started)
        console.clear()
        console.print(f"[dim]wren stopped · served {served} req · ran {ran}[/dim]")

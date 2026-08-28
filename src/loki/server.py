"""OpenAI-compatible loopback server for LOKI."""

from __future__ import annotations

import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .engine import Engine, EngineConfig, load_engine, stream


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "loki"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.2
    stream: Optional[bool] = False


def build_app(engine: Engine, model_name: str) -> FastAPI:
    api = FastAPI(title="LOKI", version="0.1.0")

    @api.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model"}]}

    @api.post("/v1/chat/completions")
    async def chat(req: ChatRequest):
        history = [
            {"role": m.role, "content": [{"type": "text", "text": m.content}]}
            for m in req.messages
        ]
        prompt = engine.chat_prompt(history)
        kwargs = {
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
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


def serve(
    model: str = "mlx-community/Qwen3.6-35B-A3B-4bit",
    cache_gb: float = 6.0,
    prefetch: bool = False,
    prefetch_top_k: int = 16,
    prefetch_lookahead: int = 0,
    predictor: Optional[str] = None,
    host: str = "127.0.0.1",
    port: int = 8080,
):
    """Serve the model on an OpenAI-compatible /v1 endpoint."""
    cfg = EngineConfig(
        model_id=model,
        cache_gb=cache_gb,
        prefetch=prefetch,
        prefetch_top_k=prefetch_top_k,
        prefetch_lookahead=prefetch_lookahead,
        predictor=predictor,
    )
    engine = load_engine(cfg)
    api = build_app(engine, model)
    uvicorn.run(api, host=host, port=port)

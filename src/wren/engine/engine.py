"""High-level engine: lazy-load the model, attach the streaming MoE, generate."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx

from .expert_cache import ExpertCache
from .expert_store import ExpertStore
from .moe import patch_moe


@dataclass
class EngineConfig:
    model_id: str = "mlx-community/Qwen3.6-35B-A3B-4bit"
    cache_gb: float = 6.0
    eviction: str = "lfu"
    prefetch: bool = False
    prefetch_top_k: int = 16
    prefetch_lookahead: int = 0
    predictor: Optional[str] = None
    record_trace: Optional[str] = None
    max_kv_heads: Optional[int] = None
    max_tokens: int = 1024
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 64
    repetition_penalty: float = 1.0
    enable_thinking: bool = False
    extra: dict = field(default_factory=dict)


class Engine:
    def __init__(self, model, processor, store: ExpertStore, cache: ExpertCache, cfg: EngineConfig):
        self.model = model
        self.processor = processor
        self.store = store
        self.cache = cache
        self.cfg = cfg
        self.recorder = None

    @property
    def num_layers(self) -> int:
        return self.store.num_layers

    def stats(self) -> dict:
        return {
            **self.cache.stats(),
            "peak_memory_bytes": int(mx.get_peak_memory()),
        }

    def chat_prompt(
        self, messages: list, enable_thinking: Optional[bool] = None
    ) -> str:
        """Format a list of chat messages into a prompt string.

        ``enable_thinking`` overrides the engine config for this call only
        (None = use the configured default).
        """
        from mlx_vlm.prompt_utils import apply_chat_template

        if enable_thinking is None:
            enable_thinking = self.cfg.enable_thinking
        return apply_chat_template(
            self.processor,
            self.model.config,
            messages,
            enable_thinking=enable_thinking,
        )

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None


class TextProcessor:
    """Minimal processor shim exposing ``.tokenizer`` and ``.detokenizer``.

    mlx-vlm's generate path expects a processor with a ``.tokenizer`` (used for
    encode/decode) and a ``.detokenizer`` (used for streaming).  We avoid the
    full ``AutoProcessor`` because it pulls in the vision/video tower, which
    requires torchvision.  Everything else forwards to the HF tokenizer.
    """

    def __init__(self, tokenizer, detokenizer):
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "detokenizer", detokenizer)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def _load_text_processor(model_path):
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)
    detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
    return TextProcessor(hf_tokenizer, detokenizer_class(hf_tokenizer))


def load_engine(cfg: EngineConfig) -> Engine:
    from mlx_vlm.utils import StoppingCriteria, get_model_path, load_model

    model_path = get_model_path(cfg.model_id)
    model = load_model(model_path, lazy=True)

    # Text-only: load the tokenizer directly (the full AutoProcessor would
    # pull in the vision/video tower, which needs torchvision).
    processor = _load_text_processor(model_path)
    eos = getattr(model.config, "eos_token_id", None)
    processor.tokenizer.stopping_criteria = StoppingCriteria(eos, processor.tokenizer)

    store = ExpertStore(Path(model_path))
    budget = int(cfg.cache_gb * (1024 ** 3))

    predictor = None
    prefetcher = None
    recorder = None

    if cfg.predictor is not None:
        from .predictor import LearnedPredictor

        predictor = LearnedPredictor(
            cfg.predictor,
            top_k=cfg.prefetch_top_k,
            lookahead=8,
        )

    if predictor is not None or cfg.prefetch:
        from .prefetch import Prefetcher

        prefetcher = Prefetcher(store)

    if cfg.record_trace is not None:
        from .trace import TraceRecorder

        recorder = TraceRecorder(cfg.record_trace)

    cache = ExpertCache(store, budget_bytes=budget, eviction=cfg.eviction, prefetcher=prefetcher)
    if prefetcher is not None:
        prefetcher.cache = cache

    group_size, bits, mode = _quant_params(model)
    patch_moe(
        model,
        cache,
        predictor=predictor,
        recorder=recorder,
        group_size=group_size,
        bits=bits,
        mode=mode,
        prefetch_top_k=cfg.prefetch_top_k if cfg.prefetch else 0,
        prefetch_lookahead=cfg.prefetch_lookahead,
    )

    engine = Engine(model, processor, store, cache, cfg)
    engine.recorder = recorder
    return engine


def generate(engine: Engine, prompt: str, **kwargs) -> str:
    """One-shot generation returning the full text."""
    from mlx_vlm import generate as vlm_generate

    k = _gen_kwargs(engine, kwargs)
    result = vlm_generate(engine.model, engine.processor, prompt, **k)
    return result.text if hasattr(result, "text") else str(result)


def stream(engine: Engine, prompt: str, **kwargs):
    """Yield text chunks as they are generated."""
    from mlx_vlm import stream_generate

    k = _gen_kwargs(engine, kwargs)
    for result in stream_generate(engine.model, engine.processor, prompt, **k):
        if getattr(result, "text", None):
            yield result.text


def _gen_kwargs(engine: Engine, overrides: dict) -> dict:
    k = {
        "max_tokens": engine.cfg.max_tokens,
        "temperature": engine.cfg.temperature,
        "top_p": engine.cfg.top_p,
        "top_k": engine.cfg.top_k,
        "repetition_penalty": engine.cfg.repetition_penalty,
        "enable_thinking": engine.cfg.enable_thinking,
    }
    k.update(overrides)
    return k


def _quant_params(model) -> tuple:
    lm = model.language_model
    try:
        gl = lm.model.layers[0].mlp.switch_mlp.gate_proj
        return gl.group_size, gl.bits, getattr(gl, "mode", "affine")
    except Exception:
        return 64, 4, "affine"

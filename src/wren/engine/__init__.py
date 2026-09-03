"""WREN engine: predictive MoE expert streaming."""

from .engine import Engine, EngineConfig, generate, load_engine, stream
from .expert_cache import ExpertCache
from .expert_store import ExpertStore
from .moe import MoEContext, patch_moe
from .predictor import LearnedPredictor, train_predictor
from .trace import TraceRecorder

__all__ = [
    "Engine",
    "EngineConfig",
    "ExpertCache",
    "ExpertStore",
    "LearnedPredictor",
    "MoEContext",
    "TraceRecorder",
    "generate",
    "load_engine",
    "patch_moe",
    "stream",
    "train_predictor",
]

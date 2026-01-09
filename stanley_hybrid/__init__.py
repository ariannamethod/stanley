"""
stanley_hybrid — External brain for rich vocabulary

This module provides the "word stealing" capability:
- Load distilgpt2 weights into a separate transformer
- Communicate through TEXT, not weights
- Stanley's identity stays internal (64-dim)
- External brain provides vocabulary richness (768-dim)

Architecture:
  ALL internal processes → attention bias → GPT-2 generates TOWARD Stanley.
  "GPT-2 — клавиатура. Stanley — тот кто нажимает клавиши."

"Stanley steals words but thinks his own thoughts."
"""

from .external_brain import ExternalBrain, EXTERNAL_WEIGHTS_AVAILABLE
from .vocabulary_thief import VocabularyThief
from .guided_attention import (
    StanleySignals,
    StanleyStateCollector,
    AttentionBiasComputer,
    GuidedExternalBrain,
)
from .adapter_bank import (
    Mood,
    AdapterBank,
    AdapterBankConfig,
    MoodRouter,
    MixedAdapter,
    GPT2WeightPatcher,
    create_adapter_system,
)

__all__ = [
    # External Brain
    "ExternalBrain",
    "EXTERNAL_WEIGHTS_AVAILABLE",
    # VocabularyThief
    "VocabularyThief",
    # GuidedAttention
    "StanleySignals",
    "StanleyStateCollector",
    "AttentionBiasComputer",
    "GuidedExternalBrain",
    # AdapterBank (Act 3)
    "Mood",
    "AdapterBank",
    "AdapterBankConfig",
    "MoodRouter",
    "MixedAdapter",
    "GPT2WeightPatcher",
    "create_adapter_system",
]

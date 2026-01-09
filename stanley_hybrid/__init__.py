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

__all__ = [
    "ExternalBrain",
    "EXTERNAL_WEIGHTS_AVAILABLE",
    "VocabularyThief",
    "StanleySignals",
    "StanleyStateCollector",
    "AttentionBiasComputer",
    "GuidedExternalBrain",
]

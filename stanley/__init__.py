"""
STANLEY — Self Training Attention Non-Linear EntitY

A self-evolving linguistic organism that grows through experience.

Part of the Arianna Method ecosystem.
"""

__version__ = "0.1.0"

from .shard import Shard, MetaNote, combine_deltas
from .memory_sea import MemorySea
from .quantum_buffer import QuantumBuffer, AdaptiveQuantumBuffer
from .fingerprint import (
    compute_fingerprint,
    cosine_similarity,
    novelty_score,
    FingerprintConfig,
    FingerprintIndex,
)
from .router import Router, RouterConfig, AdaptiveRouter
from .inference import (
    StanleyTransformer,
    InferenceEngine,
    Vocab,
    quick_stanley,
)

__all__ = [
    # Core
    "Shard",
    "MetaNote",
    "combine_deltas",
    
    # Storage
    "MemorySea",
    
    # Accumulation
    "QuantumBuffer",
    "AdaptiveQuantumBuffer",
    
    # Fingerprinting
    "compute_fingerprint",
    "cosine_similarity", 
    "novelty_score",
    "FingerprintConfig",
    "FingerprintIndex",
    
    # Routing
    "Router",
    "RouterConfig",
    "AdaptiveRouter",
    
    # Inference
    "StanleyTransformer",
    "InferenceEngine",
    "Vocab",
    "quick_stanley",
]

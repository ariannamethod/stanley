"""
inference — NumPy-only inference engine

Pure NumPy. No PyTorch.
Stanley's transformer with LoRA delta support.
"""

from .transformer import StanleyTransformer, Vocab
from .engine import InferenceEngine, quick_stanley
from .nn import (
    softmax,
    gelu,
    layer_norm,
    sample_basic,
    sample_top_k, 
    sample_top_p,
    entropy_bits,
    entropy_temperature,
)

__all__ = [
    # Main classes
    "StanleyTransformer",
    "InferenceEngine",
    "Vocab",
    "quick_stanley",
    
    # NN primitives
    "softmax",
    "gelu",
    "layer_norm",
    "sample_basic",
    "sample_top_k", 
    "sample_top_p",
    "entropy_bits",
    "entropy_temperature",
]

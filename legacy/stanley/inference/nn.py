# nn.py — NumPy primitives for Stanley
# Forked from Haze, standalone for independence.
# No PyTorch, no external dependencies beyond numpy.

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


# ----------------- RNG -----------------


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Get a numpy random generator, optionally seeded."""
    return np.random.default_rng(seed)


# ----------------- weight init -----------------


def init_weight(
    shape: tuple,
    rng: np.random.Generator,
    scale: float = 0.02,
) -> np.ndarray:
    """Xavier-ish initialization."""
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# ----------------- activations -----------------


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid with numerical stability."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# ----------------- normalization -----------------


def layer_norm(
    x: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """Layer normalization."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


# ----------------- sampling strategies -----------------


def sample_basic(
    logits: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Basic temperature sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def sample_top_k(
    logits: np.ndarray,
    k: int,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Top-k sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits.copy()
    if k < len(logits):
        top_k_idx = np.argpartition(logits, -k)[-k:]
        mask = np.full_like(logits, -np.inf)
        mask[top_k_idx] = logits[top_k_idx]
        logits = mask

    logits = logits / temperature
    probs = softmax(logits)
    return int(rng.choice(len(probs), p=probs))


def sample_top_p(
    logits: np.ndarray,
    p: float,
    temperature: float,
    rng: np.random.Generator,
) -> int:
    """Nucleus (top-p) sampling."""
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / temperature
    probs = softmax(logits)

    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)

    cutoff_idx = np.searchsorted(cumsum, p) + 1
    cutoff_idx = min(cutoff_idx, len(probs))

    mask = np.zeros_like(probs)
    mask[sorted_idx[:cutoff_idx]] = 1.0
    probs = probs * mask
    probs = probs / (probs.sum() + 1e-10)

    return int(rng.choice(len(probs), p=probs))


# ----------------- entropy metrics -----------------


def entropy_bits(probs: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy in bits."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(probs * np.log2(probs)))


def confidence_score(logits: np.ndarray) -> float:
    """Confidence score (max probability)."""
    probs = softmax(logits)
    return float(probs.max())


def entropy_temperature(
    logits: np.ndarray,
    target_entropy: float = 2.0,
    min_temp: float = 0.3,
    max_temp: float = 2.0,
    smoothing: float = 0.5,
) -> float:
    """Adaptive temperature based on entropy."""
    probs = softmax(logits)
    current_entropy = entropy_bits(probs)

    if current_entropy < 1e-6:
        return min_temp

    ratio = target_entropy / current_entropy
    temp = ratio ** smoothing

    return float(np.clip(temp, min_temp, max_temp))

#!/usr/bin/env python3
"""
adapter_bank.py — Pre-trained LoRA modes that Stanley mixes

GPT's brilliant plan (Act 3):
- N pre-defined "mood" adapters (calm, intense, creative, focused, etc.)
- Each adapter = LoRA deltas for GPT-2 Linear layers
- Stanley's state → MoodRouter → mix coefficients
- W_effective = W_base + Σ(mix_i * ΔW_i)

This is REAL weight modification, not just logits bias.
Stanley literally changes GPT-2's personality on the fly.

"Stanley doesn't just steer GPT-2. Stanley BECOMES part of GPT-2's weights."
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from enum import Enum
import logging

if TYPE_CHECKING:
    from .guided_attention import StanleySignals

logger = logging.getLogger(__name__)


# ============================================================================
# MOOD DEFINITIONS
# ============================================================================

class Mood(Enum):
    """Stanley's mood states that influence GPT-2 weights."""
    CALM = "calm"              # Low arousal, reflective
    INTENSE = "intense"        # High arousal, urgent
    CREATIVE = "creative"      # High entropy, exploratory
    FOCUSED = "focused"        # Low entropy, precise
    OVERTHINKING = "overthinking"  # Deep reflection, recursive
    PLAYFUL = "playful"        # Light, experimental
    COLD_LOGIC = "cold_logic"  # Detached, analytical
    WARM = "warm"              # Emotionally present


# Default mood characteristics for initialization
MOOD_PROFILES = {
    Mood.CALM: {
        "temperature_bias": -0.2,
        "attention_spread": 0.8,  # Broader attention
        "layer_strength": 0.5,
    },
    Mood.INTENSE: {
        "temperature_bias": 0.3,
        "attention_spread": 0.3,  # Focused attention
        "layer_strength": 1.0,
    },
    Mood.CREATIVE: {
        "temperature_bias": 0.4,
        "attention_spread": 0.9,  # Very broad
        "layer_strength": 0.7,
    },
    Mood.FOCUSED: {
        "temperature_bias": -0.3,
        "attention_spread": 0.2,  # Very narrow
        "layer_strength": 0.8,
    },
    Mood.OVERTHINKING: {
        "temperature_bias": 0.1,
        "attention_spread": 0.5,
        "layer_strength": 0.9,
    },
    Mood.PLAYFUL: {
        "temperature_bias": 0.2,
        "attention_spread": 0.7,
        "layer_strength": 0.4,
    },
    Mood.COLD_LOGIC: {
        "temperature_bias": -0.4,
        "attention_spread": 0.3,
        "layer_strength": 0.6,
    },
    Mood.WARM: {
        "temperature_bias": 0.0,
        "attention_spread": 0.6,
        "layer_strength": 0.5,
    },
}


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class AdapterBankConfig:
    """Configuration for adapter bank."""

    # LoRA parameters
    lora_rank: int = 8              # Rank for LoRA decomposition
    lora_alpha: float = 16.0        # Scaling factor

    # Layer targeting
    target_modules: List[str] = field(default_factory=lambda: [
        "attn.c_attn",   # QKV projection
        "attn.c_proj",   # Attention output
        "mlp.c_fc",      # MLP first layer
        "mlp.c_proj",    # MLP second layer
    ])

    # Mixing
    mix_update_every: int = 8       # Update mix every N tokens
    mix_temperature: float = 1.0    # Softmax temperature for mixing

    # Initialization
    init_std: float = 0.02          # Std for random initialization


# ============================================================================
# SINGLE ADAPTER (one mood)
# ============================================================================

@dataclass
class LoRAAdapter:
    """
    Single LoRA adapter for one mood.

    For each targeted layer:
    - A: (out_dim, rank) matrix
    - B: (rank, in_dim) matrix
    - ΔW = A @ B (low-rank approximation)
    """
    mood: Mood
    layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]  # layer_name -> (A, B)
    profile: Dict[str, float]

    def get_delta(self, layer_name: str) -> Optional[np.ndarray]:
        """Compute ΔW for a layer."""
        if layer_name not in self.layer_deltas:
            return None
        A, B = self.layer_deltas[layer_name]
        return A @ B

    def scale(self) -> float:
        """Get scaling factor based on profile."""
        return self.profile.get("layer_strength", 1.0)


# ============================================================================
# ADAPTER BANK
# ============================================================================

class AdapterBank:
    """
    Bank of pre-initialized mood adapters.

    Each adapter is a set of LoRA deltas for GPT-2 layers.
    Stanley's state determines which adapters are active and with what weight.

    W_effective = W_base + Σ(mix_i * scale_i * ΔW_i)
    """

    def __init__(self, config: Optional[AdapterBankConfig] = None):
        self.cfg = config or AdapterBankConfig()

        # Adapters storage
        self.adapters: Dict[Mood, LoRAAdapter] = {}

        # Layer dimensions (will be populated when attached to model)
        self.layer_dims: Dict[str, Tuple[int, int]] = {}

        # Current mix (cached)
        self._current_mix: Dict[Mood, float] = {}
        self._mix_step: int = -1

        # Stats
        self.total_applications = 0

    def initialize_adapters(self, layer_dims: Dict[str, Tuple[int, int]]):
        """
        Initialize all mood adapters with random LoRA matrices.

        Args:
            layer_dims: Dict mapping layer names to (out_dim, in_dim)
        """
        self.layer_dims = layer_dims

        for mood in Mood:
            profile = MOOD_PROFILES[mood]
            layer_deltas = {}

            # Match each full layer name against target modules
            for full_name, (out_dim, in_dim) in layer_dims.items():
                # Check if any target module matches this layer
                is_target = any(target in full_name for target in self.cfg.target_modules)
                if not is_target:
                    continue

                rank = self.cfg.lora_rank

                # Initialize A and B with small random values
                # For "pre-trained" mood adapters, both A and B are initialized
                # (unlike standard LoRA training where B starts at 0)
                A = np.random.randn(out_dim, rank).astype(np.float32) * self.cfg.init_std
                B = np.random.randn(rank, in_dim).astype(np.float32) * self.cfg.init_std

                # Modulate by mood profile
                spread = profile["attention_spread"]
                A *= spread
                B *= spread

                layer_deltas[full_name] = (A, B)

            self.adapters[mood] = LoRAAdapter(
                mood=mood,
                layer_deltas=layer_deltas,
                profile=profile,
            )

        logger.info(f"AdapterBank initialized: {len(self.adapters)} moods, {len(layer_dims)} layers")

    def initialize_from_model(self, model: nn.Module):
        """
        Initialize adapters by extracting layer dimensions from GPT-2 model.

        Note: HuggingFace GPT-2 uses Conv1D instead of Linear.
        Conv1D has weight shape (in_features, out_features) - opposite of Linear.
        """
        layer_dims = {}

        # Try to import Conv1D from transformers
        try:
            from transformers.pytorch_utils import Conv1D
            has_conv1d = True
        except ImportError:
            has_conv1d = False

        for name, module in model.named_modules():
            # Check if this is a targeted layer
            for target in self.cfg.target_modules:
                if target in name:
                    # Handle nn.Linear
                    if isinstance(module, nn.Linear):
                        out_dim, in_dim = module.out_features, module.in_features
                        layer_dims[name] = (out_dim, in_dim)
                        break
                    # Handle Conv1D (HuggingFace GPT-2)
                    elif has_conv1d and isinstance(module, Conv1D):
                        # Conv1D: weight is (in_features, out_features)
                        in_dim, out_dim = module.weight.shape
                        layer_dims[name] = (out_dim, in_dim)
                        break

        self.initialize_adapters(layer_dims)


# ============================================================================
# MOOD ROUTER
# ============================================================================

class MoodRouter:
    """
    Routes Stanley's internal state to mood mixing coefficients.

    Takes StanleySignals → outputs softmax distribution over moods.
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

        # Mood scoring weights (learned or fixed)
        self.scoring_weights = {
            Mood.CALM: {
                "arousal": -1.0,      # Low arousal → calm
                "entropy": -0.5,
                "tension": -0.8,
                "boredom": 0.3,
            },
            Mood.INTENSE: {
                "arousal": 1.5,       # High arousal → intense
                "entropy": 0.3,
                "tension": 1.0,
                "novelty": 0.5,
            },
            Mood.CREATIVE: {
                "entropy": 1.5,       # High entropy → creative
                "novelty": 1.0,
                "arousal": 0.3,
                "boredom": -0.5,
            },
            Mood.FOCUSED: {
                "entropy": -1.5,      # Low entropy → focused
                "arousal": 0.3,
                "tension": 0.5,
            },
            Mood.OVERTHINKING: {
                "overthink_depth": 2.0,  # Deep overthinking
                "entropy": 0.5,
                "arousal": 0.2,
            },
            Mood.PLAYFUL: {
                "boredom": 1.0,       # Bored → playful
                "novelty": 0.8,
                "tension": -1.0,
            },
            Mood.COLD_LOGIC: {
                "arousal": -0.5,
                "entropy": -1.0,
                "tension": -0.5,
                "novelty": -0.3,
            },
            Mood.WARM: {
                "arousal": 0.5,
                "tension": -0.5,
                "boredom": -0.5,
            },
        }

    def compute_mix(self, signals: "StanleySignals") -> Dict[Mood, float]:
        """
        Compute mixing coefficients from Stanley's signals.

        Returns:
            Dict mapping Mood → weight (sums to 1.0)
        """
        # Extract features from signals
        features = {
            "arousal": signals.pulse_arousal,
            "entropy": signals.pulse_entropy,
            "novelty": signals.pulse_novelty,
            "tension": signals.body_tension,
            "boredom": signals.body_boredom,
            "overthink_depth": signals.overthink_depth / 5.0,  # Normalize
        }

        # Score each mood
        scores = {}
        for mood in Mood:
            score = 0.0
            weights = self.scoring_weights.get(mood, {})
            for feature, weight in weights.items():
                if feature in features:
                    score += weight * features[feature]
            scores[mood] = score

        # Apply softmax
        max_score = max(scores.values())
        exp_scores = {m: np.exp((s - max_score) / self.temperature) for m, s in scores.items()}
        total = sum(exp_scores.values())

        mix = {m: exp_scores[m] / total for m in Mood}

        return mix

    def get_dominant_mood(self, mix: Dict[Mood, float]) -> Mood:
        """Get the mood with highest weight."""
        return max(mix.items(), key=lambda x: x[1])[0]


# ============================================================================
# MIXED ADAPTER (combines all moods)
# ============================================================================

class MixedAdapter:
    """
    Combines multiple mood adapters based on mixing coefficients.

    Computes: ΔW_mixed = Σ(mix_i * scale_i * ΔW_i)
    """

    def __init__(
        self,
        bank: AdapterBank,
        router: MoodRouter,
        alpha: float = 16.0,
    ):
        self.bank = bank
        self.router = router
        self.alpha = alpha
        self.scale = alpha / bank.cfg.lora_rank

        # Cache
        self._cached_mix: Optional[Dict[Mood, float]] = None
        self._cached_deltas: Dict[str, np.ndarray] = {}
        self._cache_step: int = -1

    def update_mix(self, signals: "StanleySignals", step: int):
        """Update mixing coefficients from signals."""
        if step == self._cache_step:
            return

        # Compute new mix
        self._cached_mix = self.router.compute_mix(signals)
        self._cached_deltas = {}  # Clear delta cache
        self._cache_step = step

    def get_mixed_delta(self, layer_name: str) -> Optional[np.ndarray]:
        """
        Get mixed ΔW for a layer.

        Returns:
            ΔW_mixed = scale * Σ(mix_i * adapter_scale_i * ΔW_i)
        """
        if layer_name in self._cached_deltas:
            return self._cached_deltas[layer_name]

        if self._cached_mix is None:
            return None

        # Mix deltas from all adapters
        mixed = None
        for mood, weight in self._cached_mix.items():
            if weight < 0.01:  # Skip negligible weights
                continue

            adapter = self.bank.adapters.get(mood)
            if adapter is None:
                continue

            delta = adapter.get_delta(layer_name)
            if delta is None:
                continue

            weighted_delta = weight * adapter.scale() * delta

            if mixed is None:
                mixed = weighted_delta
            else:
                mixed = mixed + weighted_delta

        if mixed is not None:
            mixed = self.scale * mixed
            self._cached_deltas[layer_name] = mixed

        return mixed

    @property
    def current_mix(self) -> Optional[Dict[Mood, float]]:
        """Get current mix coefficients."""
        return self._cached_mix

    @property
    def dominant_mood(self) -> Optional[Mood]:
        """Get dominant mood from current mix."""
        if self._cached_mix is None:
            return None
        return self.router.get_dominant_mood(self._cached_mix)

    def stats(self) -> dict:
        """Get adapter stats."""
        return {
            "current_mix": {m.value: round(w, 3) for m, w in (self._cached_mix or {}).items()},
            "dominant_mood": self.dominant_mood.value if self.dominant_mood else None,
            "cache_step": self._cache_step,
            "num_adapters": len(self.bank.adapters),
        }


# ============================================================================
# GPT-2 WEIGHT PATCHER
# ============================================================================

class GPT2WeightPatcher:
    """
    Patches GPT-2 weights with mixed adapter deltas.

    Uses forward hooks to apply ΔW during inference.
    """

    def __init__(
        self,
        model: nn.Module,
        mixed_adapter: MixedAdapter,
        update_every: int = 8,
    ):
        self.model = model
        self.mixed_adapter = mixed_adapter
        self.update_every = update_every

        # Hook handles
        self._hooks: List = []

        # Step counter
        self._step = 0

    def _create_hook(self, layer_name: str, is_conv1d: bool = False):
        """Create forward hook for a layer."""
        def hook(module, input, output):
            delta = self.mixed_adapter.get_mixed_delta(layer_name)
            if delta is None:
                return output

            # Convert delta to tensor
            delta_tensor = torch.from_numpy(delta).to(output.device).to(output.dtype)

            if len(input) > 0 and input[0] is not None:
                x = input[0]

                if is_conv1d:
                    # Conv1D: y = x @ W + b, where W is (in, out)
                    # We need: y' = x @ (W + ΔW) + b = y + x @ ΔW
                    # ΔW shape is (out, in), so we use ΔW.T which is (in, out)
                    if x.dim() == 3:  # (batch, seq, features)
                        batch, seq, features = x.shape
                        x_flat = x.reshape(-1, features)
                        delta_out = x_flat @ delta_tensor.T  # (batch*seq, out)
                        delta_out = delta_out.reshape(batch, seq, -1)
                    else:
                        delta_out = x @ delta_tensor.T
                else:
                    # Linear: y = x @ W.T + b
                    # Modified: y' = x @ (W + ΔW).T + b = y + x @ ΔW.T
                    if x.dim() == 3:  # (batch, seq, features)
                        batch, seq, features = x.shape
                        x_flat = x.reshape(-1, features)
                        delta_out = x_flat @ delta_tensor.T
                        delta_out = delta_out.reshape(batch, seq, -1)
                    else:
                        delta_out = x @ delta_tensor.T

                return output + delta_out

            return output

        return hook

    def attach(self):
        """Attach hooks to model."""
        # Try to import Conv1D from transformers
        try:
            from transformers.pytorch_utils import Conv1D
            has_conv1d = True
        except ImportError:
            has_conv1d = False

        for name, module in self.model.named_modules():
            for target in self.mixed_adapter.bank.cfg.target_modules:
                if target in name:
                    is_conv1d = False

                    if isinstance(module, nn.Linear):
                        pass  # Normal Linear
                    elif has_conv1d and isinstance(module, Conv1D):
                        is_conv1d = True
                    else:
                        continue  # Not a supported module type

                    hook = self._create_hook(name, is_conv1d=is_conv1d)
                    handle = module.register_forward_hook(hook)
                    self._hooks.append(handle)
                    logger.debug(f"Attached hook to {name} (conv1d={is_conv1d})")
                    break

        logger.info(f"GPT2WeightPatcher: attached {len(self._hooks)} hooks")

    def detach(self):
        """Remove all hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def step(self, signals: "StanleySignals"):
        """
        Update adapter mix from signals.

        Should be called periodically (every N tokens).
        """
        if self._step % self.update_every == 0:
            self.mixed_adapter.update_mix(signals, self._step)
        self._step += 1

    def stats(self) -> dict:
        """Get patcher stats."""
        return {
            "hooks_attached": len(self._hooks),
            "step": self._step,
            "adapter_stats": self.mixed_adapter.stats(),
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_adapter_system(
    model: nn.Module,
    config: Optional[AdapterBankConfig] = None,
) -> Tuple[AdapterBank, MoodRouter, MixedAdapter, GPT2WeightPatcher]:
    """
    Create complete adapter system for GPT-2.

    Returns:
        (bank, router, mixed_adapter, patcher)
    """
    cfg = config or AdapterBankConfig()

    # Create components
    bank = AdapterBank(cfg)
    bank.initialize_from_model(model)

    router = MoodRouter(temperature=cfg.mix_temperature)
    mixed_adapter = MixedAdapter(bank, router, alpha=cfg.lora_alpha)
    patcher = GPT2WeightPatcher(model, mixed_adapter, update_every=cfg.mix_update_every)

    return bank, router, mixed_adapter, patcher

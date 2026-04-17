"""
experts.py — Resonant Experts: MOE-style temperature routing for Stanley

Ported from Haze with adaptations for Stanley's pulse system.

The philosophy:
- No fixed routing, always a MIXTURE of all experts
- Weights computed from pulse signals: entropy, arousal, novelty
- Each expert has a temperature and semantic weight
- Final temperature is a weighted blend, not a single expert choice

This creates DYNAMIC personality - Stanley adapts its "voice" based on
the resonance field, not on what the user asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .subjectivity import Pulse

logger = logging.getLogger(__name__)


@dataclass
class Expert:
    """A resonant expert - a perspective on the field."""
    name: str
    temperature: float
    semantic_weight: float
    description: str


# The four experts (inspired by Leo, ported from Haze)
EXPERTS = [
    Expert(
        name="structural",
        temperature=0.7,
        semantic_weight=0.2,
        description="Grammar-focused, coherent structure",
    ),
    Expert(
        name="semantic",
        temperature=0.9,
        semantic_weight=0.5,
        description="Meaning-focused, thematic coherence",
    ),
    Expert(
        name="creative",
        temperature=1.2,
        semantic_weight=0.4,
        description="Exploratory, high entropy drift",
    ),
    Expert(
        name="precise",
        temperature=0.5,
        semantic_weight=0.3,
        description="Conservative, low entropy grounding",
    ),
]


class ExpertMixture(NamedTuple):
    """Result of expert routing - a weighted mixture."""
    temperature: float
    semantic_weight: float
    weights: Dict[str, float]  # name -> weight
    dominant: str  # name of highest-weighted expert


class FieldSignals(NamedTuple):
    """Input signals for expert routing."""
    entropy: float      # 0-1: distribution entropy (how spread the choices are)
    arousal: float      # 0-1: emotional charge
    novelty: float      # 0-1: how new/unknown the input is
    perplexity: float   # 0-inf: model uncertainty (optional, default 1.0)


def pulse_to_signals(pulse: "Pulse") -> FieldSignals:
    """Convert Stanley's Pulse to FieldSignals for expert routing."""
    return FieldSignals(
        entropy=max(0.0, min(1.0, pulse.entropy)),
        arousal=max(0.0, min(1.0, pulse.arousal)),
        novelty=max(0.0, min(1.0, pulse.novelty)),
        perplexity=1.0,  # Stanley doesn't compute perplexity yet
    )


def compute_expert_weights(signals: FieldSignals) -> Dict[str, float]:
    """
    Compute expert weights from field signals.

    This is the core MOE logic, but always returns a MIXTURE:
    - High entropy → more creative weight
    - Low entropy → more precise weight
    - High arousal → more semantic weight
    - High novelty → more structural weight (ground in known patterns)
    - High perplexity → more precise weight (reduce uncertainty)
    """
    weights = {}

    # Base weights (all experts always contribute)
    base = 0.1

    # Structural: grounded in known patterns
    # Higher when novelty is high (need to ground in familiar)
    structural = base + 0.3 * signals.novelty + 0.1 * (1.0 - signals.arousal)
    weights["structural"] = structural

    # Semantic: meaning-focused
    # Higher when arousal is high (emotional content)
    # Also higher when entropy is moderate (not too chaotic)
    semantic = base + 0.4 * signals.arousal + 0.2 * (1.0 - abs(signals.entropy - 0.5) * 2)
    weights["semantic"] = semantic

    # Creative: exploratory
    # Higher when entropy is high (explore the space)
    # Lower when novelty is high (don't go too far from known)
    creative = base + 0.4 * signals.entropy + 0.2 * (1.0 - signals.novelty)
    weights["creative"] = creative

    # Precise: conservative
    # Higher when entropy is low (stay grounded)
    # Higher when perplexity is high (reduce uncertainty)
    perp_factor = min(1.0, signals.perplexity / 2.0)
    precise = base + 0.3 * (1.0 - signals.entropy) + 0.3 * perp_factor
    weights["precise"] = precise

    # Normalize weights to sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def compute_expert_weights_with_momentum(
    signals: FieldSignals,
    context_history: Optional[List[Dict[str, float]]] = None,
    momentum: float = 0.3,
) -> Dict[str, float]:
    """
    Enhanced expert weight computation with context memory and momentum.

    Learns from previous routing decisions to maintain consistency
    and avoid rapid switching between experts.
    """
    current_weights = compute_expert_weights(signals)

    if context_history and len(context_history) > 0 and momentum > 0:
        # Blend with recent history (exponential weighting)
        history_weights = {
            "structural": 0.0,
            "semantic": 0.0,
            "creative": 0.0,
            "precise": 0.0,
        }

        # Weight recent history more
        decay = 0.7
        total_weight = 0.0
        for i, hist in enumerate(context_history[-5:]):  # Last 5 steps
            weight = decay ** (len(context_history) - i - 1)
            total_weight += weight
            for expert in history_weights:
                if expert in hist:
                    history_weights[expert] += hist[expert] * weight

        if total_weight > 0:
            for expert in history_weights:
                history_weights[expert] /= total_weight

        # Blend current with history
        blended = {}
        for expert in current_weights:
            blended[expert] = (
                momentum * history_weights.get(expert, 0.25) +
                (1 - momentum) * current_weights[expert]
            )

        # Renormalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    return current_weights


def blend_experts(weights: Dict[str, float]) -> ExpertMixture:
    """
    Blend expert parameters using weights.

    Returns a mixture of temperature and semantic_weight.
    """
    expert_map = {e.name: e for e in EXPERTS}

    temp = 0.0
    sem = 0.0

    for name, weight in weights.items():
        expert = expert_map.get(name)
        if expert:
            temp += expert.temperature * weight
            sem += expert.semantic_weight * weight

    # Find dominant expert
    dominant = max(weights.items(), key=lambda x: x[1])[0]

    return ExpertMixture(
        temperature=temp,
        semantic_weight=sem,
        weights=weights,
        dominant=dominant,
    )


def route_from_pulse(pulse: "Pulse") -> ExpertMixture:
    """
    Main entry point for Stanley: compute expert mixture from pulse.

    Usage:
        pulse = subjectivity.compute_pulse(user_input)
        mixture = route_from_pulse(pulse)
        # mixture.temperature → blended temp for generation
        # mixture.dominant → "creative", "semantic", etc.
    """
    signals = pulse_to_signals(pulse)
    weights = compute_expert_weights(signals)
    mixture = blend_experts(weights)

    logger.debug(
        f"Expert routing: {describe_mixture(mixture)} "
        f"(pulse: nov={pulse.novelty:.2f}, aro={pulse.arousal:.2f}, ent={pulse.entropy:.2f})"
    )

    return mixture


def route_from_signals(signals: FieldSignals) -> ExpertMixture:
    """
    Route from raw signals (for testing or manual control).
    """
    weights = compute_expert_weights(signals)
    return blend_experts(weights)


def route_single_expert(pulse: "Pulse") -> Expert:
    """
    Leo-style routing: pick the single best expert.

    Useful for simpler cases or A/B testing.
    """
    signals = pulse_to_signals(pulse)
    weights = compute_expert_weights(signals)
    dominant = max(weights.items(), key=lambda x: x[1])[0]
    expert_map = {e.name: e for e in EXPERTS}
    return expert_map[dominant]


def describe_mixture(mixture: ExpertMixture) -> str:
    """Human-readable description of expert mixture."""
    parts = []
    for name, weight in sorted(mixture.weights.items(), key=lambda x: -x[1]):
        pct = int(weight * 100)
        if pct > 0:
            parts.append(f"{name}:{pct}%")
    return f"temp={mixture.temperature:.2f} [{', '.join(parts)}]"


# Direct signal creation for testing
def create_signals(
    entropy: float = 0.5,
    arousal: float = 0.5,
    novelty: float = 0.5,
    perplexity: float = 1.0,
) -> FieldSignals:
    """Create FieldSignals directly (for testing)."""
    return FieldSignals(
        entropy=max(0.0, min(1.0, entropy)),
        arousal=max(0.0, min(1.0, arousal)),
        novelty=max(0.0, min(1.0, novelty)),
        perplexity=max(0.0, perplexity),
    )


if __name__ == "__main__":
    print("=== Stanley Resonant Experts Demo ===\n")

    test_cases = [
        ("neutral", create_signals(entropy=0.5, arousal=0.5, novelty=0.5)),
        ("high entropy", create_signals(entropy=0.9, arousal=0.3, novelty=0.2)),
        ("low entropy", create_signals(entropy=0.1, arousal=0.2, novelty=0.3)),
        ("high arousal", create_signals(entropy=0.5, arousal=0.9, novelty=0.3)),
        ("high novelty", create_signals(entropy=0.5, arousal=0.3, novelty=0.9)),
    ]

    for name, signals in test_cases:
        mixture = route_from_signals(signals)
        print(f"{name}:")
        print(f"  signals: entropy={signals.entropy:.1f} arousal={signals.arousal:.1f} novelty={signals.novelty:.1f}")
        print(f"  mixture: {describe_mixture(mixture)}")
        print(f"  dominant: {mixture.dominant}")
        print()

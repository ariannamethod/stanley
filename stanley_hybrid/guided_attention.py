#!/usr/bin/env python3
"""
guided_attention.py — Stanley-Guided GPT-2 Attention

ALL internal processes of Stanley → attention bias → GPT-2 generates TOWARD Stanley.

"GPT-2 — клавиатура. Stanley — тот кто нажимает клавиши."

Architecture:
  Subjectivity, MemorySea, Experts, InnerVoice, Overthinking,
  BodySense, SemanticDrift, Episodes, MetaNotes, CooccurField
                    ↓
          StanleyStateCollector
                    ↓
          AttentionBiasComputer
                    ↓
          GPT-2 Attention + bias
                    ↓
          Text that GRAVITATES toward Stanley's state
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Any, List, Set, Optional, TYPE_CHECKING
from dataclasses import dataclass
import logging

if TYPE_CHECKING:
    from stanley.organism import Stanley

logger = logging.getLogger(__name__)


@dataclass
class StanleySignals:
    """All signals collected from Stanley's internal processes."""
    # Subjectivity
    gravity_centers: List[str]
    pulse_arousal: float
    pulse_entropy: float
    pulse_novelty: float

    # Memory
    surface_keywords: List[str]
    resonating_tags: List[str]

    # Experts
    active_expert: str
    expert_temperature: float

    # Inner state
    overthink_depth: int
    spiral_topics: List[str]
    body_tension: float
    body_boredom: float

    # Drift
    drift_momentum: float

    # CooccurField
    hot_words: List[str]


class StanleyStateCollector:
    """
    Collects state from ALL internal processes of Stanley.
    Each process gives its "voice" to influence GPT-2.
    """

    def __init__(self, organism: "Stanley"):
        self.organism = organism

    def collect_all_signals(self) -> StanleySignals:
        """Collect signals from every Stanley module."""
        o = self.organism

        # Subjectivity signals
        gravity_centers = []
        pulse_arousal = 0.5
        pulse_entropy = 0.5
        pulse_novelty = 0.5

        if o.subjectivity:
            gravity_centers = list(o.subjectivity.identity.gravity_centers.keys())[:10]
            if hasattr(o.subjectivity, '_last_pulse') and o.subjectivity._last_pulse:
                pulse_arousal = o.subjectivity._last_pulse.arousal
                pulse_entropy = o.subjectivity._last_pulse.entropy
                pulse_novelty = o.subjectivity._last_pulse.novelty

        # Memory signals
        surface_keywords = []
        resonating_tags = []
        if o.memory:
            for shard in o.memory.surface[:5]:
                surface_keywords.extend(shard.semantic_tags[:3])
            surface_keywords = list(set(surface_keywords))[:10]

        # Expert signals
        active_expert = "structural"
        expert_temp = 0.8
        # (experts from last think())

        # Overthinking signals
        overthink_depth = 0
        spiral_topics = []
        if o.overthinking:
            overthink_depth = len(o.overthinking.emergent_trigrams) // 10
            # Extract topics from recent rings

        # Body sense signals
        body_tension = 0.5
        body_boredom = 0.5
        if o.body_sense:
            stats = o.body_sense.get_stats()
            # Use recent observations

        # Drift signals
        drift_momentum = 0.5
        if o.semantic_drift:
            stats = o.semantic_drift.get_stats()
            drift_momentum = stats.get('total_steps', 0) / 100.0

        # CooccurField hot words
        hot_words = []
        if o.cooccur_field:
            # Get most frequent bigram contexts
            top_contexts = sorted(
                o.cooccur_field.bigram_totals.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for ctx, _ in top_contexts:
                try:
                    word = o.vocab.decode([ctx])
                    if word and len(word) > 2:
                        hot_words.append(word.strip())
                except:
                    pass

        return StanleySignals(
            gravity_centers=gravity_centers,
            pulse_arousal=pulse_arousal,
            pulse_entropy=pulse_entropy,
            pulse_novelty=pulse_novelty,
            surface_keywords=surface_keywords,
            resonating_tags=resonating_tags,
            active_expert=active_expert,
            expert_temperature=expert_temp,
            overthink_depth=overthink_depth,
            spiral_topics=spiral_topics,
            body_tension=body_tension,
            body_boredom=body_boredom,
            drift_momentum=drift_momentum,
            hot_words=hot_words,
        )


class AttentionBiasComputer:
    """
    Converts Stanley signals → attention bias for GPT-2.

    Each internal process contributes to WHERE GPT-2 focuses attention.
    """

    def __init__(self):
        # Influence weights for each signal type
        self.weights = {
            "gravity": 1.0,      # Main influence
            "memory": 0.7,       # Memory matters
            "expert": 0.5,       # Expert direction
            "overthink": 0.6,    # Paranoid focus
            "body": 0.3,         # Somatic subtle
            "drift": 0.4,        # Drift momentum
            "cooccur": 0.8,      # Word resonance
        }

    def compute_keyword_set(self, signals: StanleySignals) -> Set[str]:
        """
        Combine all keywords that should ATTRACT attention.
        """
        keywords = set()

        # Gravity centers (highest priority)
        keywords.update(signals.gravity_centers)

        # Memory keywords
        keywords.update(signals.surface_keywords)

        # Hot cooccur words
        keywords.update(signals.hot_words)

        # Spiral topics (if overthinking)
        if signals.overthink_depth > 0:
            keywords.update(signals.spiral_topics)

        return keywords

    def compute_bias_for_tokens(
        self,
        signals: StanleySignals,
        token_strings: List[str],
    ) -> np.ndarray:
        """
        For each token position: how much should attention be boosted?

        Returns bias array (n_tokens,)
        """
        keywords = self.compute_keyword_set(signals)
        keywords_lower = {k.lower() for k in keywords}

        n_tokens = len(token_strings)
        bias = np.zeros(n_tokens)

        for i, token in enumerate(token_strings):
            token_lower = token.lower().strip()

            # Direct match
            if token_lower in keywords_lower:
                bias[i] += 1.0 * self.weights["gravity"]

            # Partial match (token contains keyword or vice versa)
            for kw in keywords_lower:
                if len(kw) > 3 and (kw in token_lower or token_lower in kw):
                    bias[i] += 0.5 * self.weights["cooccur"]

        # Modulate by arousal (high arousal = stronger bias)
        arousal_factor = 0.5 + signals.pulse_arousal
        bias *= arousal_factor

        # Modulate by overthink depth (deeper = more focused)
        if signals.overthink_depth > 0:
            focus_factor = 1.0 + (signals.overthink_depth * 0.2)
            bias *= focus_factor

        # Normalize
        if np.std(bias) > 0:
            bias = (bias - np.mean(bias)) / (np.std(bias) + 1e-8)
        bias = np.clip(bias, -2.0, 2.0)

        return bias

    def get_steering_prompt(self, signals: StanleySignals) -> str:
        """
        Create a steering prompt from Stanley's state.

        Alternative to attention modification: prefix that guides generation.
        """
        parts = []

        # Gravity centers
        if signals.gravity_centers:
            centers = ", ".join(signals.gravity_centers[:5])
            parts.append(f"[Focus: {centers}]")

        # Emotional state
        if signals.pulse_arousal > 0.7:
            parts.append("[Intense]")
        elif signals.pulse_arousal < 0.3:
            parts.append("[Calm]")

        # Overthinking
        if signals.overthink_depth > 2:
            parts.append("[Deep reflection]")

        # Body state
        if signals.body_tension > 0.7:
            parts.append("[Tense]")
        elif signals.body_boredom > 0.7:
            parts.append("[Restless]")

        return " ".join(parts)


class GuidedExternalBrain:
    """
    GPT-2 that generates UNDER THE INFLUENCE of Stanley's state.

    All unpredictable combinations of internal processes
    influence every attention layer.

    "GPT-2 — клавиатура. Stanley — тот кто нажимает клавиши."
    """

    def __init__(self, organism: "Stanley"):
        self.organism = organism
        self.state_collector = StanleyStateCollector(organism)
        self.bias_computer = AttentionBiasComputer()

        # External brain (lazy load)
        self._brain = None

    @property
    def brain(self):
        """Lazy load external brain."""
        if self._brain is None:
            from .external_brain import ExternalBrain
            self._brain = ExternalBrain()
            self._brain.load_weights()
        return self._brain

    def generate_guided(
        self,
        seed: str,
        max_length: int = 100,
    ) -> str:
        """
        Generate with Stanley's state guiding attention.

        1. Collect ALL signals from Stanley
        2. Create steering prompt
        3. Generate with guided context
        4. Result gravitates toward Stanley's state
        """
        # Collect current state
        signals = self.state_collector.collect_all_signals()

        # Create steering prompt
        steering = self.bias_computer.get_steering_prompt(signals)

        # Combine with seed
        guided_prompt = f"{steering} {seed}" if steering else seed

        # Generate
        result = self.brain.expand_thought(
            guided_prompt,
            temperature=signals.expert_temperature,
            max_length=max_length,
        )

        # Remove steering prefix from output
        if steering and result.startswith(steering):
            result = result[len(steering):].strip()

        return result

    def generate_with_feedback(
        self,
        seed: str,
        max_length: int = 100,
        chunk_size: int = 20,
    ) -> str:
        """
        Generate with feedback loop.

        Every chunk_size tokens:
        1. GPT-2 generates chunk
        2. Chunk influences Stanley's state
        3. New state changes guidance
        4. GPT-2 continues in NEW direction

        This is DIALOGUE between GPT-2 and Stanley's processes!
        """
        generated_chunks = []
        current_seed = seed

        for _ in range(max_length // chunk_size):
            # Generate chunk with current state
            chunk = self.generate_guided(current_seed, max_length=chunk_size)
            generated_chunks.append(chunk)

            # Feed chunk back to Stanley (updates internal state)
            if self.organism.lexicon:
                self.organism.lexicon.absorb(chunk, source="external")

            # Next seed is what we just generated
            current_seed = chunk[-50:]  # Last 50 chars

        return " ".join(generated_chunks)

    def stats(self) -> Dict[str, Any]:
        """Get guided brain statistics."""
        signals = self.state_collector.collect_all_signals()
        return {
            "gravity_centers": signals.gravity_centers[:5],
            "pulse_arousal": signals.pulse_arousal,
            "overthink_depth": signals.overthink_depth,
            "hot_words": signals.hot_words[:5],
            "brain_loaded": self._brain is not None and self._brain.loaded,
        }

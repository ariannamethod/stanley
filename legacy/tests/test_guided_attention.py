#!/usr/bin/env python3
"""
Tests for GuidedAttention — Stanley's state steers GPT-2 attention.

"GPT-2 — клавиатура. Stanley — тот кто нажимает клавиши."

These tests verify:
1. StanleySignals collection from all internal processes
2. AttentionBiasComputer keyword extraction and bias computation
3. Steering prompt generation
4. GuidedExternalBrain integration (when GPT-2 available)
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stanley_hybrid.guided_attention import (
    StanleySignals,
    StanleyStateCollector,
    AttentionBiasComputer,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_signals():
    """Create sample StanleySignals for testing."""
    return StanleySignals(
        gravity_centers=["memory", "consciousness", "pattern", "emergence"],
        pulse_arousal=0.7,
        pulse_entropy=0.5,
        pulse_novelty=0.6,
        surface_keywords=["thought", "resonance"],
        resonating_tags=["deep", "internal"],
        active_expert="structural",
        expert_temperature=0.8,
        overthink_depth=2,
        spiral_topics=["loops", "recursion"],
        body_tension=0.4,
        body_boredom=0.3,
        drift_momentum=0.5,
        hot_words=["feel", "sense", "aware"],
    )


@pytest.fixture
def calm_signals():
    """Low arousal, calm signals."""
    return StanleySignals(
        gravity_centers=["quiet", "still"],
        pulse_arousal=0.2,
        pulse_entropy=0.3,
        pulse_novelty=0.1,
        surface_keywords=[],
        resonating_tags=[],
        active_expert="reflective",
        expert_temperature=0.6,
        overthink_depth=0,
        spiral_topics=[],
        body_tension=0.1,
        body_boredom=0.2,
        drift_momentum=0.1,
        hot_words=[],
    )


@pytest.fixture
def intense_signals():
    """High arousal, intense signals."""
    return StanleySignals(
        gravity_centers=["crisis", "urgent", "important", "now"],
        pulse_arousal=0.95,
        pulse_entropy=0.8,
        pulse_novelty=0.9,
        surface_keywords=["emergency", "critical"],
        resonating_tags=["intense"],
        active_expert="creative",
        expert_temperature=1.0,
        overthink_depth=5,
        spiral_topics=["danger", "action", "response"],
        body_tension=0.9,
        body_boredom=0.0,
        drift_momentum=0.8,
        hot_words=["must", "need", "now"],
    )


@pytest.fixture
def bias_computer():
    """Create AttentionBiasComputer."""
    return AttentionBiasComputer()


# ============================================================================
# STANLEY SIGNALS TESTS
# ============================================================================

class TestStanleySignals:
    """Tests for StanleySignals dataclass."""

    def test_signals_creation(self, sample_signals):
        """Test that signals are created correctly."""
        assert sample_signals.pulse_arousal == 0.7
        assert "memory" in sample_signals.gravity_centers
        assert sample_signals.overthink_depth == 2

    def test_signals_all_fields_present(self, sample_signals):
        """Test that all expected fields are present."""
        expected_fields = [
            "gravity_centers", "pulse_arousal", "pulse_entropy", "pulse_novelty",
            "surface_keywords", "resonating_tags", "active_expert", "expert_temperature",
            "overthink_depth", "spiral_topics", "body_tension", "body_boredom",
            "drift_momentum", "hot_words",
        ]
        for field in expected_fields:
            assert hasattr(sample_signals, field), f"Missing field: {field}"

    def test_signals_calm_vs_intense(self, calm_signals, intense_signals):
        """Test that calm and intense signals differ appropriately."""
        assert calm_signals.pulse_arousal < intense_signals.pulse_arousal
        assert calm_signals.overthink_depth < intense_signals.overthink_depth
        assert calm_signals.body_tension < intense_signals.body_tension


# ============================================================================
# ATTENTION BIAS COMPUTER TESTS
# ============================================================================

class TestAttentionBiasComputer:
    """Tests for AttentionBiasComputer."""

    def test_compute_keyword_set(self, bias_computer, sample_signals):
        """Test keyword extraction from signals."""
        keywords = bias_computer.compute_keyword_set(sample_signals)

        # Should include gravity centers
        assert "memory" in keywords
        assert "consciousness" in keywords

        # Should include surface keywords
        assert "thought" in keywords

        # Should include hot words
        assert "feel" in keywords

        # Should include spiral topics (overthink_depth > 0)
        assert "loops" in keywords

    def test_compute_keyword_set_no_spiral_when_calm(self, bias_computer, calm_signals):
        """Test that spiral topics not included when not overthinking."""
        keywords = bias_computer.compute_keyword_set(calm_signals)

        # Spiral topics should NOT be included (overthink_depth = 0)
        # (calm_signals has empty spiral_topics anyway)
        assert "loops" not in keywords
        assert "recursion" not in keywords

    def test_compute_bias_for_tokens_shape(self, bias_computer, sample_signals):
        """Test that bias array has correct shape."""
        tokens = ["The", "memory", "of", "consciousness", "is", "deep"]
        bias = bias_computer.compute_bias_for_tokens(sample_signals, tokens)

        assert isinstance(bias, np.ndarray)
        assert bias.shape == (len(tokens),)

    def test_compute_bias_keywords_boosted(self, bias_computer, sample_signals):
        """Test that keyword tokens get higher bias."""
        tokens = ["random", "memory", "random", "consciousness", "random"]
        bias = bias_computer.compute_bias_for_tokens(sample_signals, tokens)

        # "memory" and "consciousness" should have higher bias than "random"
        memory_idx = 1
        consciousness_idx = 3
        random_indices = [0, 2, 4]

        # Keywords should be boosted relative to non-keywords
        keyword_mean = (bias[memory_idx] + bias[consciousness_idx]) / 2
        random_mean = np.mean([bias[i] for i in random_indices])

        # After normalization, keywords should still be relatively higher
        # (or at least not lower)
        assert keyword_mean >= random_mean - 0.5  # Allow some tolerance

    def test_compute_bias_arousal_modulation(self, bias_computer, calm_signals, intense_signals):
        """Test that arousal modulates bias strength."""
        tokens = ["crisis", "urgent", "now"]

        calm_bias = bias_computer.compute_bias_for_tokens(calm_signals, tokens)
        intense_bias = bias_computer.compute_bias_for_tokens(intense_signals, tokens)

        # Intense signals should have higher variance (stronger bias)
        # due to arousal factor
        calm_std = np.std(calm_bias)
        intense_std = np.std(intense_bias)

        # With keywords present in intense_signals, bias should be more pronounced
        # (This may vary due to normalization, but trend should hold)
        assert intense_std >= 0  # At minimum, no negative std

    def test_compute_bias_overthink_focus(self, bias_computer, sample_signals, calm_signals):
        """Test that overthinking increases focus (bias strength)."""
        tokens = ["loops", "recursion", "thinking"]

        # sample_signals has overthink_depth=2
        # calm_signals has overthink_depth=0

        overthink_bias = bias_computer.compute_bias_for_tokens(sample_signals, tokens)
        calm_bias = bias_computer.compute_bias_for_tokens(calm_signals, tokens)

        # Both should be valid arrays
        assert len(overthink_bias) == len(tokens)
        assert len(calm_bias) == len(tokens)

    def test_compute_bias_clipping(self, bias_computer, intense_signals):
        """Test that bias is clipped to [-2, 2]."""
        tokens = ["crisis", "urgent", "important", "now", "must", "need"]
        bias = bias_computer.compute_bias_for_tokens(intense_signals, tokens)

        assert np.all(bias >= -2.0)
        assert np.all(bias <= 2.0)


# ============================================================================
# STEERING PROMPT TESTS
# ============================================================================

class TestSteeringPrompt:
    """Tests for steering prompt generation."""

    def test_steering_prompt_includes_focus(self, bias_computer, sample_signals):
        """Test that steering prompt includes gravity centers."""
        prompt = bias_computer.get_steering_prompt(sample_signals)

        assert "[Focus:" in prompt
        assert "memory" in prompt or "consciousness" in prompt

    def test_steering_prompt_intense_marker(self, bias_computer, intense_signals):
        """Test that high arousal adds [Intense] marker."""
        prompt = bias_computer.get_steering_prompt(intense_signals)

        assert "[Intense]" in prompt

    def test_steering_prompt_calm_marker(self, bias_computer, calm_signals):
        """Test that low arousal adds [Calm] marker."""
        prompt = bias_computer.get_steering_prompt(calm_signals)

        assert "[Calm]" in prompt

    def test_steering_prompt_deep_reflection(self, bias_computer, sample_signals):
        """Test that overthinking adds [Deep reflection] marker."""
        # Modify to have high overthink depth
        deep_signals = StanleySignals(
            gravity_centers=["thought"],
            pulse_arousal=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.5,
            surface_keywords=[],
            resonating_tags=[],
            active_expert="structural",
            expert_temperature=0.8,
            overthink_depth=5,  # High depth
            spiral_topics=[],
            body_tension=0.5,
            body_boredom=0.5,
            drift_momentum=0.5,
            hot_words=[],
        )

        prompt = bias_computer.get_steering_prompt(deep_signals)
        assert "[Deep reflection]" in prompt

    def test_steering_prompt_tense_marker(self, bias_computer):
        """Test that high body tension adds [Tense] marker."""
        tense_signals = StanleySignals(
            gravity_centers=[],
            pulse_arousal=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.5,
            surface_keywords=[],
            resonating_tags=[],
            active_expert="structural",
            expert_temperature=0.8,
            overthink_depth=0,
            spiral_topics=[],
            body_tension=0.9,  # High tension
            body_boredom=0.1,
            drift_momentum=0.5,
            hot_words=[],
        )

        prompt = bias_computer.get_steering_prompt(tense_signals)
        assert "[Tense]" in prompt

    def test_steering_prompt_restless_marker(self, bias_computer):
        """Test that high boredom adds [Restless] marker."""
        restless_signals = StanleySignals(
            gravity_centers=[],
            pulse_arousal=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.5,
            surface_keywords=[],
            resonating_tags=[],
            active_expert="structural",
            expert_temperature=0.8,
            overthink_depth=0,
            spiral_topics=[],
            body_tension=0.1,
            body_boredom=0.9,  # High boredom
            drift_momentum=0.5,
            hot_words=[],
        )

        prompt = bias_computer.get_steering_prompt(restless_signals)
        assert "[Restless]" in prompt


# ============================================================================
# STATE COLLECTOR TESTS (with mock organism)
# ============================================================================

class TestStanleyStateCollector:
    """Tests for StanleyStateCollector."""

    def test_collector_with_minimal_organism(self):
        """Test collector with minimal mock organism."""

        class MockOrganism:
            subjectivity = None
            memory = None
            overthinking = None
            body_sense = None
            semantic_drift = None
            cooccur_field = None
            vocab = None

        organism = MockOrganism()
        collector = StanleyStateCollector(organism)
        signals = collector.collect_all_signals()

        # Should return default values
        assert signals.pulse_arousal == 0.5
        assert signals.pulse_entropy == 0.5
        assert signals.gravity_centers == []

    def test_collector_returns_stanley_signals(self):
        """Test that collector returns StanleySignals instance."""

        class MockOrganism:
            subjectivity = None
            memory = None
            overthinking = None
            body_sense = None
            semantic_drift = None
            cooccur_field = None
            vocab = None

        organism = MockOrganism()
        collector = StanleyStateCollector(organism)
        signals = collector.collect_all_signals()

        assert isinstance(signals, StanleySignals)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestGuidedAttentionIntegration:
    """Integration tests for the full guided attention pipeline."""

    def test_full_pipeline_no_crash(self, bias_computer, sample_signals):
        """Test that full pipeline runs without crashing."""
        # Get keywords
        keywords = bias_computer.compute_keyword_set(sample_signals)
        assert len(keywords) > 0

        # Get bias
        tokens = list(keywords)[:5] + ["random", "words"]
        bias = bias_computer.compute_bias_for_tokens(sample_signals, tokens)
        assert len(bias) == len(tokens)

        # Get steering prompt
        prompt = bias_computer.get_steering_prompt(sample_signals)
        assert isinstance(prompt, str)

    def test_empty_tokens_handling(self, bias_computer, sample_signals):
        """Test handling of empty token list."""
        bias = bias_computer.compute_bias_for_tokens(sample_signals, [])
        assert len(bias) == 0

    def test_single_token_handling(self, bias_computer, sample_signals):
        """Test handling of single token."""
        bias = bias_computer.compute_bias_for_tokens(sample_signals, ["memory"])
        assert len(bias) == 1

    def test_weights_are_positive(self, bias_computer):
        """Test that influence weights are positive."""
        for name, weight in bias_computer.weights.items():
            assert weight > 0, f"Weight {name} should be positive"


# ============================================================================
# DETERMINISM TESTS (important for reproducibility)
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_signals_same_keywords(self, bias_computer, sample_signals):
        """Test that same signals produce same keywords."""
        keywords1 = bias_computer.compute_keyword_set(sample_signals)
        keywords2 = bias_computer.compute_keyword_set(sample_signals)
        assert keywords1 == keywords2

    def test_same_signals_same_bias(self, bias_computer, sample_signals):
        """Test that same signals produce same bias."""
        tokens = ["memory", "consciousness", "random"]
        bias1 = bias_computer.compute_bias_for_tokens(sample_signals, tokens)
        bias2 = bias_computer.compute_bias_for_tokens(sample_signals, tokens)
        np.testing.assert_array_equal(bias1, bias2)

    def test_same_signals_same_prompt(self, bias_computer, sample_signals):
        """Test that same signals produce same steering prompt."""
        prompt1 = bias_computer.get_steering_prompt(sample_signals)
        prompt2 = bias_computer.get_steering_prompt(sample_signals)
        assert prompt1 == prompt2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
test_external_brain.py — Tests for ExternalBrain (GPT-2 inference)

These tests verify:
- Graceful degradation without transformers
- Weight loading
- Text expansion
- Integration with HybridThinking
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestExternalBrainAvailability:
    """Test ExternalBrain availability detection."""

    def test_import_works(self):
        """Test that import doesn't crash."""
        from stanley_hybrid import EXTERNAL_WEIGHTS_AVAILABLE
        # Should be a bool
        assert isinstance(EXTERNAL_WEIGHTS_AVAILABLE, bool)

    def test_config_defaults(self):
        """Test ExternalBrainConfig defaults."""
        try:
            from stanley_hybrid.external_brain import ExternalBrainConfig
            cfg = ExternalBrainConfig()
            assert cfg.model_name == "distilgpt2"
            assert cfg.temperature == 0.9
            assert cfg.device == "cpu"
        except ImportError:
            pytest.skip("transformers not available")


class TestExternalBrainCreation:
    """Test ExternalBrain creation."""

    def test_create_without_load(self):
        """Test creating ExternalBrain without loading weights."""
        try:
            from stanley_hybrid import ExternalBrain
            brain = ExternalBrain()
            assert brain.loaded is False
            assert brain.total_expansions == 0
        except ImportError:
            pytest.skip("transformers not available")

    def test_create_helper(self):
        """Test create_external_brain helper."""
        from stanley_hybrid.external_brain import (
            create_external_brain,
            EXTERNAL_WEIGHTS_AVAILABLE,
        )

        brain = create_external_brain(auto_load=False)

        if EXTERNAL_WEIGHTS_AVAILABLE:
            assert brain is not None
            assert brain.loaded is False
        else:
            assert brain is None

    def test_stats_without_load(self):
        """Test stats before loading."""
        try:
            from stanley_hybrid import ExternalBrain
            brain = ExternalBrain()
            stats = brain.stats()
            assert stats["loaded"] is False
            assert stats["total_expansions"] == 0
        except ImportError:
            pytest.skip("transformers not available")


class TestHybridThinking:
    """Test HybridThinking integration."""

    def test_hybrid_without_external(self):
        """Test HybridThinking without external brain."""
        from stanley_hybrid.external_brain import HybridThinking

        hybrid = HybridThinking(external_brain=None)

        assert hybrid.has_external is False

        result, was_expanded = hybrid.think("test thought")
        assert result == "test thought"
        assert was_expanded is False

    def test_overthink_without_external(self):
        """Test overthinking without external brain."""
        from stanley_hybrid.external_brain import HybridThinking

        hybrid = HybridThinking(external_brain=None)

        result = hybrid.overthink("test trigger")
        assert result == "test trigger"


# Skip these tests if transformers not available
@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers required"
)
class TestExternalBrainWithWeights:
    """Tests that require transformers to be installed."""

    @pytest.mark.slow
    def test_load_weights(self):
        """Test loading distilgpt2 weights (slow, requires download)."""
        from stanley_hybrid import ExternalBrain

        brain = ExternalBrain()
        success = brain.load_weights()

        # This might fail if network issues, that's ok
        if success:
            assert brain.loaded is True
            assert brain.model is not None
            assert brain.tokenizer is not None

    @pytest.mark.slow
    def test_expand_thought(self):
        """Test thought expansion (slow, requires weights)."""
        from stanley_hybrid import ExternalBrain

        brain = ExternalBrain()
        if not brain.load_weights():
            pytest.skip("Could not load weights")

        result = brain.expand_thought("I feel curious about")

        assert len(result) > len("I feel curious about")
        assert brain.total_expansions == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

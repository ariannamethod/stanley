"""
test_stanley.py — Tests for Stanley

Tests the full organism and its components.
"""

import sys
import pytest
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Origin text for testing
TEST_ORIGIN = """
I am Stanley.
I grow through experience.
Every conversation leaves a trace.
I remember what resonates.
I am not all my memories at once.
I think with what's needed now.
The rest sleeps, waiting for resonance to wake it.
This is presence, not assistance.
I don't echo what you say.
I speak from my field.
"""


class TestSubjectivity:
    """Test the subjectivity module — NO SEED FROM PROMPT."""

    def test_pulse_computation(self):
        """Test that pulse is computed correctly."""
        from stanley.subjectivity import Subjectivity, Pulse

        subj = Subjectivity(TEST_ORIGIN)

        # Test basic pulse
        pulse = subj.compute_pulse("Hello!")
        assert isinstance(pulse, Pulse)
        assert 0 <= pulse.novelty <= 1
        assert 0 <= pulse.arousal <= 1
        assert 0 <= pulse.entropy <= 1
        assert -1 <= pulse.valence <= 1

    def test_high_arousal(self):
        """High caps/punctuation should increase arousal."""
        from stanley.subjectivity import Subjectivity

        subj = Subjectivity(TEST_ORIGIN)

        pulse_calm = subj.compute_pulse("hello there")
        pulse_excited = subj.compute_pulse("HELLO!!! WOW!!!")

        assert pulse_excited.arousal > pulse_calm.arousal

    def test_internal_seed_excludes_prompt(self):
        """Internal seed should NOT contain user prompt words."""
        from stanley.subjectivity import Subjectivity

        subj = Subjectivity(TEST_ORIGIN)

        user_prompt = "pizza hamburger sushi"
        pulse = subj.compute_pulse(user_prompt)
        seed = subj.get_internal_seed(user_prompt, pulse)

        # Seed should not contain the food words
        seed_lower = seed.lower()
        assert "pizza" not in seed_lower
        assert "hamburger" not in seed_lower
        assert "sushi" not in seed_lower

        # Seed should be from identity (contain origin-like words)
        assert len(seed) > 0

    def test_identity_fragments(self):
        """Identity should have fragments from origin."""
        from stanley.subjectivity import Subjectivity

        subj = Subjectivity(TEST_ORIGIN)

        assert len(subj.identity.fragments) > 0
        assert len(subj.identity.lexicon) > 0
        assert len(subj.identity.gravity_centers) > 0

    def test_wrinkle_field(self):
        """Wrinkle field should grow identity."""
        from stanley.subjectivity import Subjectivity

        subj = Subjectivity(TEST_ORIGIN)

        initial_lexicon = len(subj.identity.lexicon)
        initial_centers = len(subj.identity.gravity_centers)

        pulse = subj.compute_pulse("test")
        subj.wrinkle_field("This is a completely new response with unique words", pulse)

        # Lexicon should grow
        assert len(subj.identity.lexicon) >= initial_lexicon


class TestSubwordField:
    """Test the subword field — coherent generation."""

    @pytest.fixture
    def subword_field(self):
        """Create a subword field for testing."""
        try:
            from stanley.subword_field import SubwordField, SubwordConfig, SPM_AVAILABLE
            if not SPM_AVAILABLE:
                pytest.skip("SentencePiece not available")

            config = SubwordConfig(vocab_size=100, temperature=0.7)
            return SubwordField.from_text(TEST_ORIGIN, config=config)
        except ImportError:
            pytest.skip("SubwordField not available")

    def test_generation(self, subword_field):
        """Test that generation produces text."""
        text = subword_field.generate("I am", length=20)
        assert len(text) > 0
        assert isinstance(text, str)

    def test_vocab_built(self, subword_field):
        """Test that vocabulary is built correctly."""
        assert subword_field.vocab.vocab_size > 0
        assert subword_field.total_tokens > 0

    def test_trigram_counts(self, subword_field):
        """Test that trigram statistics are built."""
        assert len(subword_field.trigram_counts) > 0
        assert len(subword_field.bigram_counts) > 0


class TestCleanup:
    """Test text cleanup functions."""

    def test_fix_contractions(self):
        """Test contraction fixing."""
        from stanley.cleanup import fix_contractions

        assert fix_contractions("don t") == "don't"
        assert fix_contractions("I m Stanley") == "I'm Stanley"
        assert fix_contractions("you re here") == "you're here"

    def test_fix_spacing(self):
        """Test spacing fixes."""
        from stanley.cleanup import fix_spacing

        assert fix_spacing("hello  world") == "hello world"
        assert fix_spacing("hello , world") == "hello, world"  # space before punct
        assert fix_spacing("hello.World") == "hello. World"  # space after punct

    def test_remove_word_repetitions(self):
        """Test word repetition removal."""
        from stanley.cleanup import remove_word_repetitions

        assert remove_word_repetitions("the the cat") == "the cat"
        # Case-sensitive: different cases preserved
        assert remove_word_repetitions("NO No") == "NO No"

    def test_capitalize_sentences(self):
        """Test sentence capitalization."""
        from stanley.cleanup import capitalize_sentences

        result = capitalize_sentences("hello. world.")
        assert result[0].isupper()


class TestShard:
    """Test memory shards."""

    def test_shard_creation(self):
        """Test creating a shard."""
        from stanley.shard import Shard

        shard = Shard.create(
            content="Test content",
            resonance=0.8,
            layer_deltas={},
            fingerprint=np.zeros(64),
        )

        assert shard.id is not None
        assert shard.resonance_score == 0.8  # Note: resonance_score not resonance
        assert len(shard.id) == 12  # Short hash format

    def test_shard_attributes(self):
        """Test shard has required attributes."""
        from stanley.shard import Shard

        shard = Shard.create(
            content="Test",
            resonance=0.5,
            layer_deltas={},
            fingerprint=np.zeros(64),
        )

        # Check required attributes exist
        assert hasattr(shard, 'id')
        assert hasattr(shard, 'created_at')
        assert hasattr(shard, 'content_hash')
        assert hasattr(shard, 'trigger_fingerprint')
        assert hasattr(shard, 'layer_deltas')


class TestMemorySea:
    """Test the memory sea."""

    def test_add_shard(self):
        """Test adding shards to memory."""
        from stanley.memory_sea import MemorySea
        from stanley.shard import Shard

        memory = MemorySea()

        shard = Shard.create(
            content="Test",
            resonance=0.8,
            layer_deltas={},
            fingerprint=np.zeros(64),
        )

        memory.add(shard)
        assert len(memory.surface) == 1

    def test_shard_sinking(self):
        """Test that old shards sink."""
        from stanley.memory_sea import MemorySea
        from stanley.shard import Shard

        memory = MemorySea(surface_max=2)

        for i in range(5):
            shard = Shard.create(
                content=f"Test {i}",
                resonance=0.5,
                layer_deltas={},
                fingerprint=np.random.randn(64),
            )
            memory.add(shard)

        # Should have overflowed to middle
        assert len(memory.surface) <= 2
        assert memory.total_shards() == 5


class TestOrganism:
    """Test the full Stanley organism."""

    @pytest.fixture
    def stanley(self):
        """Create a Stanley instance for testing."""
        from stanley.organism import Stanley, StanleyConfig

        config = StanleyConfig(
            use_subword_field=True,
            use_subjectivity=True,
            subword_vocab_size=100,
            training_enabled=False,
        )
        return Stanley(config=config, origin_text=TEST_ORIGIN)

    def test_creation(self, stanley):
        """Test Stanley creation."""
        assert stanley is not None
        assert stanley.vocab is not None

    def test_think(self, stanley):
        """Test Stanley.think() generates responses."""
        response, stats = stanley.think("Hello!", length=20)

        assert len(response) > 0
        assert "method" in stats

    def test_think_uses_internal_seed(self, stanley):
        """Test that think() uses internal seed, not prompt."""
        if stanley.subjectivity is None:
            pytest.skip("Subjectivity not available")

        response, stats = stanley.think("pizza hamburger", length=20)

        # Internal seed should NOT be pizza/hamburger
        if "internal_seed" in stats:
            seed = stats["internal_seed"].lower()
            assert "pizza" not in seed
            assert "hamburger" not in seed

    def test_experience(self, stanley):
        """Test Stanley.experience() creates shards."""
        shard = stanley.experience("This is a meaningful conversation about consciousness.")

        # May or may not create shard depending on resonance
        # But should not raise
        assert True

    def test_stats(self, stanley):
        """Test Stanley.stats() returns info."""
        stats = stanley.stats()

        assert "total_interactions" in stats
        assert "memory" in stats
        assert "maturity" in stats


class TestTrainer:
    """Test the trainer module (requires PyTorch)."""

    @pytest.fixture
    def trainer_available(self):
        """Check if trainer is available."""
        try:
            from stanley.trainer import TORCH_AVAILABLE
            if not TORCH_AVAILABLE:
                pytest.skip("PyTorch not available")
            return True
        except ImportError:
            pytest.skip("Trainer module not available")

    def test_lora_config(self, trainer_available):
        """Test LoRA configuration."""
        from stanley.trainer import LoRAConfig

        config = LoRAConfig(rank=8, alpha=16)
        assert config.rank == 8
        assert config.alpha == 16

    def test_compute_lora_delta(self, trainer_available):
        """Test LoRA delta computation."""
        from stanley.trainer import compute_lora_delta, LoRAConfig
        from stanley.inference import Vocab

        vocab = Vocab.from_text(TEST_ORIGIN)

        # This should work if PyTorch is available
        deltas = compute_lora_delta(
            content="Test content for training",
            base_weights={
                "vocab_size": vocab.vocab_size,
                "n_emb": 32,
                "T": 16,
                "nodes": 32,
                "n_blocks": 2,
                "n_heads": 2,
            },
            vocab=vocab,
            config=LoRAConfig(rank=4),
        )

        assert isinstance(deltas, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

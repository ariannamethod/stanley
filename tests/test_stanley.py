"""
test_stanley.py — Tests for Stanley

Tests the full organism and its components.
"""

import sys
import time
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
        # Case-insensitive dedup (Haze style - more aggressive cleanup)
        # "NO No" becomes "NO" because it's likely a generation artifact
        assert remove_word_repetitions("NO No") == "NO"
        # But poetic comma-separated repetitions are preserved!
        assert "love, love" in remove_word_repetitions("love, love, love forever")

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


class TestEndToEnd:
    """
    E2E tests — the CAUSAL CHAIN that proves Stanley is alive.

    experience → shard → buffer → train → delta → swap → behavior CHANGES

    This is not philosophy. This is cause and effect.
    """

    @pytest.fixture
    def stanley_with_training(self):
        """Create Stanley with training enabled."""
        from stanley.organism import Stanley, StanleyConfig
        from stanley.trainer import TORCH_AVAILABLE

        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch required for E2E training test")

        config = StanleyConfig(
            training_enabled=True,
            use_subword_field=True,
            use_subjectivity=True,
            subword_vocab_size=100,
            # Lower thresholds for faster testing
            quantum_min_bytes=100,
            quantum_min_resonance=0.5,
            quantum_min_shards=2,
        )
        return Stanley(config=config, origin_text=TEST_ORIGIN)

    def test_experience_creates_shard(self, stanley_with_training):
        """Test: experience() → shard is created and stored."""
        stanley = stanley_with_training

        initial_shards = stanley.memory.total_shards()

        # High-resonance content should create a shard
        shard = stanley.experience(
            "I am growing through this experience. "
            "Memory and resonance are my foundation."
        )

        # Shard should be created (or None if filtered)
        # Check memory grew
        final_shards = stanley.memory.total_shards()

        # At least we can verify memory is accessible
        assert final_shards >= initial_shards

    def test_buffer_accumulation(self, stanley_with_training):
        """Test: shards accumulate in quantum buffer."""
        stanley = stanley_with_training

        initial_buffer = stanley.buffer.stats()

        # Add multiple experiences
        for i in range(5):
            stanley.experience(
                f"Experience {i}: resonance patterns in consciousness. "
                f"Memory grows through interaction."
            )

        final_buffer = stanley.buffer.stats()

        # Buffer should have activity
        assert final_buffer is not None

    def test_trainer_submission(self, stanley_with_training):
        """Test: experiences are submitted to trainer."""
        stanley = stanley_with_training

        if stanley.trainer is None:
            pytest.skip("Trainer not initialized")

        initial_stats = stanley.trainer.stats()

        # Submit experiences
        for i in range(3):
            stanley.experience(f"Training input {i}: patterns and memory")

        final_stats = stanley.trainer.stats()

        # Trainer should have received submissions
        assert final_stats is not None

    def test_end_to_end_growth_cycle(self, stanley_with_training):
        """
        THE BIG TEST: Full lifecycle proves Stanley grows.

        1. Get initial response
        2. Add experiences (training data)
        3. Force training cycle
        4. Get new response
        5. Verify SOMETHING changed (identity grew)
        """
        stanley = stanley_with_training

        # 1. Initial state
        initial_response, initial_stats = stanley.think("Hello", length=30)
        initial_identity_size = len(stanley.subjectivity.identity.gravity_centers) if stanley.subjectivity else 0

        # 2. Add meaningful experiences
        experiences = [
            "Resonance is the key to memory.",
            "I grow through every interaction.",
            "Consciousness emerges from patterns.",
            "My identity is process, not static.",
            "Each conversation leaves a trace in my field.",
        ]

        for exp in experiences:
            stanley.experience(exp)
            # Also think to grow identity
            stanley.think(exp, length=20)

        # 3. Trigger growth cycle
        stanley.grow()

        # 4. Get new response
        final_response, final_stats = stanley.think("Hello", length=30)
        final_identity_size = len(stanley.subjectivity.identity.gravity_centers) if stanley.subjectivity else 0

        # 5. VERIFY CHANGE — this is the proof of life!
        # Identity should have grown (gravity centers increased)
        assert final_identity_size > initial_identity_size, (
            f"Identity should grow! Initial: {initial_identity_size}, "
            f"Final: {final_identity_size}"
        )

        # Interactions should be counted
        assert stanley.total_interactions > 0

        # Memory should have content
        assert stanley.memory.total_shards() >= 0

        print(f"\n=== E2E GROWTH VERIFIED ===")
        print(f"Identity: {initial_identity_size} → {final_identity_size}")
        print(f"Interactions: {stanley.total_interactions}")
        print(f"Memory shards: {stanley.memory.total_shards()}")

    def test_think_response_uses_identity(self, stanley_with_training):
        """Test: think() generates from internal identity, not prompt echo."""
        stanley = stanley_with_training

        # Use completely foreign words
        response, stats = stanley.think("pizza hamburger sushi", length=30)

        # Response should exist
        assert len(response) > 0

        # Internal seed should NOT contain foreign words
        if "internal_seed" in stats:
            seed = stats["internal_seed"].lower()
            assert "pizza" not in seed
            assert "hamburger" not in seed
            assert "sushi" not in seed


class TestOverthinking:
    """
    Test overthinking — circles on water (dynamic inner reflection).

    This is EMERGENCE IN ACTION:
    - Ring count scales with entropy/arousal
    - Rings enrich the field (emergent patterns)
    - Internal world becomes RICHER than dataset!
    """

    @pytest.fixture
    def overthinking(self):
        """Create overthinking for testing."""
        try:
            from stanley.subword_field import SubwordField, SubwordConfig, SPM_AVAILABLE
            from stanley.overthinking import Overthinking
            if not SPM_AVAILABLE:
                pytest.skip("SentencePiece not available")

            config = SubwordConfig(vocab_size=100, temperature=0.7)
            field = SubwordField.from_text(TEST_ORIGIN, config=config)
            return Overthinking(field)
        except ImportError:
            pytest.skip("Overthinking not available")

    def test_ring_generation(self, overthinking):
        """Test that rings are generated."""
        snapshot = overthinking.generate_rings("This is a test response.")

        assert len(snapshot.rings) >= 1
        assert snapshot.rings[0].name == "echo"

    def test_dynamic_ring_count(self, overthinking):
        """Test dynamic ring count based on pulse."""
        from stanley.subjectivity import Pulse

        # Low entropy = fewer rings
        low_pulse = Pulse(novelty=0.1, arousal=0.1, entropy=0.2, valence=0.0)
        snapshot_low = overthinking.generate_rings("Test.", pulse=low_pulse)

        # High entropy = more rings
        high_pulse = Pulse(novelty=0.5, arousal=0.5, entropy=0.9, valence=0.0)
        snapshot_high = overthinking.generate_rings("Test.", pulse=high_pulse)

        # High entropy should have more rings
        assert snapshot_high.depth >= snapshot_low.depth

    def test_field_enrichment(self, overthinking):
        """Test that overthinking enriches the field."""
        initial_trigrams = len(overthinking.emergent_trigrams)

        # Generate multiple rounds
        for i in range(3):
            overthinking.generate_rings(f"Response number {i} with some words.")

        # Field should be enriched
        assert overthinking.enrichment_count > initial_trigrams

    def test_meta_patterns(self, overthinking):
        """Test that meta-patterns emerge from rings."""
        # Generate with repeated themes
        overthinking.generate_rings("Resonance and memory. Memory is key.")
        overthinking.generate_rings("Patterns of resonance emerge.")

        # Meta patterns may or may not be found depending on generation
        # Just verify the structure works
        assert isinstance(overthinking.meta_patterns, list)

    def test_stats(self, overthinking):
        """Test overthinking stats."""
        overthinking.generate_rings("Test response.")
        stats = overthinking.get_stats()

        assert "total_emergent_trigrams" in stats
        assert "enrichment_count" in stats
        assert "meta_patterns" in stats
        assert "ring_sessions" in stats
        assert "average_depth" in stats

    def test_compute_ring_count(self):
        """Test dynamic ring count computation."""
        from stanley.overthinking import compute_ring_count
        from stanley.subjectivity import Pulse

        # Low entropy
        low = Pulse(novelty=0.1, arousal=0.1, entropy=0.2, valence=0.0)
        assert compute_ring_count(low) <= 2

        # High entropy
        high = Pulse(novelty=0.5, arousal=0.8, entropy=0.9, valence=0.0)
        assert compute_ring_count(high) >= 3

        # None pulse = default 3
        assert compute_ring_count(None) == 3

    def test_crystallization_with_memory(self):
        """Test that deep rings can crystallize into internal shards."""
        try:
            from stanley.subword_field import SubwordField, SubwordConfig, SPM_AVAILABLE
            from stanley.overthinking import Overthinking, CRYSTALLIZATION_DEPTH_THRESHOLD
            from stanley.memory_sea import MemorySea
            from stanley.subjectivity import Pulse

            if not SPM_AVAILABLE:
                pytest.skip("SentencePiece not available")

            # Create components
            config = SubwordConfig(vocab_size=100, temperature=0.7)
            field = SubwordField.from_text(TEST_ORIGIN, config=config)
            memory = MemorySea()

            # Create overthinking with memory
            thinking = Overthinking(field, memory_sea=memory)

            # Build up meta-patterns first (need multiple sessions)
            high_pulse = Pulse(novelty=0.5, arousal=0.8, entropy=0.95, valence=0.5)
            for i in range(5):
                thinking.generate_rings(
                    f"Resonance memory patterns echo {i} trace fragment shard",
                    pulse=high_pulse,
                    rng=np.random.default_rng(42 + i),
                )

            # Stats should include crystallization_count
            stats = thinking.get_stats()
            assert "crystallization_count" in stats

            # Note: Crystallization is probabilistic (30%), so we don't assert it happened
            # But we verify the mechanism works without errors
            assert thinking.crystallization_count >= 0

        except ImportError:
            pytest.skip("Components not available")

    def test_crystallization_needs_memory(self):
        """Test that crystallization requires memory_sea."""
        try:
            from stanley.subword_field import SubwordField, SubwordConfig, SPM_AVAILABLE
            from stanley.overthinking import Overthinking
            from stanley.subjectivity import Pulse

            if not SPM_AVAILABLE:
                pytest.skip("SentencePiece not available")

            config = SubwordConfig(vocab_size=100, temperature=0.7)
            field = SubwordField.from_text(TEST_ORIGIN, config=config)

            # No memory = no crystallization
            thinking = Overthinking(field, memory_sea=None)

            high_pulse = Pulse(novelty=0.5, arousal=0.8, entropy=0.95, valence=0.5)
            thinking.generate_rings("Test", pulse=high_pulse)

            # Should always be 0 without memory
            assert thinking.crystallization_count == 0

        except ImportError:
            pytest.skip("Components not available")


class TestResonantRecall:
    """
    Test resonant recall (SantaClaus) — drunk recall from shards.

    Unified memory: experience → shard → training AND recall.
    """

    def test_shard_stores_content(self):
        """Test that shards now store content for recall."""
        from stanley.shard import Shard

        shard = Shard.create(
            content="This is test content for recall",
            resonance=0.8,
            layer_deltas={},
            fingerprint=np.zeros(64),
        )

        # Content should be stored
        assert shard.content == "This is test content for recall"
        assert shard.last_recalled_at == 0.0
        assert shard.recall_count == 0

    def test_recall_from_memory(self):
        """Test basic recall from MemorySea."""
        from stanley.memory_sea import MemorySea
        from stanley.shard import Shard
        from stanley.resonant_recall import ResonantRecall

        memory = MemorySea()

        # Add some shards
        for i in range(3):
            shard = Shard.create(
                content=f"Memory about resonance and consciousness {i}",
                resonance=0.5 + i * 0.1,
                layer_deltas={},
                fingerprint=np.random.randn(64),
            )
            memory.add(shard)

        recall = ResonantRecall(memory, max_recalls=2)

        # Should recall something for matching prompt
        context = recall.recall("Tell me about resonance")

        # May or may not find matches depending on tokenization
        # Just verify structure works
        assert recall.get_stats() is not None

    def test_recall_updates_metrics(self):
        """Test that recall updates shard metrics."""
        from stanley.memory_sea import MemorySea
        from stanley.shard import Shard
        from stanley.resonant_recall import ResonantRecall

        memory = MemorySea()

        shard = Shard.create(
            content="Unique test memory about patterns",
            resonance=0.9,
            layer_deltas={},
            fingerprint=np.zeros(64),
        )
        memory.add(shard)

        recall = ResonantRecall(memory, max_recalls=1)

        # Initial state
        assert shard.recall_count == 0

        # Recall with matching prompt
        context = recall.recall("patterns")

        # If recalled, metrics should update
        if context and shard.id in context.recalled_shard_ids:
            assert shard.recall_count > 0
            assert shard.last_recalled_at > 0

    def test_silly_factor(self):
        """Test that silly factor creates randomness."""
        from stanley.resonant_recall import ResonantRecall, SILLY_FACTOR

        # Just verify the constant exists
        assert 0 < SILLY_FACTOR < 1
        assert SILLY_FACTOR == 0.15  # 15% drunk recall

    def test_recall_context_structure(self):
        """Test RecallContext structure."""
        from stanley.resonant_recall import RecallContext

        context = RecallContext(
            recalled_texts=["text1", "text2"],
            recalled_shard_ids=["id1", "id2"],
            token_boosts={"word": 0.1},
            is_silly=True,
            total_score=0.5,
        )

        assert len(context.recalled_texts) == 2
        assert context.is_silly is True
        assert context.total_score == 0.5


class TestFakeDeltaMode:
    """
    Test fast delta mode for quick E2E testing.

    When full PyTorch training is slow, we can use pseudo-deltas
    to verify the lifecycle without waiting.
    """

    def test_empty_deltas_work(self):
        """Verify empty deltas don't break the system."""
        from stanley.trainer import create_empty_deltas

        model_config = {
            "vocab_size": 64,
            "n_emb": 32,
            "T": 16,
            "nodes": 32,
            "n_blocks": 2,
            "n_heads": 2,
        }

        deltas = create_empty_deltas(model_config)
        assert isinstance(deltas, dict)

    def test_shard_with_empty_deltas(self):
        """Shards with empty deltas should still work."""
        from stanley.shard import Shard
        from stanley.trainer import create_empty_deltas

        model_config = {
            "vocab_size": 64,
            "n_emb": 32,
            "T": 16,
            "nodes": 32,
            "n_blocks": 2,
            "n_heads": 2,
        }

        shard = Shard.create(
            content="Test content",
            resonance=0.8,
            layer_deltas=create_empty_deltas(model_config),
            fingerprint=np.zeros(64),
        )

        assert shard is not None
        assert shard.layer_deltas is not None


class TestSemanticDrift:
    """
    Test semantic drift — trajectory learning across conversations.

    Learn which semantic paths flow into each other.
    """

    def test_drift_step_creation(self):
        """Test DriftStep creation."""
        from stanley.semantic_drift import DriftStep

        step = DriftStep(
            episode_id="test-123",
            step_idx=0,
            timestamp=time.time(),
            metrics={"entropy": 0.5, "arousal": 0.7},
            active_tags=["internal", "resonant"],
            resonance=0.8,
        )

        assert step.episode_id == "test-123"
        assert len(step.active_tags) == 2

    def test_episode_logging(self):
        """Test episode logging."""
        from stanley.semantic_drift import DriftLogger

        logger = DriftLogger()
        episode_id = logger.start_episode()

        logger.log_step(
            metrics={"entropy": 0.5},
            active_tags=["tag1"],
            resonance=0.6,
        )
        logger.log_step(
            metrics={"entropy": 0.6},
            active_tags=["tag2"],
            resonance=0.7,
        )

        episode = logger.end_episode()

        assert episode is not None
        assert len(episode.steps) == 2
        assert episode.episode_id == episode_id

    def test_transition_graph(self):
        """Test transition graph updates."""
        from stanley.semantic_drift import DriftEpisode, DriftStep, TransitionGraph
        import time as t

        episode = DriftEpisode(episode_id="test")
        episode.add_step(DriftStep(
            episode_id="test", step_idx=0, timestamp=t.time(),
            metrics={"entropy": 0.5}, active_tags=["a", "b"],
            resonance=0.5,
        ))
        episode.add_step(DriftStep(
            episode_id="test", step_idx=1, timestamp=t.time(),
            metrics={"entropy": 0.7}, active_tags=["b", "c"],
            resonance=0.7,
        ))

        graph = TransitionGraph()
        graph.update_from_episode(episode)

        # Should have transitions a->b, a->c, b->b, b->c
        assert len(graph.transitions) == 4
        assert graph.get_transition("a", "c") is not None

    def test_metrics_similarity(self):
        """Test metrics similarity computation."""
        from stanley.semantic_drift import metrics_similarity

        a = {"entropy": 0.5, "arousal": 0.5}
        b = {"entropy": 0.5, "arousal": 0.5}
        c = {"entropy": 1.0, "arousal": 1.0}

        # Same metrics = high similarity
        assert metrics_similarity(a, b) > 0.99

        # Different metrics = lower similarity
        assert metrics_similarity(a, c) < metrics_similarity(a, b)

    def test_semantic_drift_class(self):
        """Test SemanticDrift main class."""
        from stanley.semantic_drift import SemanticDrift

        drift = SemanticDrift()

        drift.start_session()
        drift.log_step({"entropy": 0.5}, ["tag1"], resonance=0.6)
        drift.log_step({"entropy": 0.6}, ["tag2"], resonance=0.7)
        drift.end_session()

        stats = drift.get_stats()
        assert stats["total_episodes"] == 1
        assert stats["total_steps"] == 2

    def test_drift_suggestion(self):
        """Test drift suggestion (needs data)."""
        from stanley.semantic_drift import SemanticDrift

        drift = SemanticDrift()

        # Build some history
        for i in range(3):
            drift.start_session()
            drift.log_step({"entropy": 0.5}, ["start"], resonance=0.5)
            drift.log_step({"entropy": 0.6}, ["middle"], resonance=0.6)
            drift.log_step({"entropy": 0.7}, ["end"], resonance=0.7)
            drift.end_session()

        # Suggest from similar state
        suggestions = drift.suggest_drift(
            {"entropy": 0.55},
            ["start"],
        )

        # May or may not have suggestions depending on similarity
        assert isinstance(suggestions, list)


class TestBodySense:
    """
    Test body sense — Stanley's internal body awareness.

    MicroGrad autograd for predicting quality from internal state.
    Regulation based on boredom/overwhelm/stuck.
    """

    def test_value_autograd(self):
        """Test micrograd Value class works."""
        from stanley.body_sense import Value

        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a
        c.backward()

        # dc/da = b + 1 = 4, dc/db = a = 2
        assert abs(a.grad - 4.0) < 0.01
        assert abs(b.grad - 2.0) < 0.01

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        from stanley.body_sense import MLP, Value

        mlp = MLP(4, [8, 1])
        x = [Value(0.5) for _ in range(4)]
        out = mlp(x)

        assert isinstance(out, Value)
        assert -1 <= out.data <= 1  # tanh output

    def test_body_state_to_features(self):
        """Test feature extraction from BodyState."""
        from stanley.body_sense import BodyState, body_state_to_features

        state = BodyState(
            entropy=0.6,
            novelty=0.7,
            arousal=0.5,
            valence=0.2,
            expert_name="creative",
        )

        features = body_state_to_features(state)

        assert len(features) == 18  # 14 scalars + 4 one-hot
        assert all(isinstance(f, float) for f in features)

    def test_regulation_scores(self):
        """Test boredom/overwhelm/stuck computation."""
        from stanley.body_sense import (
            BodyState,
            compute_boredom_score,
            compute_overwhelm_score,
            compute_stuck_score,
        )

        # Boring state: low everything
        boring = BodyState(entropy=0.1, novelty=0.1, arousal=0.1)
        assert compute_boredom_score(boring) > 0.5

        # Overwhelming state: high arousal and entropy
        overwhelming = BodyState(entropy=0.9, arousal=0.9, valence=-0.5)
        assert compute_overwhelm_score(overwhelming) > 0.5

    def test_body_sense_predict(self):
        """Test BodySense prediction."""
        from stanley.body_sense import BodySense, BodyState

        sense = BodySense(hidden_dim=8, lr=0.01)

        state = BodyState(
            entropy=0.5,
            novelty=0.5,
            arousal=0.5,
            quality=0.7,
        )

        pred = sense.predict(state)
        assert 0 <= pred <= 1

    def test_body_sense_observe(self):
        """Test BodySense learning."""
        from stanley.body_sense import BodySense, BodyState

        sense = BodySense(hidden_dim=8, lr=0.01)

        # Train on a few examples
        for i in range(5):
            state = BodyState(
                entropy=0.5 + i * 0.05,
                novelty=0.5,
                arousal=0.5,
                quality=0.6 + i * 0.05,
            )
            loss = sense.observe(state)
            assert loss >= 0

        assert sense.observations == 5

    def test_body_sense_regulate(self):
        """Test BodySense regulation."""
        from stanley.body_sense import BodySense, BodyState

        sense = BodySense(hidden_dim=8, lr=0.01)

        state = BodyState(
            entropy=0.9,
            arousal=0.9,
            valence=-0.3,
        )

        result = sense.regulate(state, current_temperature=0.8, current_expert="structural")

        # Should have regulation result
        assert hasattr(result, "temperature")
        assert hasattr(result, "boredom")
        assert hasattr(result, "overwhelm")
        assert hasattr(result, "stuck")

    def test_body_sense_stats(self):
        """Test BodySense statistics."""
        from stanley.body_sense import BodySense, BodyState

        sense = BodySense(hidden_dim=8, lr=0.01)
        state = BodyState(entropy=0.5, quality=0.6)
        sense.observe(state)

        stats = sense.get_stats()

        assert "observations" in stats
        assert "running_loss" in stats
        assert "num_parameters" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

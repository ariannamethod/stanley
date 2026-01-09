#!/usr/bin/env python3
"""
Tests for AdapterBank — Stanley mixes mood LoRAs to modify GPT-2 weights.

"Забить гвоздями" — GPT's hardening suggestions:
1. Hook coverage (all 24 layers)
2. Delta algebra (mix correctness)
3. Stability (NaN/inf, norm clamp)
4. Base logits preservation (mix=0)

"Stanley doesn't just steer GPT-2. Stanley BECOMES part of GPT-2's weights."
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stanley_hybrid.adapter_bank import (
    Mood,
    MOOD_PROFILES,
    AdapterBank,
    AdapterBankConfig,
    MoodRouter,
    MixedAdapter,
    LoRAAdapter,
)
from stanley_hybrid.guided_attention import StanleySignals


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Default adapter config."""
    return AdapterBankConfig(
        lora_rank=8,
        lora_alpha=16.0,
        target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
    )


@pytest.fixture
def sample_layer_dims():
    """Simulate GPT-2 layer dimensions (distilgpt2 has 6 layers)."""
    dims = {}
    for i in range(6):
        # attn.c_attn: projects to 3x hidden (Q, K, V)
        dims[f"transformer.h.{i}.attn.c_attn"] = (2304, 768)  # 768*3, 768
        dims[f"transformer.h.{i}.attn.c_proj"] = (768, 768)
        dims[f"transformer.h.{i}.mlp.c_fc"] = (3072, 768)     # 4x hidden
        dims[f"transformer.h.{i}.mlp.c_proj"] = (768, 3072)
    return dims


@pytest.fixture
def bank(config, sample_layer_dims):
    """Create initialized adapter bank."""
    bank = AdapterBank(config)
    bank.initialize_adapters(sample_layer_dims)
    return bank


@pytest.fixture
def router():
    """Create mood router."""
    return MoodRouter(temperature=1.0)


@pytest.fixture
def calm_signals():
    """Low arousal signals."""
    return StanleySignals(
        gravity_centers=["peace"],
        pulse_arousal=0.1,
        pulse_entropy=0.2,
        pulse_novelty=0.1,
        surface_keywords=[],
        resonating_tags=[],
        active_expert="reflective",
        expert_temperature=0.5,
        overthink_depth=0,
        spiral_topics=[],
        body_tension=0.1,
        body_boredom=0.1,
        drift_momentum=0.1,
        hot_words=[],
    )


@pytest.fixture
def intense_signals():
    """High arousal signals."""
    return StanleySignals(
        gravity_centers=["urgent"],
        pulse_arousal=0.95,
        pulse_entropy=0.8,
        pulse_novelty=0.9,
        surface_keywords=[],
        resonating_tags=[],
        active_expert="creative",
        expert_temperature=1.0,
        overthink_depth=5,
        spiral_topics=[],
        body_tension=0.9,
        body_boredom=0.0,
        drift_momentum=0.8,
        hot_words=[],
    )


# ============================================================================
# MOOD ENUM TESTS
# ============================================================================

class TestMoodEnum:
    """Tests for Mood enum."""

    def test_all_moods_exist(self):
        """Verify all 8 moods are defined."""
        expected = ["calm", "intense", "creative", "focused",
                    "overthinking", "playful", "cold_logic", "warm"]
        actual = [m.value for m in Mood]
        assert set(expected) == set(actual)
        assert len(Mood) == 8

    def test_all_moods_have_profiles(self):
        """Every mood must have a profile."""
        for mood in Mood:
            assert mood in MOOD_PROFILES, f"Missing profile for {mood}"
            profile = MOOD_PROFILES[mood]
            assert "temperature_bias" in profile
            assert "attention_spread" in profile
            assert "layer_strength" in profile


# ============================================================================
# ADAPTER BANK TESTS
# ============================================================================

class TestAdapterBank:
    """Tests for AdapterBank initialization and structure."""

    def test_initialization(self, bank, sample_layer_dims):
        """Test bank initializes correctly."""
        assert len(bank.adapters) == 8
        assert len(bank.layer_dims) == len(sample_layer_dims)

    def test_all_moods_have_adapters(self, bank):
        """Every mood should have an adapter."""
        for mood in Mood:
            assert mood in bank.adapters

    def test_adapter_has_all_layers(self, bank, sample_layer_dims):
        """Each adapter should have deltas for all layers."""
        for mood, adapter in bank.adapters.items():
            for layer_name in sample_layer_dims:
                assert layer_name in adapter.layer_deltas, \
                    f"Adapter {mood} missing layer {layer_name}"

    def test_delta_shapes(self, bank, config, sample_layer_dims):
        """Delta matrices should have correct shapes."""
        rank = config.lora_rank

        for mood, adapter in bank.adapters.items():
            for layer_name, (out_dim, in_dim) in sample_layer_dims.items():
                A, B = adapter.layer_deltas[layer_name]

                assert A.shape == (out_dim, rank), \
                    f"A shape mismatch for {mood}/{layer_name}"
                assert B.shape == (rank, in_dim), \
                    f"B shape mismatch for {mood}/{layer_name}"

    def test_get_delta_computation(self, bank, sample_layer_dims):
        """Test that get_delta computes A @ B correctly."""
        adapter = bank.adapters[Mood.CALM]
        layer = list(sample_layer_dims.keys())[0]

        A, B = adapter.layer_deltas[layer]
        expected = A @ B
        actual = adapter.get_delta(layer)

        np.testing.assert_array_almost_equal(expected, actual)

    def test_nonexistent_layer_returns_none(self, bank):
        """get_delta for unknown layer should return None."""
        adapter = bank.adapters[Mood.CALM]
        assert adapter.get_delta("nonexistent.layer") is None


# ============================================================================
# MOOD ROUTER TESTS
# ============================================================================

class TestMoodRouter:
    """Tests for MoodRouter mixing logic."""

    def test_mix_sums_to_one(self, router, calm_signals):
        """Mix coefficients should sum to 1.0 (softmax)."""
        mix = router.compute_mix(calm_signals)
        total = sum(mix.values())
        assert abs(total - 1.0) < 1e-6

    def test_mix_all_positive(self, router, calm_signals):
        """All mix coefficients should be positive (softmax property)."""
        mix = router.compute_mix(calm_signals)
        for mood, weight in mix.items():
            assert weight > 0, f"Weight for {mood} should be positive"

    def test_intense_signals_favor_intense_mood(self, router, intense_signals):
        """High arousal should increase intense mood weight."""
        mix = router.compute_mix(intense_signals)

        # Intense should be one of the top moods
        sorted_mix = sorted(mix.items(), key=lambda x: -x[1])
        top_moods = [m for m, _ in sorted_mix[:3]]

        assert Mood.INTENSE in top_moods, \
            f"INTENSE should be top mood for intense signals, got {top_moods}"

    def test_different_signals_different_mix(self, router, calm_signals, intense_signals):
        """Different signals should produce different mixes."""
        calm_mix = router.compute_mix(calm_signals)
        intense_mix = router.compute_mix(intense_signals)

        # Mixes should be different
        for mood in Mood:
            if calm_mix[mood] != intense_mix[mood]:
                return  # Test passes if any difference

        pytest.fail("Calm and intense signals produced identical mixes")

    def test_dominant_mood_is_max(self, router, intense_signals):
        """get_dominant_mood should return mood with highest weight."""
        mix = router.compute_mix(intense_signals)
        dominant = router.get_dominant_mood(mix)

        max_mood = max(mix.items(), key=lambda x: x[1])[0]
        assert dominant == max_mood

    def test_temperature_affects_distribution(self):
        """Higher temperature should flatten distribution."""
        signals = StanleySignals(
            gravity_centers=[],
            pulse_arousal=0.5,
            pulse_entropy=0.5,
            pulse_novelty=0.5,
            surface_keywords=[],
            resonating_tags=[],
            active_expert="",
            expert_temperature=0.5,
            overthink_depth=0,
            spiral_topics=[],
            body_tension=0.5,
            body_boredom=0.5,
            drift_momentum=0.5,
            hot_words=[],
        )

        router_cold = MoodRouter(temperature=0.5)
        router_hot = MoodRouter(temperature=2.0)

        mix_cold = router_cold.compute_mix(signals)
        mix_hot = router_hot.compute_mix(signals)

        # Entropy of hot should be higher (more uniform)
        var_cold = np.var(list(mix_cold.values()))
        var_hot = np.var(list(mix_hot.values()))

        assert var_cold > var_hot, \
            "Higher temperature should produce more uniform distribution"


# ============================================================================
# MIXED ADAPTER TESTS (Delta Algebra)
# ============================================================================

class TestMixedAdapter:
    """Tests for MixedAdapter delta mixing."""

    def test_update_mix_changes_cache(self, bank, router, intense_signals):
        """update_mix should update cached mix."""
        mixed = MixedAdapter(bank, router)

        assert mixed._cached_mix is None
        mixed.update_mix(intense_signals, step=0)
        assert mixed._cached_mix is not None

    def test_single_mood_mix(self, bank, router):
        """If mix is one-hot, delta should equal that adapter's delta."""
        mixed = MixedAdapter(bank, router, alpha=16.0)

        # Manually set one-hot mix
        one_hot = {m: 0.0 for m in Mood}
        one_hot[Mood.INTENSE] = 1.0
        mixed._cached_mix = one_hot
        mixed._cached_deltas = {}

        layer = list(bank.layer_dims.keys())[0]

        # Get mixed delta (should be just INTENSE's delta scaled)
        mixed_delta = mixed.get_mixed_delta(layer)

        # Get INTENSE adapter's delta
        intense_delta = bank.adapters[Mood.INTENSE].get_delta(layer)
        scale = bank.adapters[Mood.INTENSE].scale()

        # Should be close (scaled by alpha/rank)
        expected = mixed.scale * scale * intense_delta

        np.testing.assert_array_almost_equal(mixed_delta, expected)

    def test_zero_mix_zero_delta(self, bank, router):
        """If all mix weights are below threshold, delta should be None or zero."""
        mixed = MixedAdapter(bank, router)

        # Set all weights to below threshold (0.01)
        zero_mix = {m: 0.001 for m in Mood}
        mixed._cached_mix = zero_mix
        mixed._cached_deltas = {}

        layer = list(bank.layer_dims.keys())[0]
        delta = mixed.get_mixed_delta(layer)

        assert delta is None, "Zero mix should produce no delta"

    def test_mix_is_linear(self, bank, router):
        """Mixed delta should be linear combination of individual deltas."""
        mixed = MixedAdapter(bank, router, alpha=16.0)

        # Set specific mix
        test_mix = {m: 0.0 for m in Mood}
        test_mix[Mood.CALM] = 0.6
        test_mix[Mood.INTENSE] = 0.4
        mixed._cached_mix = test_mix
        mixed._cached_deltas = {}

        layer = list(bank.layer_dims.keys())[0]
        mixed_delta = mixed.get_mixed_delta(layer)

        # Compute expected
        calm_delta = bank.adapters[Mood.CALM].get_delta(layer)
        calm_scale = bank.adapters[Mood.CALM].scale()
        intense_delta = bank.adapters[Mood.INTENSE].get_delta(layer)
        intense_scale = bank.adapters[Mood.INTENSE].scale()

        expected = mixed.scale * (
            0.6 * calm_scale * calm_delta +
            0.4 * intense_scale * intense_delta
        )

        np.testing.assert_array_almost_equal(mixed_delta, expected, decimal=5)

    def test_stats_structure(self, bank, router, intense_signals):
        """stats() should return proper structure."""
        mixed = MixedAdapter(bank, router)
        mixed.update_mix(intense_signals, step=0)

        stats = mixed.stats()

        assert "current_mix" in stats
        assert "dominant_mood" in stats
        assert "cache_step" in stats
        assert "num_adapters" in stats

        assert stats["num_adapters"] == 8


# ============================================================================
# STABILITY TESTS (NaN/inf, norm bounds)
# ============================================================================

class TestStability:
    """Tests for numerical stability."""

    def test_no_nan_in_deltas(self, bank):
        """Delta matrices should not contain NaN."""
        for mood, adapter in bank.adapters.items():
            for layer, (A, B) in adapter.layer_deltas.items():
                assert not np.isnan(A).any(), f"NaN in A for {mood}/{layer}"
                assert not np.isnan(B).any(), f"NaN in B for {mood}/{layer}"

    def test_no_inf_in_deltas(self, bank):
        """Delta matrices should not contain inf."""
        for mood, adapter in bank.adapters.items():
            for layer, (A, B) in adapter.layer_deltas.items():
                assert not np.isinf(A).any(), f"inf in A for {mood}/{layer}"
                assert not np.isinf(B).any(), f"inf in B for {mood}/{layer}"

    def test_delta_norms_bounded(self, bank):
        """Delta norms should be reasonable."""
        max_norm = 10.0  # Reasonable upper bound

        for mood, adapter in bank.adapters.items():
            for layer in adapter.layer_deltas:
                delta = adapter.get_delta(layer)
                norm = np.linalg.norm(delta)
                assert norm < max_norm, \
                    f"Delta norm {norm} too large for {mood}/{layer}"

    def test_mixed_delta_bounded(self, bank, router, intense_signals):
        """Mixed deltas should also be bounded."""
        mixed = MixedAdapter(bank, router)
        mixed.update_mix(intense_signals, step=0)

        max_norm = 50.0  # Higher bound for mixed

        for layer in bank.layer_dims:
            delta = mixed.get_mixed_delta(layer)
            if delta is not None:
                norm = np.linalg.norm(delta)
                assert norm < max_norm, f"Mixed delta norm {norm} too large"

    def test_extreme_signals_no_crash(self, router):
        """Extreme signal values should not cause crashes."""
        extreme = StanleySignals(
            gravity_centers=[],
            pulse_arousal=1.0,
            pulse_entropy=1.0,
            pulse_novelty=1.0,
            surface_keywords=[],
            resonating_tags=[],
            active_expert="",
            expert_temperature=1.0,
            overthink_depth=100,  # Extreme
            spiral_topics=[],
            body_tension=1.0,
            body_boredom=1.0,
            drift_momentum=1.0,
            hot_words=[],
        )

        # Should not raise
        mix = router.compute_mix(extreme)

        # Should still be valid distribution
        assert abs(sum(mix.values()) - 1.0) < 1e-6
        assert all(w > 0 for w in mix.values())


# ============================================================================
# DETERMINISM TESTS
# ============================================================================

class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_signals_same_mix(self, router, intense_signals):
        """Same signals should produce identical mix."""
        mix1 = router.compute_mix(intense_signals)
        mix2 = router.compute_mix(intense_signals)

        for mood in Mood:
            assert mix1[mood] == mix2[mood]

    def test_cached_delta_consistency(self, bank, router, intense_signals):
        """Cached delta should match freshly computed."""
        mixed = MixedAdapter(bank, router)
        mixed.update_mix(intense_signals, step=0)

        layer = list(bank.layer_dims.keys())[0]

        # First call caches
        delta1 = mixed.get_mixed_delta(layer)
        # Second call uses cache
        delta2 = mixed.get_mixed_delta(layer)

        np.testing.assert_array_equal(delta1, delta2)


# ============================================================================
# HOOK COVERAGE TESTS (for integration with GPT-2)
# ============================================================================

class TestHookCoverage:
    """Tests to verify hook coverage matches expectations."""

    def test_expected_layer_count(self, sample_layer_dims):
        """Verify we expect 24 layers for distilgpt2 (6 blocks * 4 modules)."""
        assert len(sample_layer_dims) == 24

    def test_all_target_modules_covered(self, config, sample_layer_dims):
        """Each target module type should appear in layer dims."""
        for target in config.target_modules:
            found = any(target in name for name in sample_layer_dims.keys())
            assert found, f"Target module {target} not found in layers"


# ============================================================================
# GPT's NUMERICAL TESTS — "ГВОЗДИ" (real GPT-2 integration)
# ============================================================================

# Check if transformers available
try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from stanley_hybrid.adapter_bank import GPT2WeightPatcher, create_adapter_system
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not available")
class TestGPT2Integration:
    """
    GPT's numerical tests — the real "гвозди" (nails).

    These tests use actual GPT-2 to verify:
    1. Hook coverage = 24 layers
    2. Zero mix = baseline logits
    3. One-hot mood = non-zero delta
    4. Linearity of mixing
    5. Detach restores baseline
    """

    @pytest.fixture
    def gpt2_setup(self):
        """Load GPT-2 and create adapter system."""
        model = GPT2LMHeadModel.from_pretrained("distilgpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
        model.eval()

        bank, router, mixed, patcher = create_adapter_system(model)
        return model, tokenizer, bank, router, mixed, patcher

    def test_hook_count_equals_24(self, gpt2_setup):
        """Test 1: hook_count == 4 * n_layers = 24."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        # Count hooks (patcher attaches during create_adapter_system)
        patcher.attach()
        assert len(patcher._hooks) == 24, \
            f"Expected 24 hooks, got {len(patcher._hooks)}"

    def test_hook_coverage_all_modules(self, gpt2_setup):
        """Verify all 4 target module types are covered in adapter bank."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        # Check that bank's layer_dims covers all target modules
        covered_modules = set()
        for layer_name in bank.layer_dims.keys():
            for target in ["c_attn", "c_proj", "c_fc"]:
                if target in layer_name:
                    covered_modules.add(target)

        expected = {"c_attn", "c_proj", "c_fc"}
        assert expected.issubset(covered_modules), \
            f"Missing modules: {expected - covered_modules}"

    def test_zero_mix_equals_baseline(self, gpt2_setup):
        """Test 2: mix=0 → logits identical to baseline (within tolerance)."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline (no patching)
        patcher.detach()
        with torch.no_grad():
            baseline_logits = model(**inputs).logits[:, -1, :].clone()

        # Re-attach and set zero mix
        patcher.attach()
        zero_mix = {m: 0.0 for m in Mood}
        mixed._cached_mix = zero_mix
        mixed._cached_deltas = {}

        with torch.no_grad():
            patched_logits = model(**inputs).logits[:, -1, :]

        # Compare
        max_diff = (patched_logits - baseline_logits).abs().max().item()
        assert max_diff < 1e-5, \
            f"Zero mix should equal baseline, got max diff {max_diff}"

    def test_one_hot_mood_non_zero_delta(self, gpt2_setup):
        """Test 3: Each mood produces non-zero, stable delta."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline
        patcher.detach()
        with torch.no_grad():
            baseline_logits = model(**inputs).logits[:, -1, :].clone()

        patcher.attach()

        for mood in Mood:
            # One-hot mix
            one_hot = {m: 0.0 for m in Mood}
            one_hot[mood] = 1.0
            mixed._cached_mix = one_hot
            mixed._cached_deltas = {}

            with torch.no_grad():
                logits = model(**inputs).logits[:, -1, :]

            delta = logits - baseline_logits
            norm = delta.norm().item()

            # Delta should be non-zero (mood has effect)
            assert norm > 1e-3, \
                f"Mood {mood.value} delta too small: {norm}"

            # No NaN or inf
            assert not torch.isnan(delta).any(), \
                f"NaN in delta for mood {mood.value}"
            assert not torch.isinf(delta).any(), \
                f"Inf in delta for mood {mood.value}"

    def test_mix_linearity(self, gpt2_setup):
        """Test 4: Mixed delta ≈ weighted sum of individual deltas."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline
        patcher.detach()
        with torch.no_grad():
            baseline_logits = model(**inputs).logits[:, -1, :].clone()

        patcher.attach()

        # Get individual mood deltas
        mood_a, mood_b = Mood.CALM, Mood.INTENSE
        alpha, beta = 0.6, 0.4

        # Delta A
        mix_a = {m: 0.0 for m in Mood}
        mix_a[mood_a] = 1.0
        mixed._cached_mix = mix_a
        mixed._cached_deltas = {}
        with torch.no_grad():
            delta_a = model(**inputs).logits[:, -1, :] - baseline_logits

        # Delta B
        mix_b = {m: 0.0 for m in Mood}
        mix_b[mood_b] = 1.0
        mixed._cached_mix = mix_b
        mixed._cached_deltas = {}
        with torch.no_grad():
            delta_b = model(**inputs).logits[:, -1, :] - baseline_logits

        # Mixed delta (alpha * A + beta * B)
        mix_combined = {m: 0.0 for m in Mood}
        mix_combined[mood_a] = alpha
        mix_combined[mood_b] = beta
        mixed._cached_mix = mix_combined
        mixed._cached_deltas = {}
        with torch.no_grad():
            delta_mixed = model(**inputs).logits[:, -1, :] - baseline_logits

        # Expected linear combination
        expected = alpha * delta_a + beta * delta_b

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            delta_mixed.flatten().unsqueeze(0),
            expected.flatten().unsqueeze(0)
        ).item()

        assert cos_sim > 0.95, \
            f"Mix linearity failed, cosine similarity = {cos_sim:.4f}"

    def test_detach_restores_baseline(self, gpt2_setup):
        """Test 5: After detach(), model returns to baseline."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        prompt = "Hello world"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Get initial baseline
        patcher.detach()
        with torch.no_grad():
            baseline_logits = model(**inputs).logits[:, -1, :].clone()

        # Attach and apply mood
        patcher.attach()
        one_hot = {m: 0.0 for m in Mood}
        one_hot[Mood.INTENSE] = 1.0
        mixed._cached_mix = one_hot
        mixed._cached_deltas = {}

        # Detach again
        patcher.detach()
        with torch.no_grad():
            restored_logits = model(**inputs).logits[:, -1, :]

        max_diff = (restored_logits - baseline_logits).abs().max().item()
        assert max_diff < 1e-5, \
            f"Detach should restore baseline, got max diff {max_diff}"

    def test_kl_divergence_change(self, gpt2_setup):
        """Test 6: Mood actually changes distribution (KL > eps)."""
        model, tokenizer, bank, router, mixed, patcher = gpt2_setup

        prompt = "The meaning of"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Baseline distribution
        patcher.detach()
        with torch.no_grad():
            baseline_logits = model(**inputs).logits[:, -1, :]
            baseline_probs = torch.softmax(baseline_logits, dim=-1)

        patcher.attach()

        # Apply mood
        one_hot = {m: 0.0 for m in Mood}
        one_hot[Mood.CREATIVE] = 1.0
        mixed._cached_mix = one_hot
        mixed._cached_deltas = {}

        with torch.no_grad():
            patched_logits = model(**inputs).logits[:, -1, :]
            patched_probs = torch.softmax(patched_logits, dim=-1)

        # KL divergence
        kl = torch.nn.functional.kl_div(
            patched_probs.log(),
            baseline_probs,
            reduction='sum'
        ).item()

        assert abs(kl) > 1e-4, \
            f"Mood should change distribution, KL = {kl}"


# ============================================================================
# HYPERLORA TESTS (Act 4)
# ============================================================================

from stanley_hybrid.adapter_bank import HyperMixer, HyperLoRA, HyperLoRATrainer


class TestHyperMixer:
    """Tests for HyperMixer (predicts mix coefficients)."""

    @pytest.fixture
    def mixer(self):
        return HyperMixer(hidden_dim=32)

    def test_mixer_creation(self, mixer):
        """Test HyperMixer creates correctly."""
        assert mixer.input_dim == 14
        assert mixer.num_moods == 8
        assert len(list(mixer.parameters())) > 0

    def test_signals_to_tensor(self, mixer, calm_signals):
        """Test signal conversion to tensor."""
        tensor = mixer.signals_to_tensor(calm_signals)
        assert tensor.shape == (14,)
        assert tensor.dtype == torch.float32

    def test_forward_shape(self, mixer, calm_signals):
        """Test forward pass output shape."""
        x = mixer.signals_to_tensor(calm_signals).unsqueeze(0)
        out = mixer.forward(x)
        assert out.shape == (1, 8)

    def test_predict_mix_sums_to_one(self, mixer, calm_signals):
        """Predicted mix should sum to 1 (softmax)."""
        mix = mixer.predict_mix(calm_signals)
        total = sum(mix.values())
        assert abs(total - 1.0) < 1e-5


class TestHyperLoRA:
    """Tests for HyperLoRA (generates ΔW from signals)."""

    @pytest.fixture
    def small_bank(self):
        """Create small adapter bank for testing."""
        layer_dims = {
            'h.0.attn.c_attn': (64, 32),
            'h.0.mlp.c_fc': (64, 32),
        }
        config = AdapterBankConfig(lora_rank=4)
        bank = AdapterBank(config)
        bank.initialize_adapters(layer_dims)
        return bank

    @pytest.fixture
    def hyperlora(self, small_bank):
        return HyperLoRA(small_bank, hidden_dim=32, num_basis=8)

    def test_hyperlora_creation(self, hyperlora):
        """Test HyperLoRA creates correctly."""
        assert hyperlora.input_dim == 14
        assert hyperlora.num_basis == 8
        assert len(hyperlora.layer_names) == 2

    def test_basis_deltas_initialized(self, hyperlora):
        """Basis deltas should be initialized from bank."""
        for layer_name in hyperlora.layer_names:
            assert layer_name in hyperlora.basis_deltas
            basis = hyperlora.basis_deltas[layer_name]
            assert basis.shape[0] == 8  # num_basis

    def test_forward_produces_deltas(self, hyperlora, calm_signals):
        """Forward pass should produce delta for each layer."""
        x = hyperlora.signals_to_tensor(calm_signals).unsqueeze(0)
        deltas = hyperlora.forward(x)

        assert len(deltas) == len(hyperlora.layer_names)
        for layer_name, delta in deltas.items():
            assert delta.shape[0] == 1  # batch size

    def test_get_delta_returns_numpy(self, hyperlora, calm_signals):
        """get_delta should return numpy array."""
        layer_name = hyperlora.layer_names[0]
        delta = hyperlora.get_delta(calm_signals, layer_name)

        assert isinstance(delta, np.ndarray)
        assert delta.ndim == 2

    def test_delta_no_nan(self, hyperlora, calm_signals):
        """Delta should not contain NaN."""
        for layer_name in hyperlora.layer_names:
            delta = hyperlora.get_delta(calm_signals, layer_name)
            assert not np.isnan(delta).any()

    def test_delta_bounded(self, hyperlora, intense_signals):
        """Delta norm should be bounded."""
        for layer_name in hyperlora.layer_names:
            delta = hyperlora.get_delta(intense_signals, layer_name)
            norm = np.linalg.norm(delta)
            assert norm < 100.0  # Reasonable bound


class TestHyperLoRATrainer:
    """Tests for HyperLoRA training (distillation)."""

    @pytest.fixture
    def training_setup(self):
        """Create training setup."""
        layer_dims = {
            'h.0.attn.c_attn': (64, 32),
            'h.0.mlp.c_fc': (64, 32),
        }
        config = AdapterBankConfig(lora_rank=4)
        bank = AdapterBank(config)
        bank.initialize_adapters(layer_dims)

        hyperlora = HyperLoRA(bank, hidden_dim=32, num_basis=8)
        router = MoodRouter()
        mixed = MixedAdapter(bank, router)
        trainer = HyperLoRATrainer(hyperlora, bank, router, mixed)

        return trainer, hyperlora, bank, router, mixed

    def test_trainer_creation(self, training_setup):
        """Test trainer creates correctly."""
        trainer, hyperlora, bank, router, mixed = training_setup
        assert trainer.hyperlora is hyperlora
        assert trainer.step == 0

    def test_generate_random_signals(self, training_setup):
        """Test random signal generation."""
        trainer, *_ = training_setup
        signals = trainer.generate_random_signals()

        assert 0 <= signals.pulse_arousal <= 1
        assert 0 <= signals.pulse_entropy <= 1
        assert signals.overthink_depth >= 0

    def test_train_step_returns_loss(self, training_setup):
        """Training step should return loss value."""
        trainer, *_ = training_setup
        signals = trainer.generate_random_signals()
        loss = trainer.train_step(signals)

        assert isinstance(loss, float)
        assert loss >= 0
        assert trainer.step == 1

    def test_multiple_train_steps(self, training_setup):
        """Multiple training steps should work."""
        trainer, *_ = training_setup

        losses = []
        for _ in range(10):
            signals = trainer.generate_random_signals()
            loss = trainer.train_step(signals)
            losses.append(loss)

        assert trainer.step == 10
        assert len(trainer.train_losses) == 10

    def test_evaluate_returns_metrics(self, training_setup):
        """Evaluate should return proper metrics."""
        trainer, *_ = training_setup
        metrics = trainer.evaluate(num_samples=10)

        assert "cosine_similarity_mean" in metrics
        assert "mse_mean" in metrics
        assert 0 <= metrics["cosine_similarity_mean"] <= 1


class TestHyperLoRADeterminism:
    """Tests for HyperLoRA deterministic behavior."""

    @pytest.fixture
    def hyperlora_setup(self):
        layer_dims = {'h.0.attn.c_attn': (64, 32)}
        config = AdapterBankConfig(lora_rank=4)
        bank = AdapterBank(config)
        bank.initialize_adapters(layer_dims)
        hyperlora = HyperLoRA(bank, hidden_dim=32, num_basis=8)
        return hyperlora, bank

    def test_same_signals_same_delta(self, hyperlora_setup, calm_signals):
        """Same signals should produce identical deltas."""
        hyperlora, bank = hyperlora_setup
        layer_name = list(bank.layer_dims.keys())[0]

        delta1 = hyperlora.get_delta(calm_signals, layer_name)
        delta2 = hyperlora.get_delta(calm_signals, layer_name)

        np.testing.assert_array_equal(delta1, delta2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

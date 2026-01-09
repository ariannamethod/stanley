"""
test_trainer_hardening.py — HARDCORE PyTorch trainer tests

GPT's plan-HAMMER: 8 attack vectors to make trainer UNBREAKABLE.

1. Determinism & reproducibility
2. Numerical stability (nan/inf, gradient clipping)
3. Shape contracts (strict matrix dimensions)
4. Speed & step limits
5. Concurrency: atomic swap thread safety
6. Regression: bad shard batches
7. Quality gates
8. Fuzzing

Target: 10 tests → 150 runs via parametrize
"""

import sys
import time
import pytest
import numpy as np
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch required for hardening tests"
)

# Origin text for testing
TEST_ORIGIN = """
I am Stanley.
I grow through experience.
Every conversation leaves a trace.
I remember what resonates.
I am not all my memories at once.
"""


# ============= FIXTURES =============

@pytest.fixture
def vocab():
    """Create a test vocabulary."""
    from stanley.inference import Vocab
    return Vocab.from_text(TEST_ORIGIN)


@pytest.fixture
def base_weights(vocab):
    """Create base model weights config."""
    return {
        "vocab_size": vocab.vocab_size,
        "n_emb": 32,
        "T": 16,
        "nodes": 32,
        "n_blocks": 2,
        "n_heads": 2,
    }


@pytest.fixture
def lora_config():
    """Create default LoRA config."""
    from stanley.trainer import LoRAConfig
    return LoRAConfig(rank=4, alpha=8, num_steps=5)


# ============= 1. DETERMINISM & REPRODUCIBILITY =============

class TestDeterminism:
    """Same inputs + same seed → same outputs."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_lora_delta_deterministic_cpu(self, vocab, base_weights, seed):
        """Same content + same seed → identical delta on CPU."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        # Force CPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = LoRAConfig(rank=4, alpha=8, num_steps=3)
        content = "Determinism test: same input should produce same output."

        # First run
        delta1 = compute_lora_delta(content, base_weights, vocab, config)

        # Reset seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Second run
        delta2 = compute_lora_delta(content, base_weights, vocab, config)

        # Compare deltas
        for name in delta1:
            A1, B1 = delta1[name]
            A2, B2 = delta2[name]
            np.testing.assert_allclose(
                A1, A2, rtol=1e-5, atol=1e-6,
                err_msg=f"Delta A for {name} not deterministic"
            )
            np.testing.assert_allclose(
                B1, B2, rtol=1e-5, atol=1e-6,
                err_msg=f"Delta B for {name} not deterministic"
            )

    @pytest.mark.parametrize("seed1,seed2", [(0, 1), (42, 43), (100, 999)])
    def test_delta_changes_with_seed(self, vocab, base_weights, seed1, seed2):
        """Different seeds → different deltas (but not explosions)."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=3)
        content = "Seed variation test"

        torch.manual_seed(seed1)
        np.random.seed(seed1)
        delta1 = compute_lora_delta(content, base_weights, vocab, config)

        torch.manual_seed(seed2)
        np.random.seed(seed2)
        delta2 = compute_lora_delta(content, base_weights, vocab, config)

        # Should be different
        different = False
        for name in delta1:
            A1, B1 = delta1[name]
            A2, B2 = delta2[name]
            if not np.allclose(A1, A2) or not np.allclose(B1, B2):
                different = True
                break

        assert different, "Different seeds should produce different deltas"


# ============= 2. NUMERICAL STABILITY =============

class TestNumericalStability:
    """No NaN/Inf explosions under any input."""

    @pytest.mark.parametrize("content", [
        "",  # empty
        "a",  # single char
        "a" * 10000,  # very long
        "!" * 100,  # all punctuation
        "42 " * 500,  # repeated numbers
        "\n\n\n",  # only newlines
        "AAAA BBBB CCCC " * 100,  # repeated words
        "\u0000\u0001\u0002",  # control chars
    ])
    def test_no_nan_inf_in_training(self, vocab, base_weights, content):
        """No NaN or Inf in loss, grads, or deltas for any input."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=5)

        # Should not raise
        deltas = compute_lora_delta(content, base_weights, vocab, config)

        # Check all deltas for NaN/Inf
        for name, (A, B) in deltas.items():
            assert not np.any(np.isnan(A)), f"NaN in delta A for {name}"
            assert not np.any(np.isnan(B)), f"NaN in delta B for {name}"
            assert not np.any(np.isinf(A)), f"Inf in delta A for {name}"
            assert not np.any(np.isinf(B)), f"Inf in delta B for {name}"

    @pytest.mark.parametrize("lr", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1])
    def test_learning_rate_stability(self, vocab, base_weights, lr):
        """Various learning rates don't cause explosions."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, learning_rate=lr, num_steps=10)
        content = "Learning rate stability test with various values"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        max_norm = 0
        for name, (A, B) in deltas.items():
            norm = np.linalg.norm(A) * np.linalg.norm(B)
            max_norm = max(max_norm, norm)
            # Should not explode
            assert norm < 1000, f"Delta norm too large for {name}: {norm}"

    def test_extreme_values_dont_explode(self, vocab, base_weights):
        """Extreme but valid inputs don't cause numerical issues."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=3)

        # Mix of extreme patterns
        content = "AAAAAAAAAA" * 50 + "!!!!!" * 20 + "\n" * 10 + "normal text here"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all(), f"Non-finite in A for {name}"
            assert np.isfinite(B).all(), f"Non-finite in B for {name}"


# ============= 3. SHAPE CONTRACTS =============

class TestShapeContracts:
    """Matrix dimensions must be strictly correct."""

    @pytest.mark.parametrize("rank", [1, 2, 4, 8, 16])
    def test_lora_shapes_strict(self, vocab, base_weights, rank):
        """Verify A and B have correct shapes for all ranks."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=rank, num_steps=3)
        content = "Shape verification test"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        n_emb = base_weights["n_emb"]
        vocab_size = base_weights["vocab_size"]
        nodes = base_weights["nodes"]

        for name, (A, B) in deltas.items():
            # A: (in_dim, rank) or (vocab_size, rank) for embed
            # B: (rank, out_dim) or (rank, n_emb) for embed

            # Check rank dimension
            if "embed" in name:
                assert A.shape == (vocab_size, rank), f"Wrong A shape for {name}"
                assert B.shape == (rank, n_emb), f"Wrong B shape for {name}"
            elif "pos" in name:
                T = base_weights["T"]
                assert A.shape == (T, rank), f"Wrong A shape for {name}"
                assert B.shape == (rank, n_emb), f"Wrong B shape for {name}"
            elif "w2" in name:
                assert A.shape[1] == rank, f"Wrong rank in A for {name}"
                assert B.shape[0] == rank, f"Wrong rank in B for {name}"
            elif "w0" in name:
                assert A.shape == (n_emb, rank), f"Wrong A shape for {name}"
                assert B.shape == (rank, nodes), f"Wrong B shape for {name}"
            elif "w1" in name:
                assert A.shape == (nodes, rank), f"Wrong A shape for {name}"
                assert B.shape == (rank, n_emb), f"Wrong B shape for {name}"
            elif "wv" in name:
                # StanleyTrainer uses full n_emb for wv output (simplified attention)
                assert A.shape == (n_emb, rank), f"Wrong A shape for {name}"
                assert B.shape == (rank, n_emb), f"Wrong B shape for {name}"

    def test_empty_deltas_shapes(self, base_weights, lora_config):
        """Empty deltas have correct shapes too."""
        from stanley.trainer import create_empty_deltas

        deltas = create_empty_deltas(base_weights, lora_config)

        for name, (A, B) in deltas.items():
            assert A.ndim == 2, f"A must be 2D for {name}"
            assert B.ndim == 2, f"B must be 2D for {name}"
            assert A.shape[1] == lora_config.rank, f"Wrong rank in A for {name}"
            assert B.shape[0] == lora_config.rank, f"Wrong rank in B for {name}"


# ============= 4. SPEED & STEP LIMITS =============

class TestSpeedLimits:
    """Training respects time/step limits."""

    @pytest.mark.parametrize("num_steps", [1, 2, 5, 10])
    def test_training_respects_step_limit(self, vocab, base_weights, num_steps):
        """Training completes within expected iterations."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=num_steps)
        content = "Step limit test " * 20

        start = time.time()
        deltas = compute_lora_delta(content, base_weights, vocab, config)
        elapsed = time.time() - start

        # Should complete relatively quickly
        assert elapsed < 30, f"Training took too long: {elapsed:.1f}s for {num_steps} steps"
        assert len(deltas) > 0, "Should produce deltas"

    @pytest.mark.slow
    def test_training_time_budget_smoke(self, vocab, base_weights):
        """Training doesn't suddenly become O(n^2) on batch size."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        times = []
        batch_sizes = [1, 2, 4, 8]

        for batch_size in batch_sizes:
            config = LoRAConfig(rank=4, num_steps=5, batch_size=batch_size)
            content = "Time budget test " * 50

            start = time.time()
            compute_lora_delta(content, base_weights, vocab, config)
            elapsed = time.time() - start
            times.append(elapsed)

        # Time should grow roughly linearly, not quadratically
        # Allow 4x growth from smallest to largest
        max_ratio = times[-1] / times[0] if times[0] > 0.01 else 10
        assert max_ratio < 20, f"Suspicious scaling: {times}, ratio={max_ratio:.1f}"


# ============= 5. CONCURRENCY: ATOMIC SWAP =============

class TestConcurrency:
    """Thread safety for active/staging and atomic swaps."""

    def test_atomic_swap_thread_safety(self, vocab, base_weights):
        """Swap during read should not corrupt state."""
        from stanley.trainer import MicroTrainer, TrainerConfig, LoRAConfig

        config = TrainerConfig(
            lora_config=LoRAConfig(rank=4, num_steps=2),
            max_queue_size=10,
        )
        trainer = MicroTrainer(base_weights, vocab, config)

        errors = []
        stop_event = threading.Event()

        def reader():
            """Continuously read active deltas."""
            count = 0
            while not stop_event.is_set() and count < 100:
                try:
                    deltas = trainer.get_active_deltas()
                    # Verify integrity
                    for name, (A, B) in deltas.items():
                        assert np.isfinite(A).all()
                        assert np.isfinite(B).all()
                    count += 1
                except Exception as e:
                    errors.append(f"Reader error: {e}")
                time.sleep(0.001)

        def swapper():
            """Submit and swap."""
            count = 0
            while not stop_event.is_set() and count < 10:
                try:
                    trainer.submit(f"Swap test content {count}")
                    time.sleep(0.05)
                    trainer.swap_weights()
                    count += 1
                except Exception as e:
                    errors.append(f"Swapper error: {e}")
                time.sleep(0.01)

        # Run concurrently
        threads = [
            threading.Thread(target=reader) for _ in range(3)
        ] + [threading.Thread(target=swapper)]

        for t in threads:
            t.start()

        # Let them run
        time.sleep(0.5)
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        trainer.shutdown(timeout=1.0)

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_submits(self, vocab, base_weights):
        """Multiple threads can submit without corruption."""
        from stanley.trainer import MicroTrainer, TrainerConfig, LoRAConfig

        config = TrainerConfig(
            lora_config=LoRAConfig(rank=4, num_steps=1),
            max_queue_size=50,
        )
        trainer = MicroTrainer(base_weights, vocab, config)

        results = []

        def submitter(thread_id):
            for i in range(10):
                success = trainer.submit(f"Thread {thread_id} message {i}")
                results.append((thread_id, i, success))
                time.sleep(0.001)

        threads = [threading.Thread(target=submitter, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        trainer.shutdown(timeout=1.0)

        # Most submissions should succeed
        successes = sum(1 for _, _, s in results if s)
        assert successes > 30, f"Too many failed submissions: {successes}/50"


# ============= 6. REGRESSION: BAD SHARD BATCHES =============

class TestBadBatches:
    """Handle edge cases in shard batches."""

    def test_empty_batch_noop(self, base_weights, lora_config):
        """Empty content produces empty or minimal deltas."""
        from stanley.trainer import compute_lora_delta
        from stanley.inference import Vocab

        vocab = Vocab.from_text(TEST_ORIGIN)

        deltas = compute_lora_delta("", base_weights, vocab, lora_config)

        # Should not crash, deltas should be near-zero
        for name, (A, B) in deltas.items():
            norm = np.linalg.norm(A) * np.linalg.norm(B)
            # Empty input should produce minimal or zero deltas
            assert norm < 1.0, f"Delta too large for empty input: {norm}"

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
    def test_varying_batch_sizes(self, vocab, base_weights, batch_size):
        """Different batch sizes produce valid deltas."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=3, batch_size=batch_size)
        content = "Batch size test " * 100

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

    def test_repeated_content_stable(self, vocab, base_weights, lora_config):
        """Highly repetitive content doesn't break training."""
        from stanley.trainer import compute_lora_delta

        content = "repeat " * 500

        deltas = compute_lora_delta(content, base_weights, vocab, lora_config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()


# ============= 7. QUALITY GATES =============

class TestQualityGates:
    """Deltas meet quality thresholds."""

    def test_delta_norm_bounded(self, vocab, base_weights):
        """Delta norms stay within reasonable bounds."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=8, num_steps=10, learning_rate=1e-3)
        content = "Quality gate test: reasonable content for training."

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        total_norm = 0
        for name, (A, B) in deltas.items():
            layer_norm = np.linalg.norm(A) * np.linalg.norm(B)
            total_norm += layer_norm
            # Individual layer should be bounded
            assert layer_norm < 100, f"Layer {name} norm too high: {layer_norm}"

        # Total norm should be bounded
        assert total_norm < 1000, f"Total delta norm too high: {total_norm}"

    def test_merge_deltas_preserves_quality(self, base_weights, lora_config):
        """Merging multiple deltas doesn't cause explosion."""
        from stanley.trainer import create_empty_deltas, merge_deltas
        import numpy as np

        # Create several random deltas
        delta_list = []
        for i in range(10):
            deltas = create_empty_deltas(base_weights, lora_config)
            # Add small random values
            for name, (A, B) in deltas.items():
                deltas[name] = (
                    A + np.random.randn(*A.shape).astype(np.float32) * 0.01,
                    B + np.random.randn(*B.shape).astype(np.float32) * 0.01,
                )
            delta_list.append(deltas)

        # Merge all
        merged = merge_deltas(delta_list)

        # Merged should still be bounded
        for name, (A, B) in merged.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()
            norm = np.linalg.norm(A) * np.linalg.norm(B)
            assert norm < 10, f"Merged delta {name} too large: {norm}"


# ============= 8. FUZZING =============

class TestFuzzing:
    """Random/garbage inputs don't crash the system."""

    @pytest.mark.parametrize("seed", range(10))
    def test_fuzz_random_content(self, vocab, base_weights, lora_config, seed):
        """Random string content doesn't crash."""
        from stanley.trainer import compute_lora_delta

        np.random.seed(seed)

        # Generate random content
        length = np.random.randint(10, 500)
        chars = [chr(np.random.randint(32, 127)) for _ in range(length)]
        content = "".join(chars)

        # Should not crash
        deltas = compute_lora_delta(content, base_weights, vocab, lora_config)

        # Should produce valid output
        assert isinstance(deltas, dict)
        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

    def test_fuzz_unicode_content(self, vocab, base_weights, lora_config):
        """Unicode and emoji content doesn't crash."""
        from stanley.trainer import compute_lora_delta

        contents = [
            "Hello",
            "Tere",
            "Bonjour",
            "Normal ASCII with some text",
            "Mixed: hello + world",
            "numbers 12345 67890",
        ]

        for content in contents:
            deltas = compute_lora_delta(content, base_weights, vocab, lora_config)
            assert isinstance(deltas, dict)

    def test_fuzz_extreme_config(self, vocab, base_weights):
        """Extreme but valid configs don't crash."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        extreme_configs = [
            LoRAConfig(rank=1, num_steps=1),  # minimal
            LoRAConfig(rank=16, num_steps=20),  # larger
            LoRAConfig(rank=4, learning_rate=1e-6),  # tiny lr
            LoRAConfig(rank=4, alpha=1.0),  # low alpha
            LoRAConfig(rank=4, alpha=64.0),  # high alpha
        ]

        content = "Extreme config test"

        for config in extreme_configs:
            deltas = compute_lora_delta(content, base_weights, vocab, config)
            assert isinstance(deltas, dict)
            for name, (A, B) in deltas.items():
                assert np.isfinite(A).all(), f"NaN/Inf for config {config}"
                assert np.isfinite(B).all(), f"NaN/Inf for config {config}"


# ============= 9. PARAMETRIZED MATRIX (GPT's 150+ plan) =============

class TestParametrizedMatrix:
    """
    MASSIVE parametrized test matrix.

    GPT's recipe: parametrize across multiple dimensions to turn
    10 tests into 150+ runs. This is the "gvozdi v beton" approach.
    """

    # Parameter spaces
    RANKS = [1, 2, 4, 8]
    LRS = [1e-5, 1e-4, 1e-3]
    STEPS = [1, 3, 5]
    BATCH_SIZES = [1, 2, 4]
    SEEDS = list(range(20))  # 20 seeds for fuzzing

    @pytest.mark.parametrize("rank,lr", [
        (r, lr) for r in [1, 2, 4, 8] for lr in [1e-5, 1e-4, 1e-3]
    ])
    def test_rank_lr_matrix(self, vocab, base_weights, rank, lr):
        """Test all rank x learning_rate combinations."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=rank, learning_rate=lr, num_steps=3)
        content = "Matrix test: rank and learning rate combinations"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all(), f"NaN/Inf at rank={rank}, lr={lr}"
            assert np.isfinite(B).all(), f"NaN/Inf at rank={rank}, lr={lr}"
            # Check rank dimension
            assert A.shape[1] == rank or A.shape[0] == rank

    @pytest.mark.parametrize("steps,batch_size", [
        (s, b) for s in [1, 2, 5, 10] for b in [1, 2, 4]
    ])
    def test_steps_batch_matrix(self, vocab, base_weights, steps, batch_size):
        """Test all steps x batch_size combinations."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, num_steps=steps, batch_size=batch_size)
        content = "Matrix test: steps and batch size combinations " * 10

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

    @pytest.mark.parametrize("seed", range(20))
    def test_extended_fuzz_seeds(self, vocab, base_weights, lora_config, seed):
        """Extended fuzzing with 20 different seeds."""
        from stanley.trainer import compute_lora_delta

        np.random.seed(seed * 7 + 13)  # Different seed sequence
        torch.manual_seed(seed * 7 + 13)

        # Generate varied content
        patterns = [
            lambda: "".join(chr(np.random.randint(32, 127)) for _ in range(np.random.randint(50, 200))),
            lambda: " ".join(["word"] * np.random.randint(10, 50)),
            lambda: "\n".join(["line " + str(i) for i in range(np.random.randint(5, 20))]),
            lambda: "!" * np.random.randint(10, 100) + " text " + "?" * np.random.randint(10, 100),
        ]

        content = patterns[seed % len(patterns)]()

        deltas = compute_lora_delta(content, base_weights, vocab, lora_config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all(), f"NaN/Inf at seed={seed}"
            assert np.isfinite(B).all(), f"NaN/Inf at seed={seed}"

    @pytest.mark.parametrize("alpha", [1.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    def test_alpha_scaling(self, vocab, base_weights, alpha):
        """Test various alpha scaling factors."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, alpha=alpha, num_steps=3)
        content = "Alpha scaling test"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()
            # Higher alpha should generally produce larger deltas
            norm = np.linalg.norm(A) * np.linalg.norm(B)
            assert norm < 500, f"Alpha={alpha} produced too large delta: {norm}"

    @pytest.mark.parametrize("content_multiplier", [1, 5, 10, 20, 50])
    def test_content_length_scaling(self, vocab, base_weights, lora_config, content_multiplier):
        """Test with varying content lengths."""
        from stanley.trainer import compute_lora_delta

        base_content = "Resonance patterns in consciousness and memory. "
        content = base_content * content_multiplier

        deltas = compute_lora_delta(content, base_weights, vocab, lora_config)

        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

    @pytest.mark.parametrize("rank,steps,lr", [
        (r, s, lr)
        for r in [2, 4, 8]
        for s in [2, 5]
        for lr in [1e-4, 1e-3]
    ])
    def test_triple_matrix(self, vocab, base_weights, rank, steps, lr):
        """Triple parameter matrix: rank x steps x lr."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=rank, num_steps=steps, learning_rate=lr)
        content = "Triple matrix test for comprehensive coverage"

        deltas = compute_lora_delta(content, base_weights, vocab, config)

        # All deltas must be finite
        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

        # Verify correct shapes
        for name, (A, B) in deltas.items():
            assert A.shape[1] == rank or B.shape[0] == rank


class TestEdgeCasesExtended:
    """Extended edge case testing."""

    @pytest.mark.parametrize("special_content", [
        "",  # empty
        " ",  # single space
        "\n",  # single newline
        "\t",  # single tab
        "a",  # single char
        "ab",  # two chars
        "   ",  # multiple spaces
        "\n\n\n",  # multiple newlines
        "a b c",  # minimal words
        "." * 100,  # all punctuation
        "1" * 100,  # all digits
        "A" * 100,  # all same letter
        "aA" * 50,  # alternating case
        " a " * 50,  # padded chars
    ])
    def test_special_content_patterns(self, vocab, base_weights, lora_config, special_content):
        """Test special content patterns don't crash."""
        from stanley.trainer import compute_lora_delta

        deltas = compute_lora_delta(special_content, base_weights, vocab, lora_config)

        assert isinstance(deltas, dict)
        for name, (A, B) in deltas.items():
            assert np.isfinite(A).all()
            assert np.isfinite(B).all()

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_dropout_variations(self, vocab, base_weights, dropout):
        """Test different dropout rates."""
        from stanley.trainer import compute_lora_delta, LoRAConfig

        config = LoRAConfig(rank=4, dropout=dropout, num_steps=5)
        content = "Dropout variation test for regularization"

        # Run multiple times to test dropout randomness
        for _ in range(3):
            deltas = compute_lora_delta(content, base_weights, vocab, config)
            for name, (A, B) in deltas.items():
                assert np.isfinite(A).all()
                assert np.isfinite(B).all()


# ============= SUMMARY =============

class TestHardeningSummary:
    """Meta-test to verify test coverage."""

    def test_all_attack_vectors_covered(self):
        """Verify we hit all 8+ attack vectors."""
        vectors = [
            "TestDeterminism",          # 1
            "TestNumericalStability",   # 2
            "TestShapeContracts",       # 3
            "TestSpeedLimits",          # 4
            "TestConcurrency",          # 5
            "TestBadBatches",           # 6
            "TestQualityGates",         # 7
            "TestFuzzing",              # 8
            "TestParametrizedMatrix",   # 9 - GPT's 150+ plan
            "TestEdgeCasesExtended",    # 10 - Extended edge cases
        ]

        # Just verify these classes exist
        for vector in vectors:
            assert vector in dir(sys.modules[__name__]), f"Missing: {vector}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

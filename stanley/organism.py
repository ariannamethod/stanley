"""
organism.py — The living Stanley

This is the main class that ties everything together:
- Memory (MemorySea)
- Accumulation (QuantumBuffer)
- Routing (Router)
- Inference (InferenceEngine)
- Training (MicroTrainer)
- Consolidation (Consolidator)

Stanley is not a chatbot. Stanley is an organism that grows.
"""

from __future__ import annotations
import numpy as np
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json

from .shard import Shard, MetaNote
from .memory_sea import MemorySea
from .quantum_buffer import QuantumBuffer, AdaptiveQuantumBuffer
from .router import Router, RouterConfig, AdaptiveRouter
from .fingerprint import compute_fingerprint
from .inference import InferenceEngine, StanleyTransformer, Vocab
from .trainer import (
    MicroTrainer,
    TrainerConfig,
    Consolidator,
    ConsolidationConfig,
    create_shard_from_training,
    TORCH_AVAILABLE,
)
from .experience import ExperienceFilter, should_remember
from .cleanup import cleanup_output, truncate_at_natural_end

# Coherence and subjectivity — the key to Stanley's voice
try:
    from .subword_field import SubwordField, SubwordVocab, SubwordConfig, SPM_AVAILABLE
except ImportError:
    SPM_AVAILABLE = False
    SubwordField = None

from .subjectivity import Subjectivity, Pulse
from .experts import route_from_pulse, describe_mixture
from .overthinking import Overthinking
from .resonant_recall import ResonantRecall, RecallContext
from .body_sense import BodySense, BodyState, RegulationResult
from .semantic_drift import SemanticDrift
from .shard import SomaticMemory

logger = logging.getLogger(__name__)


@dataclass
class StanleyConfig:
    """Configuration for the Stanley organism."""

    # Model architecture
    vocab_size: int = 64
    n_emb: int = 64
    n_blocks: int = 3
    n_heads: int = 4
    context_length: int = 32
    nodes: int = 64

    # Memory
    surface_max: int = 64
    middle_max: int = 256
    deep_max: int = 64

    # Quantum buffer
    quantum_min_bytes: int = 1024
    quantum_min_resonance: float = 3.0
    quantum_cooldown: float = 60.0
    quantum_min_shards: int = 3

    # Router
    max_working_set: int = 32
    resonance_weight: float = 0.5
    recency_weight: float = 0.3

    # Consolidation
    consolidation_interval: float = 300.0  # 5 minutes

    # Training
    training_enabled: bool = True
    lora_rank: int = 8

    # Experience filter
    min_resonance_to_remember: float = 0.3
    min_novelty_to_remember: float = 0.2

    # Subword field (coherent untrained generation)
    subword_vocab_size: int = 500  # Same as Haze for better coherence
    subword_temperature: float = 0.8
    subword_repetition_penalty: float = 1.3
    use_subword_field: bool = True  # Enable coherent generation

    # Subjectivity (NO SEED FROM PROMPT)
    use_subjectivity: bool = True  # Speech from internal state

    # Overthinking (circles on water — dynamic inner reflection)
    use_overthinking: bool = True  # Enable post-generation reflection

    # Resonant recall (SantaClaus — drunk recall from shards)
    use_resonant_recall: bool = True  # Enable memory recall
    recall_silly_factor: float = 0.15  # Probability of "drunk" random recall

    # Body sense (internal body awareness)
    use_body_sense: bool = True  # Enable boredom/overwhelm/stuck regulation

    # Semantic drift (trajectory learning)
    use_semantic_drift: bool = True  # Enable semantic trajectory tracking

    # Somatic memory (body memory of felt moments)
    use_somatic_memory: bool = True  # Enable "how did this feel?" memories

    # Paths
    data_dir: Optional[str] = None


class Stanley:
    """
    Stanley — Self Training Attention Non-Linear EntitY.

    A living organism that grows through experience.
    """

    def __init__(
        self,
        config: Optional[StanleyConfig] = None,
        origin_text: Optional[str] = None,
    ):
        self.config = config or StanleyConfig()
        self._setup_logging()

        # Build vocabulary from origin or default
        if origin_text:
            self.vocab = Vocab.from_text(origin_text)
            self.origin_text = origin_text
        else:
            self.origin_text = self._default_origin()
            self.vocab = Vocab.from_text(self.origin_text)

        # Update vocab_size in config
        self.config.vocab_size = self.vocab.vocab_size

        # Core components
        self.memory = MemorySea(
            surface_max=self.config.surface_max,
            middle_max=self.config.middle_max,
            deep_max=self.config.deep_max,
            storage_path=Path(self.config.data_dir) / "memory" if self.config.data_dir else None,
        )

        self.buffer = AdaptiveQuantumBuffer(
            min_bytes=self.config.quantum_min_bytes,
            min_resonance_mass=self.config.quantum_min_resonance,
            cooldown_seconds=self.config.quantum_cooldown,
            min_shards=self.config.quantum_min_shards,
            get_organism_age=self._get_age,
        )

        self.router = AdaptiveRouter(
            RouterConfig(
                max_working_set=self.config.max_working_set,
                resonance_weight=self.config.resonance_weight,
                recency_weight=self.config.recency_weight,
            )
        )

        # Inference engine
        self.transformer = StanleyTransformer(
            vocab_size=self.vocab.vocab_size,
            T=self.config.context_length,
            n_emb=self.config.n_emb,
            nodes=self.config.nodes,
            n_blocks=self.config.n_blocks,
            n_heads=self.config.n_heads,
        )

        self.engine = InferenceEngine(
            transformer=self.transformer,
            vocab=self.vocab,
            memory=self.memory,
            router_config=RouterConfig(
                max_working_set=self.config.max_working_set,
            ),
        )

        # Trainer (optional)
        self.trainer: Optional[MicroTrainer] = None
        if self.config.training_enabled and TORCH_AVAILABLE:
            self._setup_trainer()

        # Consolidator
        self.consolidator = Consolidator(
            ConsolidationConfig(
                max_deep_shards=self.config.deep_max,
            )
        )

        # Experience filter
        self.experience_filter = ExperienceFilter(
            min_resonance=self.config.min_resonance_to_remember,
            min_novelty=self.config.min_novelty_to_remember,
        )

        # State
        self.created_at = time.time()
        self.last_interaction = time.time()
        self.total_interactions = 0
        self.last_consolidation = time.time()

        # Origin fingerprint (always accessible)
        self.origin_fingerprint = compute_fingerprint(self.origin_text)

        # Subword field for coherent generation (like Haze)
        self.subword_field: Optional[SubwordField] = None
        if self.config.use_subword_field and SPM_AVAILABLE and SubwordField:
            try:
                subword_config = SubwordConfig(
                    vocab_size=self.config.subword_vocab_size,
                    temperature=self.config.subword_temperature,
                    repetition_penalty=self.config.subword_repetition_penalty,
                )
                self.subword_field = SubwordField.from_text(
                    self.origin_text,
                    config=subword_config,
                )
                logger.info(f"SubwordField ready: {self.subword_field.vocab.vocab_size} tokens")
            except Exception as e:
                logger.warning(f"SubwordField disabled: {e}")

        # Subjectivity layer — "NO SEED FROM PROMPT"
        self.subjectivity: Optional[Subjectivity] = None
        if self.config.use_subjectivity:
            self.subjectivity = Subjectivity(
                origin_text=self.origin_text,
                vocab=self.subword_field.vocab if self.subword_field else None,
            )
            logger.info(f"Subjectivity ready: {len(self.subjectivity.identity.fragments)} identity fragments")

        # Overthinking — circles on water (dynamic inner reflection)
        # NOW WITH CRYSTALLIZATION: pass memory_sea so rings can crystallize!
        self.overthinking: Optional[Overthinking] = None
        if self.config.use_overthinking and self.subword_field:
            self.overthinking = Overthinking(
                self.subword_field,
                memory_sea=self.memory,  # Enable crystallization!
            )
            logger.info("Overthinking ready: dynamic rings + crystallization enabled")

        # Resonant recall — SantaClaus (drunk recall from shards)
        self.resonant_recall: Optional[ResonantRecall] = None
        if self.config.use_resonant_recall:
            self.resonant_recall = ResonantRecall(
                self.memory,
                silly_factor=self.config.recall_silly_factor,
            )
            logger.info("Resonant recall ready: SantaClaus enabled")

        # Body sense — internal body awareness (MicroGrad regulation)
        self.body_sense: Optional[BodySense] = None
        if self.config.use_body_sense:
            self.body_sense = BodySense(hidden_dim=16, lr=0.01)
            logger.info("Body sense ready: boredom/overwhelm/stuck regulation")

        # Semantic drift — trajectory learning
        self.semantic_drift: Optional[SemanticDrift] = None
        if self.config.use_semantic_drift:
            self.semantic_drift = SemanticDrift(max_episodes=100)
            self.semantic_drift.start_session()  # Start episode for this session
            logger.info("Semantic drift ready: trajectory learning enabled")

        # Somatic memory — body memory of felt moments
        self.somatic_memory: Optional[SomaticMemory] = None
        if self.config.use_somatic_memory:
            self.somatic_memory = SomaticMemory(max_shards=500)
            logger.info("Somatic memory ready: 'how did this feel?' enabled")

        logger.info(f"Stanley awakened. Vocab: {self.vocab.vocab_size}, "
                   f"Training: {self.config.training_enabled and TORCH_AVAILABLE}, "
                   f"Subword: {self.subword_field is not None}, "
                   f"Subjectivity: {self.subjectivity is not None}, "
                   f"Overthinking: {self.overthinking is not None}, "
                   f"BodySense: {self.body_sense is not None}")

    def _default_origin(self) -> str:
        """Default origin text if none provided."""
        return """
        I am Stanley.
        I grow through experience.
        Every conversation leaves a trace.
        I remember what resonates.
        I am not all my memories at once.
        I think with what's needed now.
        """

    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    def _setup_trainer(self):
        """Setup the micro-trainer."""
        from .trainer import LoRAConfig

        base_weights = {
            "vocab_size": self.vocab.vocab_size,
            "n_emb": self.config.n_emb,
            "T": self.config.context_length,
            "nodes": self.config.nodes,
            "n_blocks": self.config.n_blocks,
            "n_heads": self.config.n_heads,
        }

        trainer_config = TrainerConfig(
            lora_config=LoRAConfig(rank=self.config.lora_rank),
        )

        self.trainer = MicroTrainer(
            base_weights=base_weights,
            vocab=self.vocab,
            config=trainer_config,
            on_training_complete=self._on_training_complete,
        )

    def _on_training_complete(self, result):
        """Callback when training batch completes."""
        logger.info(f"Training complete: {len(result.items)} items, "
                   f"{result.training_time:.2f}s")

        # Auto-swap if successful
        if result.success and self.trainer:
            self.trainer.swap_weights()

            # Apply new deltas to transformer
            active_deltas = self.trainer.get_active_deltas()
            if active_deltas:
                self.transformer.apply_shard_deltas(active_deltas)

    def _get_age(self) -> float:
        """
        Get organism age as 0-1 value.

        Used for adaptive buffer thresholds.
        """
        total_shards = self.memory.total_shards()

        # Consider "mature" at 100 shards
        maturity_threshold = 100
        age = min(1.0, total_shards / maturity_threshold)

        return age

    def experience(self, interaction: str) -> Optional[Shard]:
        """
        Process an interaction — maybe create a shard.

        This is selective memory: not everything becomes a shard.
        Only what resonates, only what's novel.
        """
        self.last_interaction = time.time()
        self.total_interactions += 1

        # Compute resonance with origin
        interaction_fp = compute_fingerprint(interaction)
        resonance_with_origin = float(np.dot(interaction_fp, self.origin_fingerprint))

        # Compute novelty (how different from existing shards)
        existing_fps = [s.trigger_fingerprint for s in self.memory.surface + self.memory.middle]
        if existing_fps:
            similarities = [float(np.dot(interaction_fp, fp)) for fp in existing_fps]
            novelty = 1.0 - max(similarities)
        else:
            novelty = 1.0  # Everything is novel at first

        # Should we remember this?
        if not should_remember(
            resonance=resonance_with_origin,
            novelty=novelty,
            filter_config=self.experience_filter,
        ):
            logger.debug(f"Forgetting: resonance={resonance_with_origin:.2f}, novelty={novelty:.2f}")
            return None

        # Create shard
        # For now, use empty deltas (real training happens async)
        from .trainer import create_empty_deltas

        model_config = {
            "vocab_size": self.vocab.vocab_size,
            "n_emb": self.config.n_emb,
            "T": self.config.context_length,
            "nodes": self.config.nodes,
            "n_blocks": self.config.n_blocks,
            "n_heads": self.config.n_heads,
        }

        shard = Shard.create(
            content=interaction,
            resonance=resonance_with_origin,
            layer_deltas=create_empty_deltas(model_config),
            fingerprint=interaction_fp,
        )

        # Add to memory
        self.memory.add(shard)

        # Add to quantum buffer
        triggered = self.buffer.add(shard)

        # Submit for training (async)
        if self.trainer and self.config.training_enabled:
            self.trainer.submit(
                content=interaction,
                resonance=resonance_with_origin,
            )

        logger.info(f"Remembered: shard {shard.id[:8]}, "
                   f"resonance={resonance_with_origin:.2f}, novelty={novelty:.2f}")

        # Check if quantum threshold reached
        if triggered:
            self._process_quantum()

        return shard

    def _process_quantum(self):
        """Process accumulated quantum — trigger training batch."""
        shards = self.buffer.flush()
        if not shards:
            return

        logger.info(f"Quantum triggered: {len(shards)} shards")

        # Training happens async via MicroTrainer
        # The callback will handle weight swapping

    def think(self, prompt: str, length: int = 100) -> Tuple[str, dict]:
        """
        Generate a response with personality.

        This is how Stanley speaks — through its accumulated experience.

        KEY PRINCIPLE: "NO SEED FROM PROMPT"
        The user's words don't become Stanley's words.
        They wrinkle the field — create ripples that influence
        what emerges from WITHIN.
        """
        stats = {}

        # === SUBJECTIVITY: User input → Pulse (NOT seed!) ===
        pulse = None
        internal_seed = None

        if self.subjectivity:
            # Compute the "wrinkle" from user input
            pulse = self.subjectivity.compute_pulse(prompt)
            stats["pulse"] = {
                "novelty": pulse.novelty,
                "arousal": pulse.arousal,
                "entropy": pulse.entropy,
                "valence": pulse.valence,
            }

            # Get internal seed — from identity, NOT from prompt!
            internal_seed = self.subjectivity.get_internal_seed(prompt, pulse)
            stats["internal_seed"] = internal_seed

            # === EXPERT ROUTING: Dynamic temperature from pulse signals ===
            # This is MOE-style routing - temperature emerges from expert blend
            mixture = route_from_pulse(pulse)
            temperature = mixture.temperature
            stats["temperature"] = temperature
            stats["expert_mixture"] = {
                "dominant": mixture.dominant,
                "weights": mixture.weights,
                "semantic_weight": mixture.semantic_weight,
            }
        else:
            temperature = self.config.subword_temperature

        # === RESONANT RECALL: SantaClaus brings back memories ===
        # Recall resonant shards to influence generation
        recall_context = None
        if self.resonant_recall:
            recall_context = self.resonant_recall.recall(
                prompt=prompt,
                pulse=pulse,
            )
            if recall_context:
                stats["recall"] = {
                    "count": len(recall_context.recalled_texts),
                    "is_silly": recall_context.is_silly,
                    "total_score": recall_context.total_score,
                    "shard_ids": recall_context.recalled_shard_ids,
                }
                # Optionally enrich internal seed with recalled memories
                if recall_context.recalled_texts and internal_seed:
                    # Add a hint from recalled memory to the seed
                    recalled_hint = recall_context.recalled_texts[0][:30]
                    internal_seed = f"{internal_seed} {recalled_hint}"

        # === COHERENT GENERATION via SubwordField ===
        if self.subword_field:
            # Generate from INTERNAL seed, not user prompt!
            seed = internal_seed if internal_seed else "I"

            response = self.subword_field.generate(
                seed_text=seed,
                length=length,
                temperature=temperature,
            )

            stats["method"] = "subword_field"
            stats["subword_stats"] = self.subword_field.stats()

        else:
            # Fallback to old inference engine
            self.engine.load_working_set(prompt, self.config.max_working_set)
            response, engine_stats = self.engine.think(
                prompt=prompt,
                length=length,
                sampling="entropy",
                auto_load=False,
            )
            stats["method"] = "inference_engine"
            stats.update(engine_stats)

        # === WRINKLE FIELD: Response becomes part of identity ===
        if self.subjectivity and pulse:
            self.subjectivity.wrinkle_field(response, pulse)
            stats["identity_fragments"] = len(self.subjectivity.identity.fragments)
            stats["gravity_centers"] = len(self.subjectivity.identity.gravity_centers)

        # === OVERTHINKING: Circles on water (dynamic inner reflection) ===
        # Generates private reflections that enrich the field
        # Ring count is DYNAMIC based on pulse entropy/arousal
        # NOW WITH CRYSTALLIZATION: deep rings become internal shards!
        crystallized = False
        if self.overthinking and response:
            rings_snapshot = self.overthinking.generate_rings(
                source_text=response,
                pulse=pulse,
            )
            crystallized = self.overthinking.crystallization_count > 0
            stats["overthinking"] = {
                "ring_count": len(rings_snapshot.rings),
                "ring_names": [r.name for r in rings_snapshot.rings],
                "enrichment_count": self.overthinking.enrichment_count,
                "emergent_trigrams": len(self.overthinking.emergent_trigrams),
                "crystallization_count": self.overthinking.crystallization_count,
            }

        # === BODY SENSE: Regulate based on boredom/overwhelm/stuck ===
        regulation = None
        if self.body_sense and pulse:
            # Create body state from current conditions
            body_state = BodyState.from_pulse(
                pulse=pulse,
                memory=self.memory,
                overthinking_stats=self.overthinking.get_stats() if self.overthinking else None,
            )

            # Get regulation suggestion
            current_expert = stats.get("expert_mixture", {}).get("dominant", "structural")
            regulation = self.body_sense.regulate(
                body_state,
                current_temperature=temperature,
                current_expert=current_expert,
            )
            stats["body_sense"] = {
                "boredom": regulation.boredom,
                "overwhelm": regulation.overwhelm,
                "stuck": regulation.stuck,
                "predicted_quality": regulation.predicted_quality,
            }

        # === SEMANTIC DRIFT: Log step for trajectory learning ===
        if self.semantic_drift and pulse:
            # Get active tags from recent shards
            active_tags = []
            if self.memory.surface:
                for shard in self.memory.surface[:5]:
                    active_tags.extend(shard.semantic_tags)
            active_tags = list(set(active_tags))[:10]

            self.semantic_drift.log_step(
                metrics={"entropy": pulse.entropy, "arousal": pulse.arousal, "novelty": pulse.novelty},
                active_tags=active_tags,
                resonance=pulse.arousal,
                overthinking_depth=len(rings_snapshot.rings) if self.overthinking else 0,
                crystallized=crystallized,
            )
            stats["semantic_drift"] = self.semantic_drift.get_stats()

        # === SOMATIC MEMORY: Record how this moment felt ===
        if self.somatic_memory and pulse:
            # Use regulation quality or default
            outcome_quality = regulation.predicted_quality if regulation else 0.5
            outcome_tag = "neutral"
            if regulation:
                if regulation.boredom > 0.6:
                    outcome_tag = "bored"
                elif regulation.overwhelm > 0.6:
                    outcome_tag = "overwhelmed"
                elif regulation.stuck > 0.6:
                    outcome_tag = "stuck"
                elif regulation.predicted_quality > 0.7:
                    outcome_tag = "good"

            self.somatic_memory.record_moment(
                entropy=pulse.entropy,
                novelty=pulse.novelty,
                arousal=pulse.arousal,
                valence=pulse.valence,
                outcome_quality=outcome_quality,
                outcome_tag=outcome_tag,
            )
            stats["somatic_memory"] = self.somatic_memory.get_stats()

        # Mark useful shards (for router learning)
        for shard in self.engine.working_set:
            if isinstance(self.router, AdaptiveRouter):
                self.router.mark_useful(shard.id)

        # === CLEANUP: Apply Haze-style coherence magic ===
        # 1. Truncate at natural sentence boundary (no trailing "...")
        response = truncate_at_natural_end(response, max_length=len(response))
        # 2. Fix contractions, repetitions, capitalization
        response = cleanup_output(response, mode="moderate")

        return response, stats

    def grow(self):
        """
        Growth cycle — consolidation, training checks.

        Should be called periodically (e.g., every few minutes).
        """
        now = time.time()

        # Consolidation
        if now - self.last_consolidation > self.config.consolidation_interval:
            self.consolidator.run(
                surface=self.memory.surface,
                middle=self.memory.middle,
                deep=self.memory.deep,
                abyss=self.memory.abyss,
            )
            self.last_consolidation = now

        # Memory cleanup
        self.memory.consolidate()

    def stats(self) -> dict:
        """Get organism statistics."""
        return {
            "age_seconds": time.time() - self.created_at,
            "total_interactions": self.total_interactions,
            "vocab_size": self.vocab.vocab_size,
            "memory": self.memory.stats(),
            "buffer": self.buffer.stats(),
            "trainer": self.trainer.stats() if self.trainer else None,
            "consolidator": self.consolidator.stats(),
            "working_set_size": len(self.engine.working_set),
            "maturity": self._get_age(),
            "subword_field": self.subword_field.stats() if self.subword_field else None,
            "subjectivity": self.subjectivity.stats() if self.subjectivity else None,
        }

    def save(self, path: Optional[str] = None):
        """Save organism state."""
        save_path = Path(path or self.config.data_dir or "./stanley_data")
        save_path.mkdir(parents=True, exist_ok=True)

        # Save memory
        self.memory.storage_path = save_path / "memory"
        self.memory.save_all()

        # Save transformer
        self.transformer.save_base_weights(save_path / "transformer.npz")

        # Save config and state
        state = {
            "created_at": self.created_at,
            "total_interactions": self.total_interactions,
            "origin_text": self.origin_text,
            "config": {
                "vocab_size": self.config.vocab_size,
                "n_emb": self.config.n_emb,
                "n_blocks": self.config.n_blocks,
                "n_heads": self.config.n_heads,
                "context_length": self.config.context_length,
            },
        }

        with open(save_path / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved Stanley to {save_path}")

    @classmethod
    def load(cls, path: str) -> "Stanley":
        """Load organism from saved state."""
        load_path = Path(path)

        # Load state
        with open(load_path / "state.json") as f:
            state = json.load(f)

        # Create config
        config = StanleyConfig(
            data_dir=str(load_path),
            **state.get("config", {}),
        )

        # Create organism
        stanley = cls(
            config=config,
            origin_text=state.get("origin_text"),
        )

        # Restore state
        stanley.created_at = state.get("created_at", time.time())
        stanley.total_interactions = state.get("total_interactions", 0)

        # Load memory
        if (load_path / "memory").exists():
            stanley.memory = MemorySea.load(load_path / "memory")
            stanley.engine.memory = stanley.memory

        # Load transformer
        if (load_path / "transformer.npz").exists():
            stanley.transformer = StanleyTransformer.load_base_weights(
                load_path / "transformer.npz"
            )
            stanley.engine.transformer = stanley.transformer

        logger.info(f"Loaded Stanley from {load_path}")
        return stanley

    def __repr__(self) -> str:
        return (
            f"Stanley(shards={self.memory.total_shards()}, "
            f"interactions={self.total_interactions}, "
            f"maturity={self._get_age():.2f})"
        )


def create_stanley(
    origin_text: Optional[str] = None,
    data_dir: Optional[str] = None,
    **kwargs,
) -> Stanley:
    """Convenience function to create a Stanley instance."""
    config = StanleyConfig(data_dir=data_dir, **kwargs)
    return Stanley(config=config, origin_text=origin_text)


def load_or_create(
    path: str,
    origin_text: Optional[str] = None,
) -> Stanley:
    """Load Stanley from path, or create new if doesn't exist."""
    path = Path(path)

    if (path / "state.json").exists():
        return Stanley.load(str(path))
    else:
        return create_stanley(origin_text=origin_text, data_dir=str(path))

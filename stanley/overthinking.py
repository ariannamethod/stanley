"""
overthinking.py — Dynamic Rings of Private Reflection for Stanley

Ported from Haze/Leo "circles on water" with Stanley-specific enhancements:
- DYNAMIC ring count based on pulse entropy/arousal
- Integration with SubwordField for coherent generation
- Field enrichment through emergent patterns
- CRYSTALLIZATION: Resonant rings become internal shards!

The rings are PRIVATE REFLECTIONS - never shown to user.
They influence the next generation through field state.

KEY INSIGHT: The internal world becomes RICHER than the training data!
Rings generate NEW patterns that are injected back into the field.

CRYSTALLIZATION (Stanley innovation):
    Highly resonant rings (depth >= 3, many meta-patterns) can
    crystallize into "internal shards" — Stanley's own memories
    of his thoughts. These are tagged "internal" and join MemorySea.

    experience → external shards (what happened)
    overthinking → internal shards (what Stanley thought)

    Both types can be recalled and influence training!

Dynamic Ring Count (Stanley innovation):
    - Low entropy (< 0.3): 1 ring (Echo only - stay grounded)
    - Medium entropy (0.3-0.6): 2 rings (Echo + Drift)
    - High entropy (0.6-0.8): 3 rings (Echo + Drift + Shard)
    - Very high entropy (> 0.8): 4-5 rings (deep reflection needed)

"The model thinks about what it just said."
"""

from __future__ import annotations

import re
import numpy as np
from typing import List, Tuple, Optional, Dict, TYPE_CHECKING
from dataclasses import dataclass, field as dataclass_field
from collections import Counter
import logging

if TYPE_CHECKING:
    from .subword_field import SubwordField
    from .subjectivity import Pulse
    from .memory_sea import MemorySea

logger = logging.getLogger(__name__)

# Crystallization threshold - rings with this many meta-patterns crystallize
CRYSTALLIZATION_META_THRESHOLD = 3
# Minimum ring depth for crystallization
CRYSTALLIZATION_DEPTH_THRESHOLD = 3
# Probability of crystallization when thresholds are met
CRYSTALLIZATION_PROBABILITY = 0.3


# Ring configuration - temperatures increase with depth
# Deeper rings are more exploratory/abstract
RING_CONFIGS = {
    0: {
        "name": "echo",
        "description": "Rephrase what was generated - grounding",
        "temperature": 0.7,
        "length": 25,
    },
    1: {
        "name": "drift",
        "description": "Explore tangential themes - association",
        "temperature": 0.9,
        "length": 30,
    },
    2: {
        "name": "shard",
        "description": "Abstract meta-note - crystallization",
        "temperature": 1.1,
        "length": 20,
    },
    3: {
        "name": "deep",
        "description": "Deep resonance - emergence",
        "temperature": 1.3,
        "length": 15,
    },
    4: {
        "name": "void",
        "description": "Edge of coherence - pure drift",
        "temperature": 1.5,
        "length": 10,
    },
}


@dataclass
class Ring:
    """Single overthinking ring."""
    level: int
    name: str
    content: str
    temperature: float
    trigrams: List[Tuple[str, str, str]] = dataclass_field(default_factory=list)

    def __repr__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Ring({self.level}/{self.name}: \"{preview}\")"


@dataclass
class RingsSnapshot:
    """
    Complete overthinking snapshot.
    Contains all rings generated after a response.
    """
    rings: List[Ring] = dataclass_field(default_factory=list)
    source_text: str = ""
    pulse_entropy: float = 0.5
    pulse_arousal: float = 0.5

    @property
    def echo(self) -> Optional[Ring]:
        """Get ring 0 (echo)."""
        return next((r for r in self.rings if r.level == 0), None)

    @property
    def drift(self) -> Optional[Ring]:
        """Get ring 1 (drift)."""
        return next((r for r in self.rings if r.level == 1), None)

    @property
    def shard(self) -> Optional[Ring]:
        """Get ring 2 (shard)."""
        return next((r for r in self.rings if r.level == 2), None)

    @property
    def deep(self) -> Optional[Ring]:
        """Get ring 3 (deep)."""
        return next((r for r in self.rings if r.level == 3), None)

    @property
    def void(self) -> Optional[Ring]:
        """Get ring 4 (void)."""
        return next((r for r in self.rings if r.level == 4), None)

    def get_all_trigrams(self) -> List[Tuple[str, str, str]]:
        """Get combined trigrams from all rings."""
        result = []
        for ring in self.rings:
            result.extend(ring.trigrams)
        return result

    def get_influence_words(self) -> List[str]:
        """Get words from rings to influence next generation."""
        words = []
        for ring in self.rings:
            ring_words = re.findall(r'\b\w+\b', ring.content.lower())
            words.extend(ring_words)
        return words

    @property
    def depth(self) -> int:
        """How deep did the reflection go?"""
        return len(self.rings)


def compute_ring_count(pulse: Optional["Pulse"]) -> int:
    """
    Compute dynamic ring count based on pulse.

    Stanley innovation: Ring count scales with field state!

    Low entropy → fewer rings (stay coherent)
    High entropy → more rings (need processing)
    High arousal → more rings (emotional intensity)

    Args:
        pulse: Current pulse state (or None for default)

    Returns:
        Number of rings to generate (1-5)
    """
    if pulse is None:
        return 3  # Default: standard 3 rings

    # Base from entropy
    entropy = pulse.entropy
    arousal = pulse.arousal

    if entropy < 0.3:
        base_rings = 1  # Very grounded, minimal reflection
    elif entropy < 0.5:
        base_rings = 2  # Moderate, echo + drift
    elif entropy < 0.7:
        base_rings = 3  # Standard reflection
    elif entropy < 0.85:
        base_rings = 4  # High entropy needs processing
    else:
        base_rings = 5  # Maximum reflection

    # Arousal modifier: high arousal adds rings
    if arousal > 0.7:
        base_rings = min(5, base_rings + 1)
    elif arousal > 0.5:
        base_rings = min(5, base_rings)  # No change
    elif arousal < 0.2:
        base_rings = max(1, base_rings - 1)  # Very calm, fewer rings

    return base_rings


class Overthinking:
    """
    Dynamic private reflection generator for Stanley.

    Creates "rings on water" after each generation:
    - Ring 0 (Echo): Rephrase (temp=0.7)
    - Ring 1 (Drift): Tangential themes (temp=0.9)
    - Ring 2 (Shard): Abstract meta-note (temp=1.1)
    - Ring 3 (Deep): Deep resonance (temp=1.3)
    - Ring 4 (Void): Edge of coherence (temp=1.5)

    Number of rings is DYNAMIC based on pulse entropy/arousal!

    KEY: These rings ENRICH THE FIELD!
    - Rings generate NEW patterns not in original corpus
    - These patterns are INJECTED back into the subword field
    - Inner world becomes RICHER than the dataset!

    "The internal world is richer than the training data."
    """

    def __init__(
        self,
        subword_field: "SubwordField",
        memory_sea: Optional["MemorySea"] = None,
    ):
        """
        Initialize overthinking module.

        Args:
            subword_field: SubwordField for generation AND enrichment
            memory_sea: Optional MemorySea for crystallization
        """
        self.field = subword_field
        self.memory = memory_sea  # For crystallization

        # Ring history (for meta-analysis)
        self.ring_history: List[RingsSnapshot] = []

        # Meta patterns that emerge from rings
        self.meta_patterns: List[str] = []

        # Patterns generated by overthinking (emergent vocabulary)
        self.emergent_trigrams: List[Tuple[str, str, str]] = []
        self.enrichment_count: int = 0

        # Track ring depth over time
        self.depth_history: List[int] = []

        # Crystallization stats
        self.crystallization_count: int = 0

    def _extract_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract trigrams from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i+1], words[i+2]))
        return trigrams

    def _inject_trigram_into_field(self, trigram: Tuple[str, str, str]) -> bool:
        """
        Inject a trigram from overthinking into the subword field.

        This is EMERGENCE - the internal world becomes richer than the dataset!

        Returns:
            True if successfully injected
        """
        # Encode each word
        w1_tokens = self.field.vocab.encode(trigram[0])
        w2_tokens = self.field.vocab.encode(trigram[1])
        w3_tokens = self.field.vocab.encode(trigram[2])

        if not w1_tokens or not w2_tokens or not w3_tokens:
            return False

        # Get boundary tokens for bigram injection
        last_w1 = w1_tokens[-1]
        first_w2 = w2_tokens[0]
        last_w2 = w2_tokens[-1]
        first_w3 = w3_tokens[0]

        # Inject into bigram counts (with lower weight - emergent patterns are softer)
        self.field.bigram_counts[last_w1][first_w2] += 1
        self.field.bigram_totals[last_w1] += 1

        self.field.bigram_counts[last_w2][first_w3] += 1
        self.field.bigram_totals[last_w2] += 1

        # Track emergent patterns
        if trigram not in self.emergent_trigrams:
            self.emergent_trigrams.append(trigram)
            self.enrichment_count += 1

        # Keep reasonable size
        if len(self.emergent_trigrams) > 500:
            self.emergent_trigrams = self.emergent_trigrams[-500:]

        return True

    def _enrich_field_from_ring(self, ring: Ring) -> int:
        """
        Enrich the field with patterns from a ring.

        Returns:
            Number of patterns injected
        """
        injected = 0
        for trigram in ring.trigrams:
            if self._inject_trigram_into_field(trigram):
                injected += 1
        return injected

    def _generate_ring_content(
        self,
        seed_text: str,
        config: dict,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """
        Generate content for a single ring.

        Uses SubwordField for coherent generation.
        """
        rng = rng or np.random.default_rng()

        # Generate from subword field
        content = self.field.generate(
            seed_text=seed_text,
            length=config["length"],
            temperature=config["temperature"],
            rng=rng,
            adaptive_temp=False,  # Use fixed temp for rings
        )

        return content

    def generate_rings(
        self,
        source_text: str,
        pulse: Optional["Pulse"] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> RingsSnapshot:
        """
        Generate overthinking rings from source text.

        These are PRIVATE REFLECTIONS - never shown to user.
        They influence the next generation through field state.

        Args:
            source_text: The generated text to reflect on
            pulse: Current pulse for dynamic ring count
            rng: Random generator for reproducibility

        Returns:
            RingsSnapshot with all rings
        """
        rng = rng or np.random.default_rng()

        # DYNAMIC ring count based on pulse!
        num_rings = compute_ring_count(pulse)
        self.depth_history.append(num_rings)
        if len(self.depth_history) > 50:
            self.depth_history = self.depth_history[-50:]

        entropy_val = pulse.entropy if pulse else 0.5
        logger.debug(f"Overthinking: {num_rings} rings (entropy={entropy_val:.2f})")

        # Extract key patterns from source
        source_words = re.findall(r'\b\w+\b', source_text.lower())

        rings = []

        for level in range(num_rings):
            if level not in RING_CONFIGS:
                break

            config = RING_CONFIGS[level]

            # Determine seed based on ring level
            if level == 0:
                # Echo: seed from end of source
                seed = ' '.join(source_words[-5:]) if len(source_words) >= 5 else source_text[:30]
            elif level == 1:
                # Drift: random word from source
                seed = rng.choice(source_words) if source_words else "I"
            elif level == 2:
                # Shard: from previous ring or meta-patterns
                if self.meta_patterns:
                    seed = rng.choice(self.meta_patterns[-10:])
                elif rings:
                    seed = rings[-1].content[-30:]
                else:
                    seed = source_text[:20]
            elif level == 3:
                # Deep: from emergent trigrams or ring chain
                if self.emergent_trigrams:
                    trigram = rng.choice(self.emergent_trigrams[-20:])
                    seed = ' '.join(trigram)
                elif rings:
                    seed = rings[-1].content[:20]
                else:
                    seed = "resonance"
            else:
                # Void: minimal seed, pure drift
                seed = rng.choice(["I", "the", "and", "what", "why"])

            content = self._generate_ring_content(seed, config, rng)

            ring = Ring(
                level=level,
                name=config["name"],
                content=content,
                temperature=config["temperature"],
                trigrams=self._extract_trigrams(content),
            )
            rings.append(ring)

        # Create snapshot
        snapshot = RingsSnapshot(
            rings=rings,
            source_text=source_text,
            pulse_entropy=pulse.entropy if pulse else 0.5,
            pulse_arousal=pulse.arousal if pulse else 0.5,
        )

        # Store in history
        self.ring_history.append(snapshot)
        if len(self.ring_history) > 20:
            self.ring_history = self.ring_history[-20:]

        # Extract meta-patterns from this reflection
        self._update_meta_patterns(snapshot)

        # EMERGENCE: Enrich the field with patterns from rings!
        total_injected = 0
        for ring in rings:
            injected = self._enrich_field_from_ring(ring)
            total_injected += injected

        logger.debug(f"Field enrichment: +{total_injected} patterns from {num_rings} rings")

        # CRYSTALLIZATION: Deep reflections become internal shards!
        crystallized = self._crystallize_snapshot(snapshot, rng)
        if crystallized:
            logger.debug(f"Ring snapshot crystallized into internal shard!")

        return snapshot

    def _update_meta_patterns(self, snapshot: RingsSnapshot) -> None:
        """Update meta-patterns from ring content."""
        # Find words that appear in multiple rings
        word_counts: Counter = Counter()

        for ring in snapshot.rings:
            words = set(re.findall(r'\b\w+\b', ring.content.lower()))
            for word in words:
                word_counts[word] += 1

        # Words appearing in 2+ rings are "meta"
        for word, count in word_counts.items():
            if count >= 2 and len(word) > 3:
                self.meta_patterns.append(word)

        # Keep reasonable size
        self.meta_patterns = self.meta_patterns[-100:]

    def _crystallize_snapshot(
        self,
        snapshot: RingsSnapshot,
        rng: Optional[np.random.Generator] = None,
    ) -> bool:
        """
        Crystallize a deep/resonant snapshot into an internal shard.

        CRYSTALLIZATION: When overthinking is deep and rich with meta-patterns,
        the reflection crystallizes into Stanley's own memory — an "internal shard".

        These internal shards:
        - Are tagged "internal" to distinguish from external experience
        - Join MemorySea and can be recalled/trained on
        - Represent Stanley's THOUGHTS about things, not just what happened

        Args:
            snapshot: The ring snapshot to potentially crystallize
            rng: Random generator

        Returns:
            True if crystallized, False otherwise
        """
        if self.memory is None:
            return False  # No memory sea = no crystallization

        rng = rng or np.random.default_rng()

        # Check depth threshold
        if snapshot.depth < CRYSTALLIZATION_DEPTH_THRESHOLD:
            return False

        # Check meta-patterns threshold
        # Count meta-patterns that appeared in THIS snapshot
        snapshot_words = set()
        for ring in snapshot.rings:
            words = set(re.findall(r'\b\w+\b', ring.content.lower()))
            snapshot_words.update(words)

        # Count how many of our meta-patterns are in this snapshot
        meta_in_snapshot = len([p for p in self.meta_patterns if p in snapshot_words])

        if meta_in_snapshot < CRYSTALLIZATION_META_THRESHOLD:
            return False

        # Probabilistic crystallization
        if rng.random() > CRYSTALLIZATION_PROBABILITY:
            return False

        # CRYSTALLIZE! Create internal shard
        # Combine content from all rings (weighted by depth)
        combined_content = []
        for ring in snapshot.rings:
            # Deeper rings contribute more to the crystal
            weight = ring.level + 1
            combined_content.extend([ring.content] * weight)

        crystal_content = " ".join(combined_content)

        # Truncate if too long
        if len(crystal_content) > 500:
            crystal_content = crystal_content[:500]

        # Compute resonance score based on depth and meta-patterns
        resonance = min(1.0, 0.3 + snapshot.depth * 0.1 + meta_in_snapshot * 0.05)

        # Import Shard here to avoid circular import
        from .shard import Shard

        # Create internal shard with special tags
        shard = Shard.create(
            content=crystal_content,
            resonance=resonance,
            layer_deltas={},  # Internal shards start without deltas
            tags=["internal", "overthinking", f"depth_{snapshot.depth}"],
        )

        # Add to memory sea surface
        self.memory.surface.append(shard)
        self.crystallization_count += 1

        logger.info(
            f"Crystallized internal shard: id={shard.id}, "
            f"depth={snapshot.depth}, resonance={resonance:.2f}, "
            f"meta_patterns={meta_in_snapshot}"
        )

        return True

    def get_field_influence(self) -> Dict:
        """
        Get influence data for the next generation.

        Returns patterns and words that should bias the next response.
        """
        if not self.ring_history:
            return {"words": [], "trigrams": [], "temperature_mod": 0.0, "depth": 0}

        # Get recent rings
        recent = self.ring_history[-3:]

        # Collect influence words
        influence_words = []
        influence_trigrams = []
        total_depth = 0

        for snapshot in recent:
            influence_words.extend(snapshot.get_influence_words())
            influence_trigrams.extend(snapshot.get_all_trigrams())
            total_depth += snapshot.depth

        avg_depth = total_depth / len(recent)

        # Temperature modification based on depth and variety
        if avg_depth >= 4:
            # Deep reflection = slightly higher temp (exploration mode)
            temp_mod = 0.1
        elif avg_depth <= 1:
            # Shallow reflection = slightly lower temp (grounding mode)
            temp_mod = -0.1
        else:
            # Moderate
            temp_mod = 0.0

        # Variety bonus
        if len(set(influence_words)) > 30:
            temp_mod += 0.05

        return {
            "words": influence_words[-50:],
            "trigrams": influence_trigrams[-20:],
            "temperature_mod": temp_mod,
            "depth": avg_depth,
        }

    def bias_next_generation(
        self,
        probs: np.ndarray,
        influence_alpha: float = 0.1,
    ) -> np.ndarray:
        """
        Bias probability distribution based on overthinking influence.

        Args:
            probs: Probability distribution for next token
            influence_alpha: How much to bias (0 = none, 1 = full)

        Returns:
            Biased probabilities
        """
        if not self.ring_history:
            return probs

        influence = self.get_field_influence()
        influence_words = influence["words"]

        if not influence_words:
            return probs

        # Create bias vector
        bias = np.zeros(len(probs), dtype=np.float32)

        # Boost tokens that appear in influence words
        for word in influence_words:
            tokens = self.field.vocab.encode(word)
            for token in tokens:
                if token < len(bias):
                    bias[token] += 0.1

        # Normalize bias
        if bias.sum() > 0:
            bias = bias / bias.sum()

        # Blend with original probs
        biased = (1 - influence_alpha) * probs + influence_alpha * (probs + bias)

        # Renormalize
        biased = biased / biased.sum()

        return biased

    def get_stats(self) -> Dict:
        """
        Get statistics about overthinking activity.

        Returns:
            Dict with overthinking metrics
        """
        avg_depth = np.mean(self.depth_history) if self.depth_history else 0

        return {
            "total_emergent_trigrams": len(self.emergent_trigrams),
            "enrichment_count": self.enrichment_count,
            "meta_patterns": len(self.meta_patterns),
            "ring_sessions": len(self.ring_history),
            "average_depth": round(avg_depth, 2),
            "recent_depths": self.depth_history[-10:] if self.depth_history else [],
            "sample_emergent": self.emergent_trigrams[-5:] if self.emergent_trigrams else [],
            "crystallization_count": self.crystallization_count,
        }

    def __repr__(self) -> str:
        avg_depth = np.mean(self.depth_history) if self.depth_history else 0
        return (f"Overthinking(sessions={len(self.ring_history)}, "
                f"avg_depth={avg_depth:.1f}, "
                f"emergent={len(self.emergent_trigrams)})")


# ============================================================
#  ASYNC DISCIPLINE — Lock-protected field operations
# ============================================================

import asyncio


class AsyncOverthinking:
    """
    Async-safe wrapper for Overthinking with field lock.

    Maintains coherence through atomic operations.
    Use this when running Stanley in async contexts (servers, etc.)
    """

    def __init__(
        self,
        subword_field: "SubwordField",
        memory_sea: Optional["MemorySea"] = None,
    ):
        self._sync = Overthinking(subword_field, memory_sea)
        self._field_lock = asyncio.Lock()

    @property
    def ring_history(self) -> List[RingsSnapshot]:
        return self._sync.ring_history

    @property
    def meta_patterns(self) -> List[str]:
        return self._sync.meta_patterns

    @property
    def emergent_trigrams(self) -> List[Tuple[str, str, str]]:
        return self._sync.emergent_trigrams

    @property
    def enrichment_count(self) -> int:
        return self._sync.enrichment_count

    @property
    def depth_history(self) -> List[int]:
        return self._sync.depth_history

    @property
    def crystallization_count(self) -> int:
        return self._sync.crystallization_count

    async def generate_rings(
        self,
        source_text: str,
        pulse: Optional["Pulse"] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> RingsSnapshot:
        """Generate rings with atomic field access."""
        async with self._field_lock:
            return self._sync.generate_rings(source_text, pulse, rng)

    async def get_field_influence(self) -> Dict:
        """Get influence data atomically."""
        async with self._field_lock:
            return self._sync.get_field_influence()

    async def bias_next_generation(
        self,
        probs: np.ndarray,
        influence_alpha: float = 0.1,
    ) -> np.ndarray:
        """Bias probabilities atomically."""
        async with self._field_lock:
            return self._sync.bias_next_generation(probs, influence_alpha)

    async def get_stats(self) -> Dict:
        """Get stats atomically."""
        async with self._field_lock:
            return self._sync.get_stats()

    def __repr__(self) -> str:
        return f"Async{repr(self._sync)}"


if __name__ == "__main__":
    print("=== Stanley Overthinking Demo ===")
    print()
    print("Dynamic ring count based on entropy/arousal:")
    print("  Low entropy (<0.3): 1 ring")
    print("  Medium (0.3-0.6): 2 rings")
    print("  High (0.6-0.8): 3 rings")
    print("  Very high (>0.8): 4-5 rings")
    print()
    print("Rings enrich the field - internal world becomes richer than dataset!")
    print()
    print("CRYSTALLIZATION (Stanley innovation):")
    print(f"  Meta threshold: {CRYSTALLIZATION_META_THRESHOLD}")
    print(f"  Depth threshold: {CRYSTALLIZATION_DEPTH_THRESHOLD}")
    print(f"  Probability: {CRYSTALLIZATION_PROBABILITY}")
    print()
    print("  experience -> external shards (what happened)")
    print("  overthinking -> internal shards (what Stanley thought)")
    print()
    print("  Both types can be recalled and influence training!")

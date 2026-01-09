"""
experience.py — Selective memory for Stanley

Not everything becomes a shard.
Only what resonates. Only what's novel. Only what matters.

This is the filter between raw interaction and lasting memory.
Like human attention: we don't remember everything,
we remember what stood out.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
import logging

from .fingerprint import compute_fingerprint, cosine_similarity

logger = logging.getLogger(__name__)


@dataclass
class ExperienceFilter:
    """
    Configuration for what experiences become shards.

    The filter uses multiple criteria:
    - Resonance: how much it connects to origin identity
    - Novelty: how different from existing memories
    - Emotional weight: detected sentiment intensity
    - Length: very short or very long interactions
    """

    # Thresholds (all must be met)
    min_resonance: float = 0.2       # minimum connection to origin
    min_novelty: float = 0.15        # minimum difference from existing

    # Optional emotional detection
    emotional_weight: float = 0.3    # boost for emotional content
    emotional_keywords: List[str] = None

    # Length filters
    min_length: int = 10             # minimum characters
    max_length: int = 10000          # maximum characters

    # Randomness (for exploration)
    random_remember_prob: float = 0.05  # chance to remember anyway

    def __post_init__(self):
        if self.emotional_keywords is None:
            self.emotional_keywords = [
                # Positive
                "love", "joy", "happy", "wonderful", "amazing", "beautiful",
                "grateful", "excited", "hope", "dream",
                # Negative
                "sad", "angry", "fear", "worried", "pain", "hurt",
                "frustrated", "anxious", "scared", "lost",
                # Intense
                "always", "never", "forever", "absolutely", "completely",
                # Connection
                "understand", "feel", "believe", "trust", "remember",
            ]


def should_remember(
    resonance: float,
    novelty: float,
    filter_config: Optional[ExperienceFilter] = None,
    content: Optional[str] = None,
    rng: Optional[np.random.Generator] = None,
) -> bool:
    """
    Decide if an interaction should become a shard.

    Args:
        resonance: similarity to origin identity (0-1)
        novelty: difference from existing memories (0-1)
        filter_config: filter configuration
        content: optional content for emotional analysis
        rng: random generator for exploration

    Returns:
        True if should remember, False if should forget
    """
    cfg = filter_config or ExperienceFilter()
    rng = rng or np.random.default_rng()

    # Random exploration — sometimes remember anyway
    if rng.random() < cfg.random_remember_prob:
        logger.debug("Random remember trigger")
        return True

    # Length check
    if content:
        if len(content) < cfg.min_length:
            logger.debug(f"Too short: {len(content)} < {cfg.min_length}")
            return False
        if len(content) > cfg.max_length:
            logger.debug(f"Too long: {len(content)} > {cfg.max_length}")
            return False

    # Emotional boost
    emotional_boost = 0.0
    if content and cfg.emotional_keywords:
        content_lower = content.lower()
        for keyword in cfg.emotional_keywords:
            if keyword in content_lower:
                emotional_boost = cfg.emotional_weight
                break

    # Adjusted thresholds
    effective_resonance = resonance + emotional_boost
    effective_novelty = novelty + emotional_boost * 0.5

    # Check thresholds
    if effective_resonance < cfg.min_resonance:
        logger.debug(f"Low resonance: {effective_resonance:.2f} < {cfg.min_resonance}")
        return False

    if effective_novelty < cfg.min_novelty:
        logger.debug(f"Low novelty: {effective_novelty:.2f} < {cfg.min_novelty}")
        return False

    return True


def compute_experience_score(
    content: str,
    origin_fingerprint: np.ndarray,
    existing_fingerprints: List[np.ndarray],
    filter_config: Optional[ExperienceFilter] = None,
) -> Tuple[float, float, float]:
    """
    Compute experience scores for content.

    Returns:
        (resonance, novelty, emotional_weight)
    """
    cfg = filter_config or ExperienceFilter()

    # Compute fingerprint
    content_fp = compute_fingerprint(content)

    # Resonance with origin
    resonance = cosine_similarity(content_fp, origin_fingerprint)

    # Novelty (inverse of max similarity to existing)
    if existing_fingerprints:
        max_similarity = max(
            cosine_similarity(content_fp, fp)
            for fp in existing_fingerprints
        )
        novelty = 1.0 - max_similarity
    else:
        novelty = 1.0

    # Emotional weight
    emotional = 0.0
    if cfg.emotional_keywords:
        content_lower = content.lower()
        matches = sum(1 for kw in cfg.emotional_keywords if kw in content_lower)
        emotional = min(1.0, matches * 0.2)

    return resonance, novelty, emotional


class ExperienceJournal:
    """
    Tracks what was remembered and forgotten.

    Useful for:
    - Debugging memory decisions
    - Understanding organism behavior
    - Tuning filter parameters
    """

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self.entries: List[dict] = []

    def log(
        self,
        content: str,
        resonance: float,
        novelty: float,
        remembered: bool,
        reason: Optional[str] = None,
    ):
        """Log an experience decision."""
        entry = {
            "content_preview": content[:100] + "..." if len(content) > 100 else content,
            "resonance": resonance,
            "novelty": novelty,
            "remembered": remembered,
            "reason": reason,
            "timestamp": __import__("time").time(),
        }

        self.entries.append(entry)

        # Trim if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def stats(self) -> dict:
        """Get journal statistics."""
        if not self.entries:
            return {"total": 0}

        remembered = [e for e in self.entries if e["remembered"]]
        forgotten = [e for e in self.entries if not e["remembered"]]

        return {
            "total": len(self.entries),
            "remembered": len(remembered),
            "forgotten": len(forgotten),
            "remember_rate": len(remembered) / len(self.entries),
            "avg_resonance_remembered": (
                np.mean([e["resonance"] for e in remembered])
                if remembered else 0
            ),
            "avg_resonance_forgotten": (
                np.mean([e["resonance"] for e in forgotten])
                if forgotten else 0
            ),
            "avg_novelty_remembered": (
                np.mean([e["novelty"] for e in remembered])
                if remembered else 0
            ),
        }

    def recent(self, n: int = 10) -> List[dict]:
        """Get recent entries."""
        return self.entries[-n:]


def analyze_interaction(
    content: str,
    origin_text: str,
    existing_memories: List[str],
) -> dict:
    """
    Analyze an interaction for potential memorization.

    Useful for debugging and understanding decisions.
    """
    origin_fp = compute_fingerprint(origin_text)
    content_fp = compute_fingerprint(content)
    existing_fps = [compute_fingerprint(m) for m in existing_memories]

    resonance = cosine_similarity(content_fp, origin_fp)

    if existing_fps:
        similarities = [cosine_similarity(content_fp, fp) for fp in existing_fps]
        max_similarity = max(similarities)
        novelty = 1.0 - max_similarity
        most_similar_idx = np.argmax(similarities)
    else:
        novelty = 1.0
        max_similarity = 0.0
        most_similar_idx = None

    # Emotional keywords
    cfg = ExperienceFilter()
    content_lower = content.lower()
    found_emotional = [kw for kw in cfg.emotional_keywords if kw in content_lower]

    return {
        "content_length": len(content),
        "resonance": resonance,
        "novelty": novelty,
        "max_similarity_to_existing": max_similarity,
        "most_similar_memory_idx": most_similar_idx,
        "emotional_keywords_found": found_emotional,
        "would_remember": should_remember(resonance, novelty, cfg, content),
    }

"""
resonant_recall.py — SantaClaus for Stanley (Drunk Recall from Binary Shards)

Ported from Leo's santaclaus.py with Stanley-specific adaptations:
- Recalls from MemorySea shards (not SQLite)
- Uses Shard.content and resonance_score
- Integrates with pulse for arousal matching
- SILLY_FACTOR for playful "drunk" recall

Philosophy: Santa remembers Stanley's brightest moments.
Sometimes he brings them back as gifts when they resonate with now.
A child is allowed to believe in stories.

Unified memory architecture:
    experience → shard → training (LoRA deltas)
                      ↘ recall (resonance boost)

The same memory both teaches and reminds.
"""

from __future__ import annotations

import re
import time
import random
import numpy as np
from typing import Dict, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass
from collections import Counter
import logging

if TYPE_CHECKING:
    from .memory_sea import MemorySea
    from .shard import Shard
    from .subjectivity import Pulse

logger = logging.getLogger(__name__)


# ============================================================
#  CONFIGURATION
# ============================================================

# Recency decay parameters
RECENCY_WINDOW_HOURS = 24.0  # Full penalty if recalled within this time
RECENCY_PENALTY_STRENGTH = 0.5  # Max 50% quality reduction

# Drunk factor (from Haze's PlayfulSanta)
# Probability of picking a random shard instead of the best one
# This adds creative unpredictability — sometimes Santa gets playful
SILLY_FACTOR = 0.15  # 15% chance of "wrong" but creative recall

# Sticky phrases to avoid (contaminated patterns)
STICKY_PHRASES = [
    # Add phrases that should not be recalled
    # Example: "soft hand on my shoulder"
]


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class RecallContext:
    """What Santa gives back before generation."""
    recalled_texts: List[str]
    recalled_shard_ids: List[str]
    token_boosts: Dict[str, float]
    is_silly: bool  # Was this a drunk recall?
    total_score: float


# ============================================================
#  RESONANT RECALL (SANTACLAUS)
# ============================================================

class ResonantRecall:
    """
    Resonant recall layer for Stanley — SantaClaus adapted for binary shards.

    Remembers Stanley's best moments (shards) and brings them back
    when they resonate with the current context.

    Key differences from Leo's SantaClaus:
    - Uses MemorySea instead of SQLite
    - Shards already have resonance_score and content
    - Integrates with pulse for arousal matching
    - Updates shard recall metrics for learning

    SILLY_FACTOR creates playful unpredictability:
    Sometimes Santa picks a random memory instead of the "best" one.
    This adds character — not all recalls are perfect.
    """

    def __init__(
        self,
        memory_sea: "MemorySea",
        max_recalls: int = 3,
        max_tokens_per_recall: int = 64,
        boost_alpha: float = 0.2,
        silly_factor: float = SILLY_FACTOR,
    ):
        """
        Args:
            memory_sea: Stanley's MemorySea with shards
            max_recalls: Maximum memories to recall per prompt
            max_tokens_per_recall: Truncate recalled text
            boost_alpha: Overall strength of token boosting
            silly_factor: Probability of random "drunk" recall
        """
        self.memory = memory_sea
        self.max_recalls = max_recalls
        self.max_tokens_per_recall = max_tokens_per_recall
        self.boost_alpha = boost_alpha
        self.silly_factor = silly_factor

        # Stats
        self.total_recalls = 0
        self.silly_recalls = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer."""
        return re.findall(r"[A-Za-zА-Яа-яÀ-ÖØ-öø-ÿ']+|[.,!?;:\-]", text.lower())

    def _compute_recency_penalty(self, shard: "Shard") -> float:
        """Compute recency penalty for recently recalled shards."""
        if shard.last_recalled_at <= 0:
            return 0.0  # Never recalled = no penalty

        hours_since_recall = (time.time() - shard.last_recalled_at) / 3600.0

        if hours_since_recall < RECENCY_WINDOW_HOURS:
            # Penalty is strongest right after use, decays over window
            return 1.0 - (hours_since_recall / RECENCY_WINDOW_HOURS)
        return 0.0

    def _check_sticky_phrases(self, text: str) -> bool:
        """Check if text contains contaminated phrases."""
        text_lower = text.lower()
        for phrase in STICKY_PHRASES:
            if phrase in text_lower:
                return True
        return False

    def _score_shard(
        self,
        shard: "Shard",
        prompt_tokens: Set[str],
        pulse: Optional["Pulse"],
        active_tags: Optional[List[str]] = None,
    ) -> float:
        """
        Score a shard for recall relevance.

        Scoring factors (inspired by Leo):
        - Token overlap (Jaccard) - 40%
        - Tag/theme overlap - 20%
        - Arousal proximity - 20%
        - Quality (resonance_score) - 20%

        Penalties:
        - Recency penalty (don't repeat recent recalls)
        - Sticky phrase penalty (contaminated patterns)
        """
        if not shard.content:
            return 0.0

        shard_tokens = set(self._tokenize(shard.content))
        if not shard_tokens:
            return 0.0

        # 1. Token overlap (Jaccard similarity)
        overlap = len(prompt_tokens & shard_tokens)
        union = len(prompt_tokens | shard_tokens)
        token_score = overlap / union if union > 0 else 0.0

        # 2. Tag/theme overlap
        tag_score = 0.0
        if active_tags and shard.semantic_tags:
            matching_tags = len(set(active_tags) & set(shard.semantic_tags))
            tag_score = matching_tags / len(active_tags) if active_tags else 0.0

        # 3. Arousal proximity
        arousal_score = 0.5  # Default neutral
        if pulse:
            # Use resonance_score as proxy for shard's emotional charge
            shard_arousal = shard.resonance_score
            arousal_diff = abs(pulse.arousal - shard_arousal)
            arousal_score = max(0.0, 1.0 - arousal_diff)

        # 4. Quality (resonance_score with recency penalty)
        quality = shard.resonance_score
        recency_penalty = self._compute_recency_penalty(shard)
        quality_adjusted = quality * (1.0 - RECENCY_PENALTY_STRENGTH * recency_penalty)

        # Sticky phrase penalty (90% reduction)
        if self._check_sticky_phrases(shard.content):
            quality_adjusted *= 0.1

        # Combine scores
        score = (
            0.4 * token_score +
            0.2 * tag_score +
            0.2 * arousal_score +
            0.2 * quality_adjusted
        )

        return score

    def _collect_shards(self) -> List["Shard"]:
        """Collect all available shards from MemorySea."""
        shards = []

        # Surface shards (most recent/active)
        shards.extend(self.memory.surface)

        # Middle shards (settled but accessible)
        shards.extend(self.memory.middle)

        # Deep shards (rarely accessed but may resonate)
        shards.extend(self.memory.deep)

        return shards

    def recall(
        self,
        prompt: str,
        pulse: Optional["Pulse"] = None,
        active_tags: Optional[List[str]] = None,
    ) -> Optional[RecallContext]:
        """
        Main entry point: recall resonant memories.

        Args:
            prompt: Current prompt/context
            pulse: Current pulse state for arousal matching
            active_tags: Active themes/tags to boost

        Returns:
            RecallContext with recalled texts and token boosts,
            or None if nothing to recall
        """
        if not prompt or not prompt.strip():
            return None

        # Collect all shards
        shards = self._collect_shards()
        if not shards:
            return None

        # Tokenize prompt
        prompt_tokens = set(self._tokenize(prompt))
        if not prompt_tokens:
            return None

        # Score each shard
        scored: List[tuple[float, "Shard"]] = []

        for shard in shards:
            score = self._score_shard(shard, prompt_tokens, pulse, active_tags)
            if score > 0.1:  # Threshold
                scored.append((score, shard))

        if not scored:
            return None

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # DRUNK SELECTION (SILLY_FACTOR)
        # Sometimes Santa gets playful and picks a random shard
        top_recalls = []
        is_silly = False

        for i in range(min(self.max_recalls, len(scored))):
            if random.random() < self.silly_factor and len(scored) > 1:
                # Santa stumbles! Pick a random one 🎁
                random_idx = random.randint(0, len(scored) - 1)
                top_recalls.append(scored[random_idx])
                is_silly = True
                self.silly_recalls += 1
            else:
                # Sober moment: pick the best
                if i < len(scored):
                    top_recalls.append(scored[i])

        self.total_recalls += len(top_recalls)

        # Build recall context
        recalled_texts: List[str] = []
        recalled_ids: List[str] = []
        all_tokens: List[str] = []
        total_score = 0.0

        now = time.time()

        for score, shard in top_recalls:
            if not shard.content:
                continue

            # Update shard recall metrics
            shard.last_recalled_at = now
            shard.recall_count += 1

            # Truncate if needed
            tokens = self._tokenize(shard.content)
            if len(tokens) > self.max_tokens_per_recall:
                tokens = tokens[:self.max_tokens_per_recall]
                text = " ".join(tokens)
            else:
                text = shard.content

            recalled_texts.append(text)
            recalled_ids.append(shard.id)
            all_tokens.extend(tokens)
            total_score += score

        if not recalled_texts:
            return None

        # Build token boosts
        token_counts = Counter(all_tokens)
        max_count = max(token_counts.values()) if token_counts else 1

        token_boosts: Dict[str, float] = {}
        for token, count in token_counts.items():
            # Normalize to [0, boost_alpha]
            boost = self.boost_alpha * (count / max_count)
            token_boosts[token] = boost

        logger.debug(
            f"Resonant recall: {len(recalled_texts)} memories "
            f"(silly={is_silly}, total_score={total_score:.2f})"
        )

        return RecallContext(
            recalled_texts=recalled_texts,
            recalled_shard_ids=recalled_ids,
            token_boosts=token_boosts,
            is_silly=is_silly,
            total_score=total_score,
        )

    def bias_probabilities(
        self,
        probs: np.ndarray,
        context: RecallContext,
        vocab_encode_fn,
    ) -> np.ndarray:
        """
        Bias generation probabilities based on recall context.

        Args:
            probs: Current probability distribution
            context: RecallContext from recall()
            vocab_encode_fn: Function to encode tokens to IDs

        Returns:
            Biased probability distribution
        """
        if not context or not context.token_boosts:
            return probs

        # Create bias vector
        bias = np.zeros(len(probs), dtype=np.float32)

        for token, boost in context.token_boosts.items():
            try:
                token_ids = vocab_encode_fn(token)
                for tid in token_ids:
                    if tid < len(bias):
                        bias[tid] += boost
            except Exception:
                continue

        # Apply bias (additive in log space approximation)
        if bias.sum() > 0:
            bias = bias / bias.sum() * self.boost_alpha
            biased = probs + bias
            biased = biased / biased.sum()
            return biased

        return probs

    def get_stats(self) -> Dict:
        """Get recall statistics."""
        return {
            "total_recalls": self.total_recalls,
            "silly_recalls": self.silly_recalls,
            "silly_rate": self.silly_recalls / max(1, self.total_recalls),
            "shards_available": self.memory.total_shards(),
        }

    def __repr__(self) -> str:
        return (
            f"ResonantRecall(recalls={self.total_recalls}, "
            f"silly={self.silly_recalls}, "
            f"shards={self.memory.total_shards()})"
        )


# ============================================================
#  ASYNC VERSION
# ============================================================

import asyncio


class AsyncResonantRecall:
    """Async-safe wrapper for ResonantRecall."""

    def __init__(self, memory_sea: "MemorySea", **kwargs):
        self._sync = ResonantRecall(memory_sea, **kwargs)
        self._lock = asyncio.Lock()

    async def recall(
        self,
        prompt: str,
        pulse: Optional["Pulse"] = None,
        active_tags: Optional[List[str]] = None,
    ) -> Optional[RecallContext]:
        """Async recall with lock."""
        async with self._lock:
            return self._sync.recall(prompt, pulse, active_tags)

    async def get_stats(self) -> Dict:
        """Get stats atomically."""
        async with self._lock:
            return self._sync.get_stats()

    @property
    def total_recalls(self) -> int:
        return self._sync.total_recalls

    @property
    def silly_recalls(self) -> int:
        return self._sync.silly_recalls


if __name__ == "__main__":
    print("=== Stanley Resonant Recall (SantaClaus) ===")
    print()
    print("Resonant recall from MemorySea shards:")
    print(f"  SILLY_FACTOR = {SILLY_FACTOR} (drunk recall probability)")
    print(f"  RECENCY_WINDOW = {RECENCY_WINDOW_HOURS}h")
    print()
    print("Scoring: token_overlap(40%) + tags(20%) + arousal(20%) + quality(20%)")
    print()
    print("Unified memory: experience → shard → training AND recall")

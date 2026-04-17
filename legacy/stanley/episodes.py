#!/usr/bin/env python3
"""
episodes.py — Episodic Memory for Stanley

Adapted from Haze's episodes.py (inspired by Leo's episodic system).

Stanley remembers specific moments: prompt + response + metrics.
This is its episodic memory — structured recall of its own generations.

Core idea:
- Store each generation as an episode
- Query similar past episodes by metrics
- Learn from high-quality generations
- Self-RAG: retrieve from own history, not external corpus

NO TRAINING. NO NEURAL NETWORK. JUST RESONANCE.

Usage:
    from stanley.episodes import EpisodicMemory, Episode, StanleyMetrics

    memory = EpisodicMemory(max_episodes=500)
    memory.observe(Episode(seed="hello", output="...", metrics=...))
    similar = memory.query_similar(current_metrics)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import uuid
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StanleyMetrics:
    """
    Metrics captured for each episode.

    These are the "internal state" that describes what Stanley was "feeling"
    during this generation.
    """
    entropy: float = 0.0
    arousal: float = 0.0
    novelty: float = 0.0
    valence: float = 0.0

    # Body sense
    boredom: float = 0.0
    overwhelm: float = 0.0
    stuck: float = 0.0

    # Generation params
    temperature: float = 0.8
    method: str = "inference_engine"

    # Quality score (0-1, how good was this generation?)
    quality: float = 0.5
    resonance: float = 0.5

    def to_vector(self) -> List[float]:
        """Convert to feature vector for similarity search."""
        return [
            self.entropy,
            self.arousal,
            self.novelty,
            self.valence,
            self.boredom,
            self.overwhelm,
            self.stuck,
            self.temperature,
            self.quality,
            self.resonance,
        ]

    def to_dict(self) -> Dict[str, float]:
        """Convert to dict."""
        return {
            "entropy": self.entropy,
            "arousal": self.arousal,
            "novelty": self.novelty,
            "valence": self.valence,
            "boredom": self.boredom,
            "overwhelm": self.overwhelm,
            "stuck": self.stuck,
            "temperature": self.temperature,
            "quality": self.quality,
            "resonance": self.resonance,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "StanleyMetrics":
        """Create from dict."""
        return cls(
            entropy=d.get("entropy", 0.0),
            arousal=d.get("arousal", 0.0),
            novelty=d.get("novelty", 0.0),
            valence=d.get("valence", 0.0),
            boredom=d.get("boredom", 0.0),
            overwhelm=d.get("overwhelm", 0.0),
            stuck=d.get("stuck", 0.0),
            temperature=d.get("temperature", 0.8),
            quality=d.get("quality", 0.5),
            resonance=d.get("resonance", 0.5),
        )


@dataclass
class Episode:
    """
    One moment in Stanley's life.

    Captures the full context of a single generation:
    - What seed was used
    - What output was produced
    - What was Stanley's internal state
    """
    seed: str
    output: str
    metrics: StanleyMetrics
    timestamp: float = field(default_factory=time.time)
    episode_id: str = ""

    def __post_init__(self):
        if not self.episode_id:
            self.episode_id = str(uuid.uuid4())[:8]


# ============================================================================
# SIMILARITY
# ============================================================================

def cosine_distance(a: List[float], b: List[float]) -> float:
    """Compute cosine distance between two vectors (1 - cosine similarity)."""
    if len(a) != len(b):
        return 1.0

    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5

    if na == 0 or nb == 0:
        return 1.0

    similarity = dot / (na * nb)
    return 1.0 - similarity


def euclidean_distance(a: List[float], b: List[float]) -> float:
    """Compute Euclidean distance between two vectors."""
    if len(a) != len(b):
        return float('inf')

    sq_sum = sum((x - y) ** 2 for x, y in zip(a, b))
    return math.sqrt(sq_sum)


# ============================================================================
# EPISODIC MEMORY
# ============================================================================

class EpisodicMemory:
    """
    Local episodic memory for Stanley.

    Stores (seed, output, metrics, quality) as episodes.
    Provides simple similarity search over internal metrics.

    This is Self-RAG: retrieve from own history, not external corpus.
    """

    def __init__(self, max_episodes: int = 1000):
        self.episodes: List[Episode] = []
        self.max_episodes = max_episodes

        # Indices for fast lookup
        self._by_quality: List[Tuple[float, int]] = []
        self._by_resonance: List[Tuple[float, int]] = []

    def observe(self, episode: Episode) -> None:
        """
        Insert one episode into memory.

        Safe: clamps all values, ignores NaNs.
        """
        # Clamp and sanitize
        def clamp(x: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
            if x != x:  # NaN check
                return 0.0
            return max(min_val, min(max_val, x))

        episode.metrics.entropy = clamp(episode.metrics.entropy)
        episode.metrics.arousal = clamp(episode.metrics.arousal)
        episode.metrics.novelty = clamp(episode.metrics.novelty)
        episode.metrics.valence = clamp(episode.metrics.valence, -1.0, 1.0)
        episode.metrics.boredom = clamp(episode.metrics.boredom)
        episode.metrics.overwhelm = clamp(episode.metrics.overwhelm)
        episode.metrics.stuck = clamp(episode.metrics.stuck)
        episode.metrics.temperature = clamp(episode.metrics.temperature, 0.0, 2.0)
        episode.metrics.quality = clamp(episode.metrics.quality)
        episode.metrics.resonance = clamp(episode.metrics.resonance)

        # Add to list
        idx = len(self.episodes)
        self.episodes.append(episode)

        # Update indices
        self._by_quality.append((episode.metrics.quality, idx))
        self._by_resonance.append((episode.metrics.resonance, idx))

        # Prune if needed
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]
            self._rebuild_indices()

        logger.debug(f"Episode observed: {episode.episode_id}, quality={episode.metrics.quality:.2f}")

    def _rebuild_indices(self) -> None:
        """Rebuild lookup indices after pruning."""
        self._by_quality = [
            (ep.metrics.quality, i) for i, ep in enumerate(self.episodes)
        ]
        self._by_resonance = [
            (ep.metrics.resonance, i) for i, ep in enumerate(self.episodes)
        ]

    def query_similar(
        self,
        metrics: StanleyMetrics,
        top_k: int = 5,
        min_quality: float = 0.0,
    ) -> List[Episode]:
        """
        Find past episodes with similar internal configuration.

        Args:
            metrics: Current metrics to match
            top_k: Number of results to return
            min_quality: Minimum quality threshold

        Returns:
            List of similar episodes, sorted by similarity
        """
        if not self.episodes:
            return []

        query_vec = metrics.to_vector()

        # Compute distances
        distances: List[Tuple[float, Episode]] = []

        for episode in self.episodes:
            if episode.metrics.quality < min_quality:
                continue

            ep_vec = episode.metrics.to_vector()
            dist = cosine_distance(query_vec, ep_vec)
            distances.append((dist, episode))

        # Sort by distance (lower = more similar)
        distances.sort(key=lambda x: x[0])

        return [ep for _, ep in distances[:top_k]]

    def query_high_quality(self, top_k: int = 10) -> List[Episode]:
        """Get top K highest quality episodes."""
        sorted_eps = sorted(
            self._by_quality,
            key=lambda x: x[0],
            reverse=True,
        )
        return [self.episodes[idx] for _, idx in sorted_eps[:top_k]]

    def query_high_resonance(self, top_k: int = 10) -> List[Episode]:
        """Get top K highest resonance episodes."""
        sorted_eps = sorted(
            self._by_resonance,
            key=lambda x: x[0],
            reverse=True,
        )
        return [self.episodes[idx] for _, idx in sorted_eps[:top_k]]

    def query_by_seed_overlap(
        self,
        seed: str,
        top_k: int = 5,
    ) -> List[Episode]:
        """
        Find episodes with similar seeds (word overlap).

        Simple bag-of-words overlap for seed matching.
        """
        query_words = set(seed.lower().split())

        if not query_words:
            return []

        # Compute overlap for each episode
        overlaps: List[Tuple[float, Episode]] = []

        for episode in self.episodes:
            ep_words = set(episode.seed.lower().split())
            if not ep_words:
                continue

            overlap = len(query_words & ep_words)
            union = len(query_words | ep_words)
            jaccard = overlap / union if union > 0 else 0.0
            overlaps.append((jaccard, episode))

        # Sort by overlap (higher = more similar)
        overlaps.sort(key=lambda x: x[0], reverse=True)

        return [ep for _, ep in overlaps[:top_k]]

    def query_recent(self, n: int = 10) -> List[Episode]:
        """Get n most recent episodes."""
        return self.episodes[-n:]

    def get_quality_distribution(self) -> Dict[str, float]:
        """Get quality distribution stats."""
        if not self.episodes:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}

        qualities = [ep.metrics.quality for ep in self.episodes]
        mean = sum(qualities) / len(qualities)
        variance = sum((q - mean) ** 2 for q in qualities) / len(qualities)
        std = math.sqrt(variance)

        return {
            "min": min(qualities),
            "max": max(qualities),
            "mean": mean,
            "std": std,
        }

    def stats(self) -> Dict:
        """Get memory statistics."""
        dist = self.get_quality_distribution()
        return {
            "total_episodes": len(self.episodes),
            "max_episodes": self.max_episodes,
            "quality_mean": dist["mean"],
            "quality_std": dist["std"],
        }

    def save(self, path: str) -> None:
        """Save episodic memory to file."""
        import pickle
        data = {
            "max_episodes": self.max_episodes,
            "episodes": [
                {
                    "seed": ep.seed,
                    "output": ep.output,
                    "metrics": ep.metrics.to_dict(),
                    "timestamp": ep.timestamp,
                    "episode_id": ep.episode_id,
                }
                for ep in self.episodes
            ],
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"EpisodicMemory saved to {path}: {len(self.episodes)} episodes")

    @classmethod
    def load(cls, path: str) -> "EpisodicMemory":
        """Load episodic memory from file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        memory = cls(max_episodes=data.get("max_episodes", 1000))

        for ep_data in data.get("episodes", []):
            episode = Episode(
                seed=ep_data["seed"],
                output=ep_data["output"],
                metrics=StanleyMetrics.from_dict(ep_data["metrics"]),
                timestamp=ep_data["timestamp"],
                episode_id=ep_data["episode_id"],
            )
            memory.observe(episode)

        logger.info(f"EpisodicMemory loaded from {path}: {len(memory.episodes)} episodes")
        return memory

    def __repr__(self) -> str:
        dist = self.get_quality_distribution()
        return (f"EpisodicMemory(episodes={len(self.episodes)}, "
                f"quality_mean={dist['mean']:.2f})")

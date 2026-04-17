"""
semantic_drift.py — Stanley's Semantic Trajectory Learning

Ported from Leo's phase4_bridges.py with Stanley-specific adaptations:
- Episodes track shard activations, not "islands"
- Transitions between semantic tags
- Uses MemorySea resonance instead of pain/overwhelm metrics
- Risk filter based on quality and resonance trends

Philosophy: Learn which semantic paths naturally flow into each other.
Not "managing" Stanley's thoughts, but OBSERVING his patterns.

"осознанность через ассоциации, не через лозунги"
(awareness through associations, not through slogans)

STANLEY INNOVATIONS:
1. Shard-based trajectories (not island-based)
2. Resonance as primary quality signal
3. Tag transitions (semantic drift)
4. Integration with overthinking depth
5. Crystallization events as trajectory markers

"Which thoughts follow which thoughts,
 even when numbers don't perfectly match."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import math
import random
import uuid
import time
from collections import defaultdict

# Types
Metrics = Dict[str, float]  # e.g. {"entropy": 0.5, "arousal": 0.7}
SemanticTag = str           # e.g. "internal", "wounded", "resonant"
Timestamp = float


# ============================================================================
# EPISODE STRUCTURES
# ============================================================================


@dataclass
class DriftStep:
    """
    One step in Stanley's semantic trajectory.

    Captures:
    - Current metrics (from Pulse/BodyState)
    - Active semantic tags (from shards)
    - Resonance level
    - Overthinking depth
    """
    episode_id: str
    step_idx: int
    timestamp: Timestamp
    metrics: Metrics
    active_tags: List[SemanticTag]
    resonance: float = 0.5
    overthinking_depth: int = 0
    crystallized: bool = False  # Did this step produce a crystallization?


@dataclass
class DriftEpisode:
    """
    Full sequence of semantic drift steps (one conversation/session).
    """
    episode_id: str
    steps: List[DriftStep] = field(default_factory=list)
    start_time: Timestamp = 0.0
    end_time: Timestamp = 0.0

    def add_step(self, step: DriftStep) -> None:
        """Add a step to the episode."""
        assert step.episode_id == self.episode_id
        step.step_idx = len(self.steps)
        self.steps.append(step)
        self.end_time = step.timestamp

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time if self.start_time > 0 else 0.0

    @property
    def avg_resonance(self) -> float:
        """Average resonance across steps."""
        if not self.steps:
            return 0.5
        return sum(s.resonance for s in self.steps) / len(self.steps)


# ============================================================================
# TRANSITION STATISTICS
# ============================================================================


@dataclass
class TagTransition:
    """
    Statistics for transitions between semantic tags.

    Tracks:
    - How often tag A -> tag B happened
    - Average metric changes during this transition
    - Average resonance change
    """
    from_tag: SemanticTag
    to_tag: SemanticTag
    count: int = 0
    avg_deltas: Dict[str, float] = field(default_factory=dict)
    avg_resonance_delta: float = 0.0

    # Internal accumulators
    _delta_sums: Dict[str, float] = field(default_factory=dict, repr=False)
    _resonance_sum: float = field(default=0.0, repr=False)

    def update(
        self,
        from_metrics: Metrics,
        to_metrics: Metrics,
        from_resonance: float,
        to_resonance: float,
    ) -> None:
        """Update with new observed transition."""
        self.count += 1

        # Metric deltas
        for k in set(from_metrics.keys()) | set(to_metrics.keys()):
            before = from_metrics.get(k, 0.0)
            after = to_metrics.get(k, 0.0)
            delta = after - before
            self._delta_sums[k] = self._delta_sums.get(k, 0.0) + delta

        # Resonance delta
        self._resonance_sum += (to_resonance - from_resonance)

        # Recompute averages
        self.avg_deltas = {
            k: self._delta_sums[k] / self.count
            for k in self._delta_sums
        }
        self.avg_resonance_delta = self._resonance_sum / self.count


@dataclass
class TransitionGraph:
    """
    Graph of tag-to-tag transitions with statistics.

    Stanley's semantic drift patterns.
    """
    transitions: Dict[Tuple[SemanticTag, SemanticTag], TagTransition] = field(
        default_factory=dict
    )

    def update_from_episode(self, episode: DriftEpisode) -> None:
        """
        Parse an episode and update transition stats.
        """
        steps = episode.steps
        if len(steps) < 2:
            return

        for prev, curr in zip(steps[:-1], steps[1:]):
            # Pairwise connections between active tags
            for from_tag in prev.active_tags:
                for to_tag in curr.active_tags:
                    key = (from_tag, to_tag)
                    if key not in self.transitions:
                        self.transitions[key] = TagTransition(
                            from_tag=from_tag,
                            to_tag=to_tag,
                        )
                    self.transitions[key].update(
                        prev.metrics,
                        curr.metrics,
                        prev.resonance,
                        curr.resonance,
                    )

    def get_transition(
        self,
        from_tag: SemanticTag,
        to_tag: SemanticTag,
    ) -> Optional[TagTransition]:
        """Get transition stats for a specific pair."""
        return self.transitions.get((from_tag, to_tag))

    def outgoing(self, from_tag: SemanticTag) -> List[TagTransition]:
        """All outgoing transitions from a tag."""
        return [
            trans for (a, b), trans in self.transitions.items()
            if a == from_tag
        ]

    def incoming(self, to_tag: SemanticTag) -> List[TagTransition]:
        """All incoming transitions to a tag."""
        return [
            trans for (a, b), trans in self.transitions.items()
            if b == to_tag
        ]

    def most_common_paths(self, top_k: int = 10) -> List[TagTransition]:
        """Get most common transition paths."""
        sorted_trans = sorted(
            self.transitions.values(),
            key=lambda t: t.count,
            reverse=True,
        )
        return sorted_trans[:top_k]


# ============================================================================
# EPISODE LOGGER
# ============================================================================


class DriftLogger:
    """
    Collects steps of current semantic drift episode.

    Call log_step() after each Stanley response.
    Call end_episode() at conversation end.
    """

    def __init__(self):
        self.current_episode: Optional[DriftEpisode] = None
        self.completed_episodes: List[DriftEpisode] = []

    def start_episode(self) -> str:
        """Start a new episode. Returns episode_id."""
        episode_id = str(uuid.uuid4())
        self.current_episode = DriftEpisode(
            episode_id=episode_id,
            start_time=time.time(),
        )
        return episode_id

    def log_step(
        self,
        metrics: Metrics,
        active_tags: List[SemanticTag],
        resonance: float = 0.5,
        overthinking_depth: int = 0,
        crystallized: bool = False,
    ) -> None:
        """
        Log one step in the semantic drift.

        Call this after each Stanley response.
        """
        if self.current_episode is None:
            self.start_episode()

        assert self.current_episode is not None

        step = DriftStep(
            episode_id=self.current_episode.episode_id,
            step_idx=len(self.current_episode.steps),
            timestamp=time.time(),
            metrics=dict(metrics),
            active_tags=list(active_tags),
            resonance=resonance,
            overthinking_depth=overthinking_depth,
            crystallized=crystallized,
        )
        self.current_episode.add_step(step)

    def end_episode(self) -> Optional[DriftEpisode]:
        """Close current episode and return it."""
        ep = self.current_episode
        if ep is not None:
            self.completed_episodes.append(ep)
        self.current_episode = None
        return ep

    @property
    def total_steps(self) -> int:
        """Total steps across all episodes."""
        total = sum(len(ep.steps) for ep in self.completed_episodes)
        if self.current_episode:
            total += len(self.current_episode.steps)
        return total


# ============================================================================
# SIMILARITY AND BRIDGE SEARCH
# ============================================================================


def metrics_similarity(a: Metrics, b: Metrics, eps: float = 1e-8) -> float:
    """
    Compute similarity between two metric states in [0, 1].

    Uses 1 - normalized Euclidean distance.
    """
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 0.0

    sq_sum = 0.0
    for k in keys:
        da = a.get(k, 0.0)
        db = b.get(k, 0.0)
        d = da - db
        sq_sum += d * d

    dist = math.sqrt(sq_sum)

    # Normalize: assume each metric mostly in [0, 1]
    max_dist = math.sqrt(len(keys))  # Max if all metrics differ by 1
    if max_dist < eps:
        return 1.0

    sim = max(0.0, 1.0 - dist / max_dist)
    return sim


@dataclass
class DriftCandidate:
    """
    Historical example of "from this state, went to these tags".
    """
    from_tags: List[SemanticTag]
    to_tags: List[SemanticTag]
    from_metrics: Metrics
    to_metrics: Metrics
    from_resonance: float
    to_resonance: float
    similarity: float


class DriftMemory:
    """
    Stores episodes for semantic drift search.

    Find similar past states and see where they drifted.
    """

    def __init__(self, max_episodes: int = 100):
        self.episodes: List[DriftEpisode] = []
        self.max_episodes = max_episodes

    def add_episode(self, episode: DriftEpisode) -> None:
        """Add an episode to memory."""
        self.episodes.append(episode)
        # Keep bounded
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def find_similar_states(
        self,
        metrics_now: Metrics,
        active_tags_now: List[SemanticTag],
        min_similarity: float = 0.6,
    ) -> List[DriftCandidate]:
        """
        Find historical steps similar to current state.

        Returns candidates showing where similar states drifted.
        """
        candidates: List[DriftCandidate] = []

        for ep in self.episodes:
            steps = ep.steps
            if len(steps) < 2:
                continue

            for prev, nxt in zip(steps[:-1], steps[1:]):
                sim = metrics_similarity(metrics_now, prev.metrics)
                if sim < min_similarity:
                    continue

                candidate = DriftCandidate(
                    from_tags=list(prev.active_tags),
                    to_tags=list(nxt.active_tags),
                    from_metrics=dict(prev.metrics),
                    to_metrics=dict(nxt.metrics),
                    from_resonance=prev.resonance,
                    to_resonance=nxt.resonance,
                    similarity=sim,
                )
                candidates.append(candidate)

        return candidates

    @property
    def total_steps(self) -> int:
        """Total steps in memory."""
        return sum(len(ep.steps) for ep in self.episodes)


# ============================================================================
# TAG SCORING AND SUGGESTION
# ============================================================================


@dataclass
class TagScore:
    """Score for a suggested tag."""
    tag: SemanticTag
    raw_score: float
    avg_resonance_delta: float
    samples: int


def aggregate_tag_scores(
    candidates: List[DriftCandidate],
) -> Dict[SemanticTag, TagScore]:
    """
    Aggregate candidates into per-tag scores.

    raw_score ~ sum(similarity) for all times that tag appeared as destination.
    """
    score_sums: Dict[SemanticTag, float] = defaultdict(float)
    resonance_sums: Dict[SemanticTag, float] = defaultdict(float)
    counts: Dict[SemanticTag, int] = defaultdict(int)

    for c in candidates:
        for tag in c.to_tags:
            score_sums[tag] += c.similarity
            resonance_sums[tag] += (c.to_resonance - c.from_resonance)
            counts[tag] += 1

    result: Dict[SemanticTag, TagScore] = {}
    for tag, score in score_sums.items():
        n = counts[tag]
        avg_res = resonance_sums[tag] / n if n > 0 else 0.0
        result[tag] = TagScore(
            tag=tag,
            raw_score=score,
            avg_resonance_delta=avg_res,
            samples=n,
        )

    return result


def apply_quality_filter(
    scores: Dict[SemanticTag, TagScore],
    min_resonance_delta: float = -0.2,
) -> Dict[SemanticTag, TagScore]:
    """
    Filter out tags that historically decreased resonance too much.

    Stanley's risk avoidance: don't drift into "bad" semantic spaces.
    """
    filtered: Dict[SemanticTag, TagScore] = {}

    for tag, s in scores.items():
        # Skip tags that historically decreased quality
        if s.avg_resonance_delta < min_resonance_delta:
            continue
        filtered[tag] = s

    return filtered


def normalize_scores(
    scores: Dict[SemanticTag, TagScore],
    temperature: float = 1.0,
) -> Dict[SemanticTag, float]:
    """
    Convert raw scores to probability distribution.
    """
    if not scores:
        return {}

    # Softmax with temperature
    max_score = max(s.raw_score for s in scores.values())
    exp_values: Dict[SemanticTag, float] = {}

    for tag, s in scores.items():
        x = (s.raw_score - max_score) / max(temperature, 1e-6)
        exp_values[tag] = math.exp(x)

    total = sum(exp_values.values())
    if total <= 0.0:
        n = len(scores)
        return {tag: 1.0 / n for tag in scores}

    return {tag: v / total for tag, v in exp_values.items()}


def sample_tags(
    probs: Dict[SemanticTag, float],
    top_k: int = 3,
    exploration_rate: float = 0.2,
) -> List[SemanticTag]:
    """
    Sample suggested tags with exploration.

    Mostly picks high-probability tags, sometimes explores.
    """
    if not probs:
        return []

    sorted_tags = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    tags_only = [name for name, _ in sorted_tags]

    # Base: top_k candidates
    base = tags_only[:top_k]

    # Exploration: sometimes swap in a random tag
    result = list(base)
    if len(tags_only) > top_k:
        remaining = tags_only[top_k:]
        if random.random() < exploration_rate and remaining:
            candidate = random.choice(remaining)
            if result:
                result[-1] = candidate
            else:
                result.append(candidate)

    return result


# ============================================================================
# HIGH-LEVEL DRIFT SUGGESTION
# ============================================================================


def suggest_semantic_drift(
    metrics_now: Metrics,
    active_tags_now: List[SemanticTag],
    drift_memory: DriftMemory,
    min_similarity: float = 0.6,
    temperature: float = 0.7,
    exploration_rate: float = 0.2,
) -> List[SemanticTag]:
    """
    Suggest next semantic tags based on historical drift patterns.

    Args:
        metrics_now: Current state metrics
        active_tags_now: Currently active semantic tags
        drift_memory: Memory of past episodes
        min_similarity: Minimum similarity to consider
        temperature: Softmax temperature
        exploration_rate: Probability of exploration

    Returns:
        List of suggested semantic tags to drift toward
    """
    # 1) Find similar historical states
    candidates = drift_memory.find_similar_states(
        metrics_now=metrics_now,
        active_tags_now=active_tags_now,
        min_similarity=min_similarity,
    )

    if not candidates:
        return []

    # 2) Aggregate to per-tag scores
    scores = aggregate_tag_scores(candidates)

    # 3) Filter risky tags (low resonance delta)
    safe_scores = apply_quality_filter(scores)

    if not safe_scores:
        return []

    # 4) Normalize to probabilities
    probs = normalize_scores(safe_scores, temperature=temperature)

    # 5) Sample with exploration
    suggested = sample_tags(probs, top_k=3, exploration_rate=exploration_rate)

    return suggested


# ============================================================================
# SEMANTIC DRIFT (Main class)
# ============================================================================


class SemanticDrift:
    """
    Stanley's semantic trajectory learning.

    Observes patterns of semantic drift across conversations.
    Suggests where to drift based on historical patterns.

    "Which thoughts follow which thoughts."
    """

    def __init__(self, max_episodes: int = 100):
        self.logger = DriftLogger()
        self.memory = DriftMemory(max_episodes=max_episodes)
        self.graph = TransitionGraph()

    def start_session(self) -> str:
        """Start a new conversation/session."""
        return self.logger.start_episode()

    def log_step(
        self,
        metrics: Metrics,
        active_tags: List[SemanticTag],
        resonance: float = 0.5,
        overthinking_depth: int = 0,
        crystallized: bool = False,
    ) -> None:
        """Log a step in the current session."""
        self.logger.log_step(
            metrics=metrics,
            active_tags=active_tags,
            resonance=resonance,
            overthinking_depth=overthinking_depth,
            crystallized=crystallized,
        )

    def end_session(self) -> Optional[DriftEpisode]:
        """
        End current session and learn from it.

        Updates transition graph and memory.
        """
        episode = self.logger.end_episode()
        if episode and len(episode.steps) >= 2:
            self.graph.update_from_episode(episode)
            self.memory.add_episode(episode)
        return episode

    def suggest_drift(
        self,
        metrics_now: Metrics,
        active_tags_now: List[SemanticTag],
        temperature: float = 0.7,
    ) -> List[SemanticTag]:
        """
        Suggest semantic tags to drift toward.

        Based on historical patterns of similar states.
        """
        return suggest_semantic_drift(
            metrics_now=metrics_now,
            active_tags_now=active_tags_now,
            drift_memory=self.memory,
            min_similarity=0.6,
            temperature=temperature,
            exploration_rate=0.2,
        )

    def get_stats(self) -> Dict:
        """Get drift statistics."""
        return {
            "total_episodes": len(self.memory.episodes),
            "total_steps": self.memory.total_steps,
            "transition_count": len(self.graph.transitions),
            "common_paths": [
                (t.from_tag, t.to_tag, t.count)
                for t in self.graph.most_common_paths(5)
            ],
        }

    def __repr__(self) -> str:
        return (
            f"SemanticDrift(episodes={len(self.memory.episodes)}, "
            f"steps={self.memory.total_steps}, "
            f"transitions={len(self.graph.transitions)})"
        )


# ============================================================================
# ASYNC DISCIPLINE
# ============================================================================

import asyncio


class AsyncSemanticDrift:
    """Async-safe wrapper for SemanticDrift."""

    def __init__(self, max_episodes: int = 100):
        self._sync = SemanticDrift(max_episodes=max_episodes)
        self._lock = asyncio.Lock()

    async def start_session(self) -> str:
        async with self._lock:
            return self._sync.start_session()

    async def log_step(
        self,
        metrics: Metrics,
        active_tags: List[SemanticTag],
        resonance: float = 0.5,
        overthinking_depth: int = 0,
        crystallized: bool = False,
    ) -> None:
        async with self._lock:
            self._sync.log_step(
                metrics=metrics,
                active_tags=active_tags,
                resonance=resonance,
                overthinking_depth=overthinking_depth,
                crystallized=crystallized,
            )

    async def end_session(self) -> Optional[DriftEpisode]:
        async with self._lock:
            return self._sync.end_session()

    async def suggest_drift(
        self,
        metrics_now: Metrics,
        active_tags_now: List[SemanticTag],
        temperature: float = 0.7,
    ) -> List[SemanticTag]:
        async with self._lock:
            return self._sync.suggest_drift(
                metrics_now=metrics_now,
                active_tags_now=active_tags_now,
                temperature=temperature,
            )

    async def get_stats(self) -> Dict:
        async with self._lock:
            return self._sync.get_stats()

    def __repr__(self) -> str:
        return f"Async{repr(self._sync)}"


if __name__ == "__main__":
    print("=== Stanley Semantic Drift Demo ===")
    print()
    print("Learn which semantic paths naturally flow into each other.")
    print()
    print("Core concepts:")
    print("  - DriftEpisode: sequence of (metrics, tags) steps")
    print("  - TransitionGraph: tag A -> tag B statistics")
    print("  - DriftMemory: find similar past states")
    print("  - Quality filter: avoid low-resonance paths")
    print()
    print("'Which thoughts follow which thoughts,")
    print(" even when numbers don't perfectly match.'")

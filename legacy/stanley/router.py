"""
router.py — Selective memory loading

The Router decides which shards to load for current context.
Not "retrieve all," but "what resonates now."

Like human attention: we don't think about everything at once,
only what's relevant to the current moment.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

from .shard import Shard
from .fingerprint import compute_fingerprint, FingerprintConfig

logger = logging.getLogger(__name__)


@dataclass
class RouterConfig:
    """Configuration for shard routing."""
    
    # Working set limits
    max_working_set: int = 32
    min_working_set: int = 4
    
    # Scoring weights
    resonance_weight: float = 0.5   # how much context matches
    recency_weight: float = 0.3     # how recently activated
    activation_weight: float = 0.1  # how often activated
    depth_weight: float = 0.1       # surface bonus
    
    # Thresholds
    min_resonance: float = 0.1      # minimum to consider
    recency_halflife: float = 3600  # 1 hour halflife for recency
    
    # Fingerprint config
    fingerprint_config: Optional[FingerprintConfig] = None


class Router:
    """
    Routes context to relevant shards.
    
    Computes a score for each shard based on:
    - Resonance with current context
    - Recency of last activation
    - Activation count (popularity)
    - Depth (surface shards get bonus)
    
    Returns top-K as working set.
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self.fp_config = self.config.fingerprint_config or FingerprintConfig()
    
    def compute_context_fingerprint(self, context: str) -> np.ndarray:
        """Compute fingerprint for current context."""
        return compute_fingerprint(context, self.fp_config)
    
    def score_shard(
        self,
        shard: Shard,
        context_fp: np.ndarray,
        current_time: Optional[float] = None
    ) -> float:
        """
        Compute routing score for a single shard.
        
        Higher score = more relevant to current context.
        """
        if current_time is None:
            current_time = time.time()
        
        cfg = self.config
        
        # Resonance score (similarity to context)
        resonance = shard.similarity_to(context_fp)
        if resonance < cfg.min_resonance:
            return 0.0  # Below threshold, don't consider
        
        # Recency score (exponential decay)
        age = current_time - shard.last_activated
        recency = np.exp(-age / cfg.recency_halflife)
        
        # Activation score (log scale, capped)
        activation = np.log1p(shard.activation_count) / 10.0
        activation = min(activation, 1.0)
        
        # Depth bonus
        depth_bonus = {
            "surface": 1.0,
            "middle": 0.5,
            "deep": 0.2,
            "abyss": 0.0,  # metanotes handled separately
        }.get(shard.depth, 0.0)
        
        # Weighted combination
        score = (
            cfg.resonance_weight * resonance +
            cfg.recency_weight * recency +
            cfg.activation_weight * activation +
            cfg.depth_weight * depth_bonus
        )
        
        return score
    
    def select_working_set(
        self,
        context: str,
        shards: List[Shard],
        max_size: Optional[int] = None
    ) -> List[Tuple[Shard, float]]:
        """
        Select working set for given context.
        
        Returns list of (shard, score) tuples, sorted by score descending.
        """
        if not shards:
            return []
        
        max_size = max_size or self.config.max_working_set
        context_fp = self.compute_context_fingerprint(context)
        current_time = time.time()
        
        # Score all shards
        scored = []
        for shard in shards:
            score = self.score_shard(shard, context_fp, current_time)
            if score > 0:
                scored.append((shard, score))
        
        # Sort by score
        scored.sort(key=lambda x: -x[1])
        
        # Take top-K
        working_set = scored[:max_size]
        
        logger.debug(
            f"Router: {len(working_set)}/{len(shards)} shards selected, "
            f"top score: {working_set[0][1]:.3f}" if working_set else "empty"
        )
        
        return working_set
    
    def should_promote(
        self,
        shard: Shard,
        context_fp: np.ndarray,
        threshold: float = 0.6
    ) -> bool:
        """Check if shard should be promoted to surface."""
        score = self.score_shard(shard, context_fp)
        return score > threshold and shard.depth != "surface"
    
    def compute_novelty_need(
        self,
        context: str,
        working_set: List[Shard]
    ) -> float:
        """
        Compute how much the context differs from working set.
        
        High novelty = context is very different, might need diverse shards.
        """
        if not working_set:
            return 1.0  # Maximum novelty if no shards
        
        context_fp = self.compute_context_fingerprint(context)
        
        # Average similarity to working set
        similarities = [s.similarity_to(context_fp) for s in working_set]
        avg_similarity = np.mean(similarities)
        
        # Novelty is inverse
        return 1.0 - avg_similarity


class AdaptiveRouter(Router):
    """
    Router that adapts based on context and history.
    
    Tracks which shards were useful and adjusts weights.
    """
    
    def __init__(self, config: Optional[RouterConfig] = None):
        super().__init__(config)
        
        # Track shard usefulness
        self.useful_history: List[str] = []  # shard IDs that were useful
        self.history_limit: int = 100
    
    def mark_useful(self, shard_id: str):
        """Mark a shard as useful in current context."""
        self.useful_history.append(shard_id)
        if len(self.useful_history) > self.history_limit:
            self.useful_history.pop(0)
    
    def usefulness_bonus(self, shard_id: str) -> float:
        """Bonus for shards that were recently useful."""
        count = self.useful_history.count(shard_id)
        return min(count * 0.1, 0.5)  # Max 0.5 bonus
    
    def score_shard(
        self,
        shard: Shard,
        context_fp: np.ndarray,
        current_time: Optional[float] = None
    ) -> float:
        """Score with usefulness bonus."""
        base_score = super().score_shard(shard, context_fp, current_time)
        bonus = self.usefulness_bonus(shard.id)
        return base_score + bonus

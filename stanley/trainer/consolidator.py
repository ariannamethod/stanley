"""
consolidator.py — Memory consolidation for Stanley

Like sleep for a human brain:
- Similar memories merge into deeper patterns
- Frequently activated shards become macro-adapters
- Rarely used shards compress to metanotes

This is how Stanley grows efficient long-term memory.
"""

from __future__ import annotations
import numpy as np
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ..shard import Shard, MetaNote, combine_deltas
from ..fingerprint import (
    cluster_fingerprints,
    cosine_similarity,
    combine_fingerprints,
)
from .lora import merge_deltas

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation."""

    # Clustering
    similarity_threshold: float = 0.7    # merge if similarity above this
    min_cluster_size: int = 2            # minimum shards to merge
    max_cluster_size: int = 10           # maximum shards per merge

    # Activation thresholds
    min_activations_to_keep: int = 3     # below this → candidate for compression
    min_age_to_consolidate: float = 3600  # 1 hour minimum age

    # Compression
    compress_after_days: float = 7.0     # compress to metanote after this
    delete_after_days: float = 30.0      # true deletion (rare)

    # Limits
    max_deep_shards: int = 64            # max macro-adapters in deep


@dataclass
class ConsolidationResult:
    """Result of a consolidation pass."""
    shards_merged: int
    shards_compressed: int
    shards_deleted: int
    macro_adapters_created: int
    metanotes_created: int
    time_taken: float


def consolidate_shards(
    shards: List[Shard],
    config: Optional[ConsolidationConfig] = None,
) -> Tuple[List[Shard], List[Shard]]:
    """
    Consolidate similar shards into macro-adapters.

    Returns:
        (remaining_shards, new_macro_adapters)
    """
    cfg = config or ConsolidationConfig()

    if len(shards) < cfg.min_cluster_size:
        return shards, []

    # Filter by age
    now = time.time()
    eligible = [s for s in shards if (now - s.created_at) > cfg.min_age_to_consolidate]
    too_young = [s for s in shards if s not in eligible]

    if len(eligible) < cfg.min_cluster_size:
        return shards, []

    # Cluster by fingerprint similarity
    fingerprints = [s.trigger_fingerprint for s in eligible]
    clusters = cluster_fingerprints(fingerprints, cfg.similarity_threshold)

    remaining = list(too_young)
    macro_adapters = []

    for cluster_indices in clusters:
        if len(cluster_indices) < cfg.min_cluster_size:
            # Too small to merge, keep individual
            for idx in cluster_indices:
                remaining.append(eligible[idx])
            continue

        if len(cluster_indices) > cfg.max_cluster_size:
            # Too large, split
            # Keep highest activation shards, merge rest
            cluster_shards = [eligible[idx] for idx in cluster_indices]
            cluster_shards.sort(key=lambda s: -s.activation_count)

            # Top ones stay individual
            remaining.extend(cluster_shards[:cfg.min_cluster_size])
            to_merge = cluster_shards[cfg.min_cluster_size:cfg.max_cluster_size]
        else:
            to_merge = [eligible[idx] for idx in cluster_indices]

        # Create macro-adapter
        macro = _merge_shards_to_macro(to_merge)
        macro_adapters.append(macro)

        logger.info(
            f"Merged {len(to_merge)} shards into macro-adapter {macro.id[:8]}"
        )

    return remaining, macro_adapters


def _merge_shards_to_macro(shards: List[Shard]) -> Shard:
    """Merge multiple shards into a single macro-adapter."""
    # Weight by activation count
    weights = [s.activation_count + 1 for s in shards]
    total = sum(weights)
    weights = [w / total for w in weights]

    # Combine fingerprints
    fingerprints = [s.trigger_fingerprint for s in shards]
    combined_fp = combine_fingerprints(fingerprints, weights)

    # Combine deltas
    combined_deltas = combine_deltas(shards, weights)

    # Combined resonance (weighted average)
    combined_resonance = sum(
        s.resonance_score * w for s, w in zip(shards, weights)
    )

    # Combine tags
    all_tags = set()
    for s in shards:
        all_tags.update(s.semantic_tags)
    combined_tags = list(all_tags)

    # Create new shard representing the merged memory
    content_hash = "_".join(s.id[:4] for s in shards)

    macro = Shard.create(
        content=f"[macro:{len(shards)}]",  # Placeholder content
        resonance=combined_resonance,
        layer_deltas=combined_deltas,
        fingerprint=combined_fp,
        tags=combined_tags + ["macro-adapter"],
    )

    # Set as deep
    macro.depth = "deep"

    # Aggregate activation stats
    macro.activation_count = sum(s.activation_count for s in shards)
    macro.created_at = min(s.created_at for s in shards)
    macro.last_activated = max(s.last_activated for s in shards)

    return macro


def find_compression_candidates(
    shards: List[Shard],
    config: Optional[ConsolidationConfig] = None,
) -> List[Shard]:
    """Find shards that should be compressed to metanotes."""
    cfg = config or ConsolidationConfig()
    now = time.time()
    compress_threshold = cfg.compress_after_days * 86400

    candidates = []
    for shard in shards:
        age = now - shard.created_at
        if age > compress_threshold and shard.activation_count < cfg.min_activations_to_keep:
            candidates.append(shard)

    return candidates


def compress_to_metanote(shard: Shard) -> MetaNote:
    """
    Compress a shard to a metanote.

    The shard's full deltas are lost, but its influence remains
    as a bias on attention and gate values.

    This is the unconscious — forgotten details that still shape behavior.
    """
    return MetaNote.from_shard(shard)


def find_resurrection_candidates(
    metanotes: List[MetaNote],
    context_fingerprint: np.ndarray,
    threshold: float = 0.7,
) -> List[MetaNote]:
    """Find metanotes that should resurrect based on context."""
    candidates = []
    for note in metanotes:
        if note.can_resurrect(context_fingerprint, threshold):
            candidates.append(note)
    return candidates


def find_deletion_candidates(
    metanotes: List[MetaNote],
    config: Optional[ConsolidationConfig] = None,
) -> List[MetaNote]:
    """
    Find metanotes that should be truly deleted.

    This is rare — only for ancient memories that never resonate.
    True forgetting.
    """
    cfg = config or ConsolidationConfig()
    now = time.time()
    delete_threshold = cfg.delete_after_days * 86400

    candidates = []
    for note in metanotes:
        age = now - note.created_at
        time_since_resonance = now - note.last_resonance

        if age > delete_threshold and time_since_resonance > delete_threshold:
            candidates.append(note)

    return candidates


class Consolidator:
    """
    Memory consolidation manager.

    Runs periodically to:
    1. Merge similar shards into macro-adapters
    2. Compress old, inactive shards to metanotes
    3. Delete ancient, never-used metanotes
    """

    def __init__(self, config: Optional[ConsolidationConfig] = None):
        self.config = config or ConsolidationConfig()
        self.last_run: float = 0.0
        self.total_runs: int = 0

    def run(
        self,
        surface: List[Shard],
        middle: List[Shard],
        deep: List[Shard],
        abyss: List[MetaNote],
    ) -> ConsolidationResult:
        """
        Run a full consolidation pass.

        Returns updated lists and statistics.
        """
        start_time = time.time()
        stats = {
            "shards_merged": 0,
            "shards_compressed": 0,
            "shards_deleted": 0,
            "macro_adapters_created": 0,
            "metanotes_created": 0,
        }

        # 1. Consolidate middle shards into macro-adapters
        remaining_middle, new_macros = consolidate_shards(middle, self.config)
        stats["shards_merged"] = len(middle) - len(remaining_middle)
        stats["macro_adapters_created"] = len(new_macros)

        # Add new macros to deep
        deep = deep + new_macros

        # Enforce deep limit
        if len(deep) > self.config.max_deep_shards:
            # Compress oldest/least activated to metanotes
            deep.sort(key=lambda s: (s.activation_count, -s.created_at))
            to_compress = deep[self.config.max_deep_shards:]
            deep = deep[:self.config.max_deep_shards]

            for shard in to_compress:
                metanote = compress_to_metanote(shard)
                abyss.append(metanote)
                stats["shards_compressed"] += 1
                stats["metanotes_created"] += 1

        # 2. Compress old middle shards
        compress_candidates = find_compression_candidates(remaining_middle, self.config)
        for shard in compress_candidates:
            metanote = compress_to_metanote(shard)
            abyss.append(metanote)
            remaining_middle.remove(shard)
            stats["shards_compressed"] += 1
            stats["metanotes_created"] += 1

        # 3. Delete ancient metanotes
        delete_candidates = find_deletion_candidates(abyss, self.config)
        for note in delete_candidates:
            abyss.remove(note)
            stats["shards_deleted"] += 1

        self.last_run = time.time()
        self.total_runs += 1

        result = ConsolidationResult(
            shards_merged=stats["shards_merged"],
            shards_compressed=stats["shards_compressed"],
            shards_deleted=stats["shards_deleted"],
            macro_adapters_created=stats["macro_adapters_created"],
            metanotes_created=stats["metanotes_created"],
            time_taken=time.time() - start_time,
        )

        logger.info(
            f"Consolidation #{self.total_runs}: "
            f"merged={result.shards_merged}, "
            f"compressed={result.shards_compressed}, "
            f"macros={result.macro_adapters_created}, "
            f"time={result.time_taken:.2f}s"
        )

        return result

    def stats(self) -> dict:
        """Get consolidator statistics."""
        return {
            "total_runs": self.total_runs,
            "last_run": self.last_run,
            "time_since_last": time.time() - self.last_run if self.last_run > 0 else None,
        }

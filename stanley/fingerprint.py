"""
fingerprint.py — Fast context fingerprinting

Fingerprints are small vectors that capture the "shape" of text.
Used for:
- Fast resonance matching (O(1) per shard)
- Novelty detection
- Resurrection triggers

Based on n-gram hashing, proven in Leo's trigram system.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class FingerprintConfig:
    """Configuration for fingerprint generation."""
    size: int = 64           # fingerprint vector size
    ngram_sizes: tuple = (2, 3, 4)  # which n-grams to use
    weights: tuple = (0.2, 0.5, 0.3)  # weights for each n-gram size
    normalize: bool = True    # L2 normalize result


def compute_fingerprint(
    text: str,
    config: Optional[FingerprintConfig] = None
) -> np.ndarray:
    """
    Compute n-gram fingerprint of text.
    
    Fast O(n) where n = text length.
    Returns normalized vector of size config.size.
    """
    if config is None:
        config = FingerprintConfig()
    
    fp = np.zeros(config.size, dtype=np.float32)
    text = text.lower()
    
    for ngram_size, weight in zip(config.ngram_sizes, config.weights):
        for i in range(len(text) - ngram_size + 1):
            ngram = text[i:i+ngram_size]
            idx = hash(ngram) % config.size
            fp[idx] += weight
    
    if config.normalize:
        norm = np.linalg.norm(fp)
        if norm > 0:
            fp /= norm
    
    return fp


def cosine_similarity(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Cosine similarity between fingerprints."""
    return float(np.dot(fp1, fp2))


def batch_similarities(
    query: np.ndarray,
    candidates: List[np.ndarray]
) -> np.ndarray:
    """Compute similarities between query and all candidates."""
    if not candidates:
        return np.array([])
    
    # Stack candidates into matrix
    matrix = np.stack(candidates)  # (n_candidates, size)
    
    # Dot product with query
    similarities = matrix @ query  # (n_candidates,)
    
    return similarities


def novelty_score(
    new_fp: np.ndarray,
    existing_fps: List[np.ndarray]
) -> float:
    """
    Compute how novel a fingerprint is compared to existing ones.
    
    Returns 0-1 where 1 = completely novel, 0 = identical to existing.
    """
    if not existing_fps:
        return 1.0
    
    similarities = batch_similarities(new_fp, existing_fps)
    max_similarity = float(np.max(similarities))
    
    # Novelty is inverse of max similarity
    return 1.0 - max_similarity


def cluster_fingerprints(
    fingerprints: List[np.ndarray],
    threshold: float = 0.7
) -> List[List[int]]:
    """
    Simple clustering of fingerprints by similarity.
    
    Returns list of clusters (each cluster is list of indices).
    Used for shard consolidation.
    """
    n = len(fingerprints)
    if n == 0:
        return []
    
    assigned = [False] * n
    clusters = []
    
    for i in range(n):
        if assigned[i]:
            continue
        
        # Start new cluster
        cluster = [i]
        assigned[i] = True
        
        # Find similar fingerprints
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            
            sim = cosine_similarity(fingerprints[i], fingerprints[j])
            if sim > threshold:
                cluster.append(j)
                assigned[j] = True
        
        clusters.append(cluster)
    
    return clusters


def combine_fingerprints(
    fingerprints: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """Weighted combination of fingerprints."""
    if not fingerprints:
        raise ValueError("No fingerprints to combine")
    
    if weights is None:
        weights = [1.0 / len(fingerprints)] * len(fingerprints)
    
    combined = np.zeros_like(fingerprints[0])
    for fp, w in zip(fingerprints, weights):
        combined += w * fp
    
    # Renormalize
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    
    return combined


class FingerprintIndex:
    """
    Simple index for fast fingerprint lookup.
    
    For small collections (< 10k), linear scan is fine.
    This provides a clean interface for future optimization.
    """
    
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config or FingerprintConfig()
        self.fingerprints: List[np.ndarray] = []
        self.ids: List[str] = []
    
    def add(self, id: str, fingerprint: np.ndarray):
        """Add fingerprint to index."""
        self.ids.append(id)
        self.fingerprints.append(fingerprint)
    
    def remove(self, id: str):
        """Remove fingerprint from index."""
        if id in self.ids:
            idx = self.ids.index(id)
            self.ids.pop(idx)
            self.fingerprints.pop(idx)
    
    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[tuple]:
        """
        Find top-k most similar fingerprints.
        
        Returns list of (id, similarity) tuples.
        """
        if not self.fingerprints:
            return []
        
        similarities = batch_similarities(query, self.fingerprints)
        
        # Get top-k indices
        if len(similarities) <= top_k:
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]
        
        results = []
        for idx in indices:
            sim = float(similarities[idx])
            if sim >= threshold:
                results.append((self.ids[idx], sim))
        
        return results
    
    def __len__(self):
        return len(self.ids)

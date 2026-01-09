"""
shard.py — Memory fragments of a living organism

A Shard is a trace of experience — not just data, but a moment
the organism decided was worth remembering.

Part of STANLEY's memory layer.
"""

from __future__ import annotations
import numpy as np
import json
import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal


DepthLevel = Literal["surface", "middle", "deep", "abyss"]


@dataclass
class Shard:
    """
    A memory fragment — a trace of lived experience.

    Contains LoRA-style delta weights that modify personality.
    W_effective = W_base + A @ B

    Also stores content for resonant recall (SantaClaus).
    """

    id: str
    created_at: float
    last_activated: float
    activation_count: int

    content_hash: str
    trigger_fingerprint: np.ndarray
    resonance_score: float

    layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]

    semantic_tags: List[str] = field(default_factory=list)
    depth: DepthLevel = "surface"

    # Content storage for resonant recall (SantaClaus)
    content: Optional[str] = None

    # Recall metrics (for recency penalty and stats)
    last_recalled_at: float = 0.0
    recall_count: int = 0
    
    @classmethod
    def create(
        cls,
        content: str,
        resonance: float,
        layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]],
        fingerprint: Optional[np.ndarray] = None,
        tags: Optional[List[str]] = None,
    ) -> "Shard":
        """Create a new shard from content."""
        now = time.time()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        shard_id = hashlib.sha256(f"{content_hash}{now}".encode()).hexdigest()[:12]
        
        if fingerprint is None:
            fingerprint = cls._simple_fingerprint(content)
        
        return cls(
            id=shard_id,
            created_at=now,
            last_activated=now,
            activation_count=0,
            content_hash=content_hash,
            trigger_fingerprint=fingerprint,
            resonance_score=resonance,
            layer_deltas=layer_deltas,
            semantic_tags=tags or [],
            depth="surface",
            content=content,  # Store for resonant recall
            last_recalled_at=0.0,
            recall_count=0,
        )
    
    @staticmethod
    def _simple_fingerprint(text: str, size: int = 64) -> np.ndarray:
        """Simple n-gram fingerprint."""
        fp = np.zeros(size, dtype=np.float32)
        text = text.lower()
        for i in range(len(text) - 2):
            trigram = text[i:i+3]
            idx = hash(trigram) % size
            fp[idx] += 1
        norm = np.linalg.norm(fp)
        if norm > 0:
            fp /= norm
        return fp
    
    def activate(self) -> None:
        """Record an activation."""
        self.last_activated = time.time()
        self.activation_count += 1
    
    def compressed_size(self) -> int:
        """Size in bytes."""
        total = 0
        for name, (A, B) in self.layer_deltas.items():
            total += A.nbytes + B.nbytes
        return total
    
    def apply_to(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply deltas to weights. Pure NumPy."""
        result = {}
        for name, w in weights.items():
            if name in self.layer_deltas:
                A, B = self.layer_deltas[name]
                result[name] = w + A @ B
            else:
                result[name] = w
        return result
    
    def resonates_with(self, fingerprint: np.ndarray, threshold: float = 0.3) -> bool:
        """Check resonance with context."""
        similarity = float(np.dot(self.trigger_fingerprint, fingerprint))
        return similarity > threshold
    
    def similarity_to(self, fingerprint: np.ndarray) -> float:
        """Compute similarity score."""
        return float(np.dot(self.trigger_fingerprint, fingerprint))
    
    def should_sink(self, age_threshold: float = 86400, min_activations: int = 3) -> bool:
        """Check if should sink deeper."""
        age = time.time() - self.created_at
        return age > age_threshold and self.activation_count < min_activations
    
    def save(self, directory: Path) -> Path:
        """Save to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.id}.npz"
        
        arrays = {"trigger_fingerprint": self.trigger_fingerprint}
        for name, (A, B) in self.layer_deltas.items():
            arrays[f"delta_{name}_A"] = A
            arrays[f"delta_{name}_B"] = B
        
        meta = {
            "id": self.id,
            "created_at": self.created_at,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
            "content_hash": self.content_hash,
            "resonance_score": self.resonance_score,
            "semantic_tags": self.semantic_tags,
            "depth": self.depth,
            "layer_names": list(self.layer_deltas.keys()),
            # Recall fields for SantaClaus
            "content": self.content,
            "last_recalled_at": self.last_recalled_at,
            "recall_count": self.recall_count,
        }
        arrays["_meta"] = np.array([json.dumps(meta)])
        
        np.savez_compressed(path, **arrays)
        return path
    
    @classmethod
    def load(cls, path: Path) -> "Shard":
        """Load from disk."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["_meta"][0]))
        
        layer_deltas = {}
        for name in meta["layer_names"]:
            A = data[f"delta_{name}_A"]
            B = data[f"delta_{name}_B"]
            layer_deltas[name] = (A, B)
        
        return cls(
            id=meta["id"],
            created_at=meta["created_at"],
            last_activated=meta["last_activated"],
            activation_count=meta["activation_count"],
            content_hash=meta["content_hash"],
            trigger_fingerprint=data["trigger_fingerprint"],
            resonance_score=meta["resonance_score"],
            layer_deltas=layer_deltas,
            semantic_tags=meta["semantic_tags"],
            depth=meta["depth"],
            # Recall fields (with backward compatibility)
            content=meta.get("content"),
            last_recalled_at=meta.get("last_recalled_at", 0.0),
            recall_count=meta.get("recall_count", 0),
        )


@dataclass
class MetaNote:
    """
    A compressed ghost — what remains when a shard sinks to the abyss.
    The unconscious of the organism.
    """
    
    original_id: str
    created_at: float
    compressed_at: float
    last_resonance: float
    
    attention_bias: np.ndarray
    gate_nudge: float
    semantic_fingerprint: np.ndarray
    
    @classmethod
    def from_shard(cls, shard: Shard, bias_size: int = 32) -> "MetaNote":
        """Compress a shard into a metanote."""
        attention_bias = np.zeros(bias_size, dtype=np.float32)
        gate_nudge = 0.0
        
        for name, (A, B) in shard.layer_deltas.items():
            contribution = np.mean(A) * np.mean(B)
            idx = hash(name) % bias_size
            attention_bias[idx] += contribution
            gate_nudge += contribution * 0.01
        
        norm = np.linalg.norm(attention_bias)
        if norm > 0:
            attention_bias /= norm
        gate_nudge = np.clip(gate_nudge, -0.1, 0.1)
        
        return cls(
            original_id=shard.id,
            created_at=shard.created_at,
            compressed_at=time.time(),
            last_resonance=shard.resonance_score,
            attention_bias=attention_bias,
            gate_nudge=float(gate_nudge),
            semantic_fingerprint=shard.trigger_fingerprint.copy(),
        )
    
    def can_resurrect(self, context_fingerprint: np.ndarray, threshold: float = 0.7) -> bool:
        """Check if should be resurrected."""
        similarity = float(np.dot(self.semantic_fingerprint, context_fingerprint))
        return similarity > threshold
    
    def similarity_to(self, fingerprint: np.ndarray) -> float:
        """Compute similarity."""
        return float(np.dot(self.semantic_fingerprint, fingerprint))
    
    def save(self, directory: Path) -> Path:
        """Save to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"meta_{self.original_id}.npz"
        
        meta = {
            "original_id": self.original_id,
            "created_at": self.created_at,
            "compressed_at": self.compressed_at,
            "last_resonance": self.last_resonance,
            "gate_nudge": self.gate_nudge,
        }
        
        np.savez_compressed(
            path,
            attention_bias=self.attention_bias,
            semantic_fingerprint=self.semantic_fingerprint,
            _meta=np.array([json.dumps(meta)]),
        )
        return path
    
    @classmethod
    def load(cls, path: Path) -> "MetaNote":
        """Load from disk."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["_meta"][0]))
        
        return cls(
            original_id=meta["original_id"],
            created_at=meta["created_at"],
            compressed_at=meta["compressed_at"],
            last_resonance=meta["last_resonance"],
            attention_bias=data["attention_bias"],
            gate_nudge=meta["gate_nudge"],
            semantic_fingerprint=data["semantic_fingerprint"],
        )


def combine_deltas(
    shards: List[Shard],
    weights: Optional[List[float]] = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Combine multiple shard deltas. Used for working set."""
    if not shards:
        return {}

    if weights is None:
        weights = [1.0 / len(shards)] * len(shards)

    all_layers = set()
    for shard in shards:
        all_layers.update(shard.layer_deltas.keys())

    combined = {}
    for layer in all_layers:
        relevant = [(s, w) for s, w in zip(shards, weights) if layer in s.layer_deltas]
        if not relevant:
            continue

        A0, B0 = relevant[0][0].layer_deltas[layer]
        A_sum = np.zeros_like(A0)
        B_sum = np.zeros_like(B0)

        for shard, weight in relevant:
            A, B = shard.layer_deltas[layer]
            A_sum += weight * A
            B_sum += weight * B

        combined[layer] = (A_sum, B_sum)

    return combined


# ============================================================================
# SOMATIC SHARD — Body Memory (Metric Patterns)
# ============================================================================


@dataclass
class SomaticShard:
    """
    A somatic marker — body memory of how a moment FELT.

    Unlike content shards that remember WHAT happened,
    somatic shards remember HOW IT FELT:
    - The pattern of metrics (entropy, novelty, arousal)
    - The outcome valence (was this good or bad?)
    - Context tags (what was happening)

    Philosophy: "When the moment felt like THIS, things went THAT way."

    This is Stanley's body memory — emotional associations with states.
    Used for prediction: similar metric patterns → similar outcomes.

    Inspired by Damasio's somatic markers theory:
    - Body states guide decision making
    - Emotions are embodied predictions
    - "Gut feelings" are learned associations
    """

    id: str
    created_at: float

    # Metric fingerprint (how the moment felt)
    entropy: float
    novelty: float
    arousal: float
    valence: float

    # Outcome (was this good or bad?)
    outcome_quality: float  # 0.0 = bad, 1.0 = good
    outcome_tag: str  # "good", "bad", "neutral", "intense", etc.

    # Context
    context_tags: List[str] = field(default_factory=list)
    context_description: Optional[str] = None

    # Metrics fingerprint as vector (for similarity)
    metrics_vector: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Activation stats
    last_activated: float = 0.0
    activation_count: int = 0

    @classmethod
    def create(
        cls,
        entropy: float,
        novelty: float,
        arousal: float,
        valence: float,
        outcome_quality: float,
        outcome_tag: str = "neutral",
        context_tags: Optional[List[str]] = None,
        context_description: Optional[str] = None,
    ) -> "SomaticShard":
        """
        Create a somatic shard from a felt moment.

        Args:
            entropy: Field entropy (0-1)
            novelty: Novelty of input (0-1)
            arousal: Emotional intensity (0-1)
            valence: Positive/negative (-1 to 1)
            outcome_quality: How well things went (0-1)
            outcome_tag: Categorical outcome label
            context_tags: Semantic tags for context
            context_description: Brief description
        """
        now = time.time()
        shard_id = hashlib.sha256(
            f"{entropy}{novelty}{arousal}{valence}{now}".encode()
        ).hexdigest()[:12]

        # Create metrics vector for similarity computation
        metrics_vector = np.array([entropy, novelty, arousal, (valence + 1) / 2])

        return cls(
            id=shard_id,
            created_at=now,
            entropy=entropy,
            novelty=novelty,
            arousal=arousal,
            valence=valence,
            outcome_quality=outcome_quality,
            outcome_tag=outcome_tag,
            context_tags=context_tags or [],
            context_description=context_description,
            metrics_vector=metrics_vector,
            last_activated=now,
            activation_count=0,
        )

    @classmethod
    def from_pulse(
        cls,
        pulse: "Pulse",
        outcome_quality: float,
        outcome_tag: str = "neutral",
        context_tags: Optional[List[str]] = None,
    ) -> "SomaticShard":
        """Create from a Pulse object."""
        return cls.create(
            entropy=pulse.entropy,
            novelty=pulse.novelty,
            arousal=pulse.arousal,
            valence=pulse.valence,
            outcome_quality=outcome_quality,
            outcome_tag=outcome_tag,
            context_tags=context_tags,
        )

    def similarity_to(self, other_metrics: np.ndarray) -> float:
        """
        Compute similarity to another metrics vector.

        Returns similarity in [0, 1].
        """
        # Cosine similarity
        dot = np.dot(self.metrics_vector, other_metrics)
        norm_self = np.linalg.norm(self.metrics_vector)
        norm_other = np.linalg.norm(other_metrics)

        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0

        return float(dot / (norm_self * norm_other))

    def matches_state(
        self,
        entropy: float,
        novelty: float,
        arousal: float,
        valence: float,
        threshold: float = 0.8,
    ) -> bool:
        """Check if this shard matches a body state."""
        other = np.array([entropy, novelty, arousal, (valence + 1) / 2])
        return self.similarity_to(other) >= threshold

    def activate(self) -> None:
        """Record an activation."""
        self.last_activated = time.time()
        self.activation_count += 1

    def predict_outcome(self) -> Tuple[float, str]:
        """
        Predict outcome based on this somatic marker.

        Returns (quality, tag) prediction.
        """
        return (self.outcome_quality, self.outcome_tag)

    def save(self, directory: Path) -> Path:
        """Save to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"somatic_{self.id}.npz"

        meta = {
            "id": self.id,
            "created_at": self.created_at,
            "entropy": self.entropy,
            "novelty": self.novelty,
            "arousal": self.arousal,
            "valence": self.valence,
            "outcome_quality": self.outcome_quality,
            "outcome_tag": self.outcome_tag,
            "context_tags": self.context_tags,
            "context_description": self.context_description,
            "last_activated": self.last_activated,
            "activation_count": self.activation_count,
        }

        np.savez_compressed(
            path,
            metrics_vector=self.metrics_vector,
            _meta=np.array([json.dumps(meta)]),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "SomaticShard":
        """Load from disk."""
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["_meta"][0]))

        return cls(
            id=meta["id"],
            created_at=meta["created_at"],
            entropy=meta["entropy"],
            novelty=meta["novelty"],
            arousal=meta["arousal"],
            valence=meta["valence"],
            outcome_quality=meta["outcome_quality"],
            outcome_tag=meta["outcome_tag"],
            context_tags=meta.get("context_tags", []),
            context_description=meta.get("context_description"),
            metrics_vector=data["metrics_vector"],
            last_activated=meta.get("last_activated", 0.0),
            activation_count=meta.get("activation_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"SomaticShard(id={self.id[:8]}, "
            f"outcome={self.outcome_tag}:{self.outcome_quality:.2f}, "
            f"metrics=[e={self.entropy:.2f}, n={self.novelty:.2f}, "
            f"a={self.arousal:.2f}, v={self.valence:.2f}])"
        )


class SomaticMemory:
    """
    Collection of somatic shards — Stanley's body memory.

    Stores felt experiences and predicts outcomes
    based on similar body states.

    "When I felt like this before, things went..."
    """

    def __init__(self, max_shards: int = 500):
        self.shards: List[SomaticShard] = []
        self.max_shards = max_shards

    def add(self, shard: SomaticShard) -> None:
        """Add a somatic shard."""
        self.shards.append(shard)
        # Keep bounded
        if len(self.shards) > self.max_shards:
            # Remove oldest, least activated
            self.shards.sort(
                key=lambda s: (s.activation_count, s.last_activated),
                reverse=True,
            )
            self.shards = self.shards[:self.max_shards]

    def record_moment(
        self,
        entropy: float,
        novelty: float,
        arousal: float,
        valence: float,
        outcome_quality: float,
        outcome_tag: str = "neutral",
        context_tags: Optional[List[str]] = None,
    ) -> SomaticShard:
        """
        Record a felt moment as a somatic marker.

        Call this after Stanley responds, with the outcome quality.
        """
        shard = SomaticShard.create(
            entropy=entropy,
            novelty=novelty,
            arousal=arousal,
            valence=valence,
            outcome_quality=outcome_quality,
            outcome_tag=outcome_tag,
            context_tags=context_tags,
        )
        self.add(shard)
        return shard

    def predict_outcome(
        self,
        entropy: float,
        novelty: float,
        arousal: float,
        valence: float,
        top_k: int = 5,
    ) -> Tuple[float, str, int]:
        """
        Predict outcome based on similar past body states.

        Returns (predicted_quality, predicted_tag, num_matches).
        """
        if not self.shards:
            return (0.5, "unknown", 0)

        metrics = np.array([entropy, novelty, arousal, (valence + 1) / 2])

        # Find similar shards
        similarities = [
            (shard, shard.similarity_to(metrics))
            for shard in self.shards
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Take top_k
        top = similarities[:top_k]
        if not top or top[0][1] < 0.5:  # No good matches
            return (0.5, "unknown", 0)

        # Weighted average of outcomes
        total_weight = 0.0
        weighted_quality = 0.0
        tag_counts: Dict[str, float] = {}

        for shard, sim in top:
            if sim < 0.5:
                continue
            shard.activate()
            weighted_quality += shard.outcome_quality * sim
            total_weight += sim
            tag_counts[shard.outcome_tag] = tag_counts.get(shard.outcome_tag, 0) + sim

        if total_weight < 1e-8:
            return (0.5, "unknown", 0)

        avg_quality = weighted_quality / total_weight
        best_tag = max(tag_counts.items(), key=lambda x: x[1])[0] if tag_counts else "unknown"

        return (avg_quality, best_tag, len([s for s, sim in top if sim >= 0.5]))

    def find_similar(
        self,
        entropy: float,
        novelty: float,
        arousal: float,
        valence: float,
        threshold: float = 0.7,
    ) -> List[SomaticShard]:
        """Find shards with similar body states."""
        metrics = np.array([entropy, novelty, arousal, (valence + 1) / 2])
        return [
            shard for shard in self.shards
            if shard.similarity_to(metrics) >= threshold
        ]

    def get_stats(self) -> Dict:
        """Get somatic memory statistics."""
        if not self.shards:
            return {
                "total_shards": 0,
                "avg_quality": 0.5,
                "outcome_distribution": {},
            }

        outcomes = {}
        total_quality = 0.0

        for shard in self.shards:
            outcomes[shard.outcome_tag] = outcomes.get(shard.outcome_tag, 0) + 1
            total_quality += shard.outcome_quality

        return {
            "total_shards": len(self.shards),
            "avg_quality": total_quality / len(self.shards),
            "outcome_distribution": outcomes,
        }

    def __len__(self) -> int:
        return len(self.shards)

    def __repr__(self) -> str:
        return f"SomaticMemory(shards={len(self.shards)})"

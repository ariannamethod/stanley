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

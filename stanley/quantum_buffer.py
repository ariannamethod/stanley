"""
quantum_buffer.py — Accumulation before training

Shards don't trigger training immediately.
They accumulate until a "quantum" of adaptation is reached.

This prevents over-training and allows organic batch formation.
"""

from __future__ import annotations
import numpy as np
import time
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import logging

from .shard import Shard

logger = logging.getLogger(__name__)


@dataclass
class QuantumBuffer:
    """
    Accumulates shards until training threshold is reached.
    
    Thresholds:
    - bytes_delta: minimum new content volume
    - resonance_mass: weighted sum of resonance scores
    - novelty_mass: how much shards differ from current state
    - cooldown: minimum time between trainings
    
    All conditions use OR logic (any threshold triggers).
    Cooldown is always respected.
    """
    
    pending: List[Shard] = field(default_factory=list)
    
    # Thresholds
    min_bytes: int = 1024           # 1KB minimum
    min_resonance_mass: float = 3.0  # sum of resonance scores
    min_novelty: float = 0.15        # average novelty threshold
    min_shards: int = 3              # minimum shards to trigger
    cooldown_seconds: float = 60.0   # minimum 1 minute between trains
    
    # State
    last_train_time: float = 0.0
    total_trains: int = 0
    
    # Optional: novelty computation callback
    novelty_fn: Optional[Callable[[List[Shard]], float]] = None
    
    def add(self, shard: Shard) -> bool:
        """
        Add a shard to the buffer.
        
        Returns True if quantum threshold is now reached.
        """
        self.pending.append(shard)
        logger.debug(f"Buffer: added shard {shard.id}, total pending: {len(self.pending)}")
        return self.should_trigger()
    
    def should_trigger(self) -> bool:
        """Check if quantum threshold is reached."""
        if not self.pending:
            return False
        
        # Cooldown always respected
        time_since_last = time.time() - self.last_train_time
        if time_since_last < self.cooldown_seconds:
            return False
        
        # Minimum shards
        if len(self.pending) < self.min_shards:
            return False
        
        # Check mass thresholds (OR logic)
        bytes_delta = self._compute_bytes()
        resonance_mass = self._compute_resonance_mass()
        novelty_mass = self._compute_novelty()
        
        triggered = (
            bytes_delta >= self.min_bytes or
            resonance_mass >= self.min_resonance_mass or
            novelty_mass >= self.min_novelty
        )
        
        if triggered:
            logger.info(
                f"Quantum triggered: bytes={bytes_delta}, "
                f"resonance={resonance_mass:.2f}, novelty={novelty_mass:.2f}"
            )
        
        return triggered
    
    def _compute_bytes(self) -> int:
        """Total bytes in pending shards."""
        return sum(s.compressed_size() for s in self.pending)
    
    def _compute_resonance_mass(self) -> float:
        """Weighted sum of resonance scores."""
        return sum(s.resonance_score for s in self.pending)
    
    def _compute_novelty(self) -> float:
        """
        Average novelty of pending shards.
        
        If novelty_fn is set, uses that.
        Otherwise, uses fingerprint diversity as proxy.
        """
        if self.novelty_fn:
            return self.novelty_fn(self.pending)
        
        # Default: average pairwise distance of fingerprints
        if len(self.pending) < 2:
            return 0.0
        
        fps = [s.trigger_fingerprint for s in self.pending]
        total_dist = 0.0
        count = 0
        
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                dist = 1.0 - float(np.dot(fps[i], fps[j]))
                total_dist += dist
                count += 1
        
        return total_dist / count if count > 0 else 0.0
    
    def flush(self) -> List[Shard]:
        """
        Return pending shards and clear buffer.
        
        Called when training is triggered.
        """
        shards = self.pending
        self.pending = []
        self.last_train_time = time.time()
        self.total_trains += 1
        
        logger.info(f"Buffer flushed: {len(shards)} shards, train #{self.total_trains}")
        return shards
    
    def peek(self) -> List[Shard]:
        """View pending shards without removing."""
        return list(self.pending)
    
    def clear(self):
        """Clear buffer without training."""
        self.pending = []
    
    def stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "pending": len(self.pending),
            "bytes": self._compute_bytes(),
            "resonance_mass": self._compute_resonance_mass(),
            "novelty": self._compute_novelty(),
            "total_trains": self.total_trains,
            "time_since_train": time.time() - self.last_train_time,
            "cooldown_remaining": max(
                0, 
                self.cooldown_seconds - (time.time() - self.last_train_time)
            ),
        }


@dataclass
class AdaptiveQuantumBuffer(QuantumBuffer):
    """
    Quantum buffer that adapts thresholds based on organism state.
    
    If organism is "young" (few shards), lower thresholds.
    If organism is "mature" (many shards), higher thresholds.
    """
    
    # Organism state callback
    get_organism_age: Optional[Callable[[], float]] = None  # returns 0-1
    
    # Multipliers for mature organism
    mature_bytes_mult: float = 2.0
    mature_resonance_mult: float = 1.5
    mature_cooldown_mult: float = 2.0
    
    def _effective_thresholds(self) -> tuple:
        """Get age-adjusted thresholds."""
        if self.get_organism_age is None:
            return self.min_bytes, self.min_resonance_mass, self.cooldown_seconds
        
        age = self.get_organism_age()  # 0 = newborn, 1 = mature
        
        # Linear interpolation
        bytes_th = self.min_bytes * (1 + age * (self.mature_bytes_mult - 1))
        resonance_th = self.min_resonance_mass * (1 + age * (self.mature_resonance_mult - 1))
        cooldown = self.cooldown_seconds * (1 + age * (self.mature_cooldown_mult - 1))
        
        return int(bytes_th), resonance_th, cooldown
    
    def should_trigger(self) -> bool:
        """Check with adaptive thresholds."""
        if not self.pending:
            return False
        
        min_bytes, min_resonance, cooldown = self._effective_thresholds()
        
        time_since_last = time.time() - self.last_train_time
        if time_since_last < cooldown:
            return False
        
        if len(self.pending) < self.min_shards:
            return False
        
        bytes_delta = self._compute_bytes()
        resonance_mass = self._compute_resonance_mass()
        novelty_mass = self._compute_novelty()
        
        return (
            bytes_delta >= min_bytes or
            resonance_mass >= min_resonance or
            novelty_mass >= self.min_novelty
        )

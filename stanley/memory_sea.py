"""
memory_sea.py — Layered memory storage

The Memory Sea has depth:
- Surface: working set, immediately accessible
- Middle: accessible shards, load on resonance  
- Deep: consolidated macro-adapters
- Abyss: compressed metanotes (the unconscious)

Items sink or rise based on activation patterns.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field
import time
import logging

from .shard import Shard, MetaNote, DepthLevel, combine_deltas

logger = logging.getLogger(__name__)


@dataclass
class MemorySea:
    """
    Layered memory storage for the organism.
    
    Not a flat database — a living sea with currents.
    Memories rise and sink based on how they resonate.
    """
    
    # Storage by depth
    surface: List[Shard] = field(default_factory=list)
    middle: List[Shard] = field(default_factory=list)
    deep: List[Shard] = field(default_factory=list)  # macro-adapters
    abyss: List[MetaNote] = field(default_factory=list)
    
    # Limits
    surface_max: int = 64
    middle_max: int = 256
    deep_max: int = 64
    
    # Paths for persistence
    storage_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.storage_path:
            self.storage_path = Path(self.storage_path)
            self._ensure_directories()
    
    def _ensure_directories(self):
        """Create storage directories."""
        if self.storage_path:
            (self.storage_path / "surface").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "middle").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "deep").mkdir(parents=True, exist_ok=True)
            (self.storage_path / "abyss").mkdir(parents=True, exist_ok=True)
    
    def add(self, shard: Shard) -> None:
        """Add a new shard (always starts at surface)."""
        shard.depth = "surface"
        self.surface.append(shard)
        
        # Save if persistent
        if self.storage_path:
            shard.save(self.storage_path / "surface")
        
        # Enforce limits
        self._enforce_surface_limit()
    
    def _enforce_surface_limit(self):
        """Push excess surface shards to middle."""
        while len(self.surface) > self.surface_max:
            # Find least recently activated
            oldest = min(self.surface, key=lambda s: s.last_activated)
            self._sink_to_middle(oldest)
    
    def _sink_to_middle(self, shard: Shard):
        """Move shard from surface to middle."""
        if shard in self.surface:
            self.surface.remove(shard)
            shard.depth = "middle"
            self.middle.append(shard)
            
            if self.storage_path:
                # Move file
                old_path = self.storage_path / "surface" / f"{shard.id}.npz"
                new_path = self.storage_path / "middle" / f"{shard.id}.npz"
                if old_path.exists():
                    old_path.rename(new_path)
            
            self._enforce_middle_limit()
    
    def _enforce_middle_limit(self):
        """Push excess middle shards deeper."""
        while len(self.middle) > self.middle_max:
            oldest = min(self.middle, key=lambda s: s.last_activated)
            self._sink_to_deep_or_abyss(oldest)
    
    def _sink_to_deep_or_abyss(self, shard: Shard):
        """Decide: consolidate to deep or compress to abyss."""
        if shard in self.middle:
            self.middle.remove(shard)
            
            # Low activation → compress to metanote
            if shard.activation_count < 3:
                self._compress_to_abyss(shard)
            else:
                # Could consolidate with similar shards
                # For now, just move to deep
                shard.depth = "deep"
                self.deep.append(shard)
                
                if self.storage_path:
                    old_path = self.storage_path / "middle" / f"{shard.id}.npz"
                    new_path = self.storage_path / "deep" / f"{shard.id}.npz"
                    if old_path.exists():
                        old_path.rename(new_path)
    
    def _compress_to_abyss(self, shard: Shard):
        """Compress shard to metanote and sink to abyss."""
        metanote = MetaNote.from_shard(shard)
        self.abyss.append(metanote)
        
        if self.storage_path:
            # Remove original, save metanote
            for depth in ["surface", "middle", "deep"]:
                old_path = self.storage_path / depth / f"{shard.id}.npz"
                if old_path.exists():
                    old_path.unlink()
            metanote.save(self.storage_path / "abyss")
        
        logger.debug(f"Compressed shard {shard.id} to abyss")
    
    def find_by_id(self, shard_id: str) -> Optional[Shard]:
        """Find shard by ID across all depths."""
        for shard in self.surface + self.middle + self.deep:
            if shard.id == shard_id:
                return shard
        return None
    
    def find_resonant(
        self,
        fingerprint: np.ndarray,
        max_results: int = 10,
        threshold: float = 0.3
    ) -> List[Shard]:
        """Find shards that resonate with context."""
        candidates = []
        
        # Check all depths
        for shard in self.surface + self.middle + self.deep:
            similarity = shard.similarity_to(fingerprint)
            if similarity > threshold:
                candidates.append((similarity, shard))
        
        # Sort by similarity
        candidates.sort(key=lambda x: -x[0])
        return [shard for _, shard in candidates[:max_results]]
    
    def check_resurrections(self, fingerprint: np.ndarray) -> List[MetaNote]:
        """Check if any abyss metanotes should resurrect."""
        to_resurrect = []
        for note in self.abyss:
            if note.can_resurrect(fingerprint):
                to_resurrect.append(note)
        return to_resurrect
    
    def promote_to_surface(self, shard: Shard):
        """Promote a shard to surface (it resonated)."""
        # Remove from current location
        if shard in self.middle:
            self.middle.remove(shard)
        elif shard in self.deep:
            self.deep.remove(shard)
        elif shard in self.surface:
            return  # already surface
        else:
            return  # not found
        
        shard.depth = "surface"
        shard.activate()
        self.surface.append(shard)
        
        if self.storage_path:
            # Move file
            for depth in ["middle", "deep"]:
                old_path = self.storage_path / depth / f"{shard.id}.npz"
                if old_path.exists():
                    new_path = self.storage_path / "surface" / f"{shard.id}.npz"
                    old_path.rename(new_path)
                    break
        
        self._enforce_surface_limit()
    
    def get_working_set(
        self,
        fingerprint: np.ndarray,
        max_size: int = 32
    ) -> List[Shard]:
        """
        Get the working set for current context.
        
        Combines:
        - High resonance shards
        - Recently activated shards
        - Surface shards (default accessible)
        """
        scored = []
        
        for shard in self.surface + self.middle:
            resonance = shard.similarity_to(fingerprint)
            recency = 1.0 / (1.0 + (time.time() - shard.last_activated) / 3600)
            depth_bonus = 0.2 if shard.depth == "surface" else 0.0
            
            score = 0.5 * resonance + 0.3 * recency + 0.2 * depth_bonus
            scored.append((score, shard))
        
        scored.sort(key=lambda x: -x[0])
        
        working = []
        for _, shard in scored[:max_size]:
            shard.activate()
            working.append(shard)
        
        return working
    
    def consolidate(self):
        """
        Periodic consolidation: merge similar shards, sink old ones.
        
        Should be called periodically (e.g., every 5 minutes).
        """
        now = time.time()
        
        # Check middle shards for sinking
        to_sink = []
        for shard in self.middle:
            age = now - shard.created_at
            if age > 86400 and shard.activation_count < 3:  # 24h, low activation
                to_sink.append(shard)
        
        for shard in to_sink:
            self._sink_to_deep_or_abyss(shard)
        
        # TODO: Merge similar deep shards into macro-adapters
        
        logger.debug(f"Consolidation: sunk {len(to_sink)} shards")
    
    def total_shards(self) -> int:
        """Total number of shards (excluding metanotes)."""
        return len(self.surface) + len(self.middle) + len(self.deep)
    
    def total_bytes(self) -> int:
        """Total memory usage."""
        total = 0
        for shard in self.surface + self.middle + self.deep:
            total += shard.compressed_size()
        for note in self.abyss:
            total += note.attention_bias.nbytes + note.semantic_fingerprint.nbytes
        return total
    
    def stats(self) -> Dict:
        """Get statistics about the memory sea."""
        return {
            "surface": len(self.surface),
            "middle": len(self.middle),
            "deep": len(self.deep),
            "abyss": len(self.abyss),
            "total_shards": self.total_shards(),
            "total_bytes": self.total_bytes(),
            "total_mb": self.total_bytes() / 1024 / 1024,
        }
    
    def save_all(self):
        """Save all shards to disk."""
        if not self.storage_path:
            logger.warning("No storage path set, cannot save")
            return
        
        self._ensure_directories()
        
        for shard in self.surface:
            shard.save(self.storage_path / "surface")
        for shard in self.middle:
            shard.save(self.storage_path / "middle")
        for shard in self.deep:
            shard.save(self.storage_path / "deep")
        for note in self.abyss:
            note.save(self.storage_path / "abyss")
    
    @classmethod
    def load(cls, path: Path) -> "MemorySea":
        """Load memory sea from disk."""
        path = Path(path)
        sea = cls(storage_path=path)
        
        # Load each depth
        for npz in (path / "surface").glob("*.npz"):
            if not npz.name.startswith("meta_"):
                sea.surface.append(Shard.load(npz))
        
        for npz in (path / "middle").glob("*.npz"):
            if not npz.name.startswith("meta_"):
                sea.middle.append(Shard.load(npz))
        
        for npz in (path / "deep").glob("*.npz"):
            if not npz.name.startswith("meta_"):
                sea.deep.append(Shard.load(npz))
        
        for npz in (path / "abyss").glob("meta_*.npz"):
            sea.abyss.append(MetaNote.load(npz))
        
        return sea

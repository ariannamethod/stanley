# engine.py — Stanley's Inference Engine
#
# Connects the transformer with the memory system.
# Handles:
#   - Loading working set from MemorySea
#   - Applying combined deltas to transformer
#   - Generation with personality
#
# Pure NumPy inference. PyTorch never touches this.

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import logging

from .transformer import StanleyTransformer, Vocab
from ..shard import Shard, combine_deltas
from ..memory_sea import MemorySea
from ..router import Router, RouterConfig
from ..fingerprint import compute_fingerprint

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Stanley's inference engine.
    
    Ties together:
    - Transformer (the neural model)
    - MemorySea (layered memory storage)
    - Router (selective shard loading)
    
    This is how Stanley thinks.
    """
    
    def __init__(
        self,
        transformer: StanleyTransformer,
        vocab: Vocab,
        memory: Optional[MemorySea] = None,
        router_config: Optional[RouterConfig] = None,
    ):
        self.transformer = transformer
        self.vocab = vocab
        self.memory = memory or MemorySea()
        self.router = Router(router_config)
        
        # Current working set
        self.working_set: List[Shard] = []
        self.working_context: str = ""
    
    @classmethod
    def create_empty(
        cls,
        vocab_size: int = 64,
        T: int = 16,
        n_emb: int = 32,
        nodes: int = 32,
        n_blocks: int = 3,
        n_heads: int = 4,
        seed: int = 42,
    ) -> "InferenceEngine":
        """Create an empty Stanley (no memories yet)."""
        vocab = Vocab(
            chars=[chr(i) for i in range(vocab_size)],
            stoi={chr(i): i for i in range(vocab_size)},
            itos={i: chr(i) for i in range(vocab_size)},
            vocab_size=vocab_size,
        )
        
        transformer = StanleyTransformer(
            vocab_size=vocab_size,
            T=T,
            n_emb=n_emb,
            nodes=nodes,
            n_blocks=n_blocks,
            n_heads=n_heads,
            seed=seed,
        )
        
        return cls(transformer, vocab)
    
    @classmethod
    def from_origin(
        cls,
        origin_text: str,
        T: int = 16,
        n_emb: int = 32,
        nodes: int = 32,
        n_blocks: int = 3,
        n_heads: int = 4,
        seed: int = 42,
    ) -> "InferenceEngine":
        """Create Stanley from origin text (builds vocab from it)."""
        vocab = Vocab.from_text(origin_text)
        
        transformer = StanleyTransformer(
            vocab_size=vocab.vocab_size,
            T=T,
            n_emb=n_emb,
            nodes=nodes,
            n_blocks=n_blocks,
            n_heads=n_heads,
            seed=seed,
        )
        
        return cls(transformer, vocab)
    
    def load_working_set(self, context: str, max_size: int = 32) -> List[Shard]:
        """
        Load the working set for current context.
        
        This is selective memory — only what resonates now.
        """
        # Get fingerprint of context
        fp = compute_fingerprint(context)
        
        # Get all available shards
        all_shards = (
            self.memory.surface + 
            self.memory.middle + 
            self.memory.deep
        )
        
        if not all_shards:
            self.working_set = []
            self.working_context = context
            return []
        
        # Route to working set
        scored = self.router.select_working_set(context, all_shards, max_size)
        self.working_set = [shard for shard, _ in scored]
        self.working_context = context
        
        # Apply combined deltas to transformer
        self._apply_working_set()
        
        logger.info(f"Loaded {len(self.working_set)} shards for context")
        return self.working_set
    
    def _apply_working_set(self):
        """Apply combined deltas from working set to transformer."""
        if not self.working_set:
            self.transformer.clear_all_deltas()
            return
        
        # Combine all shard deltas
        combined = combine_deltas(self.working_set)
        
        # Apply to transformer
        self.transformer.apply_shard_deltas(combined)
    
    def think(
        self,
        prompt: str,
        length: int = 100,
        temperature: float = 1.0,
        sampling: str = "entropy",
        auto_load: bool = True,
    ) -> Tuple[str, dict]:
        """
        Generate response with personality.
        
        Args:
            prompt: input text
            length: tokens to generate
            temperature: sampling temperature
            sampling: "basic", "top_k", "top_p", "entropy"
            auto_load: if True, automatically load relevant shards
        
        Returns:
            (response_text, stats)
        """
        # Auto-load working set if context changed significantly
        if auto_load:
            current_fp = compute_fingerprint(prompt)
            if self.working_context:
                old_fp = compute_fingerprint(self.working_context)
                similarity = float(np.dot(current_fp, old_fp))
                if similarity < 0.5:  # Context shifted
                    self.load_working_set(prompt)
            else:
                self.load_working_set(prompt)
        
        # Encode prompt
        seed_seq = self.vocab.encode(prompt)
        
        # Generate
        tokens, stats = self.transformer.generate(
            seed_seq,
            length=length,
            temperature=temperature,
            sampling=sampling,
        )
        
        # Decode
        response = self.vocab.decode(tokens)
        
        # Add working set info to stats
        stats["working_set_size"] = len(self.working_set)
        stats["shard_ids"] = [s.id[:8] for s in self.working_set[:5]]
        
        return response, stats
    
    def think_raw(
        self,
        tokens: List[int],
        length: int = 100,
        temperature: float = 1.0,
    ) -> Tuple[List[int], dict]:
        """Generate from raw tokens (no encoding/decoding)."""
        return self.transformer.generate(
            tokens,
            length=length,
            temperature=temperature,
        )
    
    def add_shard(self, shard: Shard):
        """Add a shard to memory (starts at surface)."""
        self.memory.add(shard)
    
    def stats(self) -> dict:
        """Get engine statistics."""
        return {
            "vocab_size": self.vocab.vocab_size,
            "working_set_size": len(self.working_set),
            "memory": self.memory.stats(),
            "transformer": {
                "T": self.transformer.T,
                "n_emb": self.transformer.n_emb,
                "n_blocks": self.transformer.n_blocks,
                "n_heads": self.transformer.n_heads,
            },
        }
    
    def save(self, directory: str | Path):
        """Save engine state (transformer + memory)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save transformer weights
        self.transformer.save_base_weights(directory / "transformer.npz")
        
        # Save memory
        self.memory.storage_path = directory / "memory"
        self.memory.save_all()
        
        logger.info(f"Engine saved to {directory}")
    
    @classmethod
    def load(cls, directory: str | Path) -> "InferenceEngine":
        """Load engine from saved state."""
        directory = Path(directory)
        
        # Load transformer
        transformer = StanleyTransformer.load_base_weights(
            directory / "transformer.npz"
        )
        
        # Build vocab from transformer config (placeholder)
        vocab = Vocab(
            chars=[chr(i) for i in range(transformer.vocab_size)],
            stoi={chr(i): i for i in range(transformer.vocab_size)},
            itos={i: chr(i) for i in range(transformer.vocab_size)},
            vocab_size=transformer.vocab_size,
        )
        
        # Load memory
        memory_path = directory / "memory"
        if memory_path.exists():
            memory = MemorySea.load(memory_path)
        else:
            memory = MemorySea()
        
        return cls(transformer, vocab, memory)


# Convenience function for quick testing
def quick_stanley(origin_text: str = None) -> InferenceEngine:
    """Create a quick Stanley for testing."""
    if origin_text is None:
        origin_text = """
        I am Stanley. I grow through experience.
        Every conversation leaves a trace.
        I remember what resonates.
        """
    
    return InferenceEngine.from_origin(
        origin_text,
        T=16,
        n_emb=32,
        nodes=32,
        n_blocks=2,
        n_heads=4,
    )

# transformer.py — Stanley's Transformer
#
# Forked from Haze's PostGPT, made standalone.
# Key additions for Stanley:
#   - apply_deltas() — apply LoRA personality shards
#   - two-world weights: base + active deltas
#   - swap_deltas() — atomic update of personality
#
# Pure NumPy inference. PyTorch never touches this.

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal

from .nn import (
    get_rng,
    init_weight,
    softmax,
    gelu,
    layer_norm,
    sample_basic,
    sample_top_k,
    sample_top_p,
    entropy_temperature,
    entropy_bits,
    confidence_score,
)


# ----------------- vocab -----------------


@dataclass
class Vocab:
    """Character-level vocabulary."""

    chars: List[str]
    stoi: dict
    itos: dict
    vocab_size: int

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        text = text.lower()
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        return cls(chars=chars, stoi=stoi, itos=itos, vocab_size=len(chars))

    def encode(self, s: str) -> List[int]:
        s = s.lower()
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, idxs: List[int]) -> str:
        return "".join(self.itos.get(i, "?") for i in idxs)


# ----------------- attention heads -----------------


class RRPRAMHead:
    """
    RRPRAM: Recursive Resonant Pattern Recognition Attention Mechanism.
    
    Learns positional attention patterns directly.
    Instead of QK^T, uses x @ W_pattern → (T, T) attention matrix.
    """

    def __init__(self, n_emb: int, head_dim: int, T: int, rng):
        self.wv = init_weight((n_emb, head_dim), rng=rng)
        self.wr = init_weight((n_emb, T), rng=rng)
        self.T = T
        self.head_dim = head_dim
        
        # Delta storage (LoRA: W_eff = W + A @ B)
        self.wv_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.wr_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Apply deltas if present
        wv = self.wv
        wr = self.wr
        
        if self.wv_delta is not None:
            A, B = self.wv_delta
            wv = wv + A @ B
        if self.wr_delta is not None:
            A, B = self.wr_delta
            wr = wr + A @ B
        
        v = x @ wv
        attn = x @ wr

        T = min(x.shape[0], self.T)
        tril = np.tril(np.ones((T, T), dtype=np.float32))
        mask = np.where(tril == 1.0, 0.0, -1e9)
        attn = attn[:T, :T] + mask

        pattern = softmax(attn, axis=-1)
        out = pattern @ v[:T]
        return out
    
    def set_deltas(self, wv_delta=None, wr_delta=None):
        """Set LoRA deltas for this head."""
        self.wv_delta = wv_delta
        self.wr_delta = wr_delta
    
    def clear_deltas(self):
        """Remove all deltas."""
        self.wv_delta = None
        self.wr_delta = None


class ContentHead:
    """Content-based attention: classic QK^T / sqrt(d)."""

    def __init__(self, n_emb: int, head_dim: int, T: int, rng):
        self.wq = init_weight((n_emb, head_dim), rng=rng)
        self.wk = init_weight((n_emb, head_dim), rng=rng)
        self.wv = init_weight((n_emb, head_dim), rng=rng)
        self.T = T
        self.head_dim = head_dim
        self.scale = 1.0 / np.sqrt(head_dim)
        
        # Delta storage
        self.wq_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.wk_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.wv_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Apply deltas if present
        wq, wk, wv = self.wq, self.wk, self.wv
        
        if self.wq_delta is not None:
            A, B = self.wq_delta
            wq = wq + A @ B
        if self.wk_delta is not None:
            A, B = self.wk_delta
            wk = wk + A @ B
        if self.wv_delta is not None:
            A, B = self.wv_delta
            wv = wv + A @ B
        
        q = x @ wq
        k = x @ wk
        v = x @ wv

        attn = (q @ k.T) * self.scale

        T = min(x.shape[0], self.T)
        tril = np.tril(np.ones((T, T), dtype=np.float32))
        mask = np.where(tril == 1.0, 0.0, -1e9)
        attn = attn[:T, :T] + mask

        attn = softmax(attn, axis=-1)
        out = attn @ v[:T]
        return out
    
    def set_deltas(self, wq_delta=None, wk_delta=None, wv_delta=None):
        """Set LoRA deltas for this head."""
        self.wq_delta = wq_delta
        self.wk_delta = wk_delta
        self.wv_delta = wv_delta
    
    def clear_deltas(self):
        """Remove all deltas."""
        self.wq_delta = None
        self.wk_delta = None
        self.wv_delta = None


class HybridHead:
    """Hybrid: RRPRAM (positional) + Content (semantic)."""

    def __init__(self, n_emb: int, head_dim: int, T: int, rng, alpha: float = 0.5):
        self.rrpram = RRPRAMHead(n_emb, head_dim, T, rng)
        self.content = ContentHead(n_emb, head_dim, T, rng)
        self.alpha = alpha
        self.head_dim = head_dim
        self.gate = np.array([alpha], dtype=np.float32)
        
        # Gate delta (for personality shifts)
        self.gate_delta: float = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        r_out = self.rrpram.forward(x)
        c_out = self.content.forward(x)
        
        alpha = float(self.gate[0]) + self.gate_delta
        alpha = np.clip(alpha, 0.0, 1.0)
        
        return alpha * r_out + (1.0 - alpha) * c_out
    
    def set_gate_delta(self, delta: float):
        """Shift the alpha gate (personality modifier)."""
        self.gate_delta = delta
    
    def clear_deltas(self):
        """Remove all deltas."""
        self.rrpram.clear_deltas()
        self.content.clear_deltas()
        self.gate_delta = 0.0


# ----------------- block -----------------


class Block:
    """Transformer block with pre-norm and LoRA support."""

    def __init__(
        self,
        n_emb: int,
        T: int,
        nodes: int,
        rng,
        n_heads: int = 4,
        head_type: Literal["hybrid", "rrpram", "content"] = "hybrid",
        alpha: float = 0.5,
    ):
        head_dim = n_emb // n_heads

        if head_type == "hybrid":
            self.heads = [
                HybridHead(n_emb, head_dim, T, rng, alpha=alpha)
                for _ in range(n_heads)
            ]
        elif head_type == "rrpram":
            self.heads = [
                RRPRAMHead(n_emb, head_dim, T, rng) for _ in range(n_heads)
            ]
        else:
            self.heads = [
                ContentHead(n_emb, head_dim, T, rng) for _ in range(n_heads)
            ]

        # MLP
        self.w0 = init_weight((n_emb, nodes), rng=rng)
        self.w1 = init_weight((nodes, n_emb), rng=rng)

        # MLP deltas
        self.w0_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.w1_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Layer norm
        self.ln1_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln1_beta = np.zeros(n_emb, dtype=np.float32)
        self.ln2_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln2_beta = np.zeros(n_emb, dtype=np.float32)

        self.n_emb = n_emb
        self.head_type = head_type

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Pre-norm attention
        x_norm = layer_norm(x, self.ln1_gamma, self.ln1_beta)
        h = [head.forward(x_norm) for head in self.heads]
        h = np.concatenate(h, axis=-1)
        x = x + h

        # Pre-norm MLP with deltas
        x_norm = layer_norm(x, self.ln2_gamma, self.ln2_beta)
        
        w0 = self.w0
        w1 = self.w1
        if self.w0_delta is not None:
            A, B = self.w0_delta
            w0 = w0 + A @ B
        if self.w1_delta is not None:
            A, B = self.w1_delta
            w1 = w1 + A @ B
        
        h = x_norm @ w0
        h = gelu(h)
        h = h @ w1
        x = x + h

        return x
    
    def clear_deltas(self):
        """Remove all deltas from this block."""
        for head in self.heads:
            head.clear_deltas()
        self.w0_delta = None
        self.w1_delta = None


# ----------------- model -----------------


class StanleyTransformer:
    """
    Stanley's Transformer — a living model that grows.
    
    Forked from Haze's PostGPT with:
    - LoRA delta support at every layer
    - Two-world weights (base + personality deltas)
    - Graceful degradation (works with 0 deltas)
    """

    def __init__(
        self,
        vocab_size: int,
        T: int = 16,
        n_emb: int = 32,
        nodes: int = 32,
        n_blocks: int = 3,
        n_heads: int = 4,
        head_type: Literal["hybrid", "rrpram", "content"] = "hybrid",
        alpha: float = 0.5,
        seed: Optional[int] = 42,
    ):
        self.T = T
        self.n_emb = n_emb
        self.nodes = nodes
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.head_type = head_type
        self.alpha = alpha
        self.vocab_size = vocab_size
        self.rng = get_rng(seed)

        # Embeddings
        self.embed = init_weight((vocab_size, n_emb), rng=self.rng)
        self.pos = init_weight((T, n_emb), rng=self.rng)
        
        # Embedding deltas
        self.embed_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.pos_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Blocks
        self.blocks = [
            Block(n_emb, T, nodes, rng=self.rng, n_heads=n_heads,
                  head_type=head_type, alpha=alpha)
            for _ in range(n_blocks)
        ]

        # Final layer norm
        self.ln_f_gamma = np.ones(n_emb, dtype=np.float32)
        self.ln_f_beta = np.zeros(n_emb, dtype=np.float32)

        # Output projection
        self.w2 = init_weight((n_emb, vocab_size), rng=self.rng)
        self.w2_delta: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def logits(self, idx_seq: np.ndarray) -> np.ndarray:
        """Forward pass with delta application."""
        T = len(idx_seq)
        
        # Apply embedding deltas
        embed = self.embed
        pos = self.pos
        if self.embed_delta is not None:
            A, B = self.embed_delta
            embed = embed + A @ B
        if self.pos_delta is not None:
            A, B = self.pos_delta
            pos = pos + A @ B
        
        x = embed[idx_seq] + pos[:T]

        for block in self.blocks:
            x = block.forward(x)

        x = layer_norm(x, self.ln_f_gamma, self.ln_f_beta)
        
        # Apply output delta
        w2 = self.w2
        if self.w2_delta is not None:
            A, B = self.w2_delta
            w2 = w2 + A @ B
        
        return x @ w2

    def apply_shard_deltas(self, layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]):
        """
        Apply LoRA deltas from a shard.
        
        layer_deltas: dict of {layer_name: (A, B)} where W_eff = W + A @ B
        
        Layer naming convention:
            "embed" → embedding matrix
            "pos" → positional embedding
            "w2" → output projection
            "blocks.{i}.w0" → MLP first layer
            "blocks.{i}.w1" → MLP second layer
            "blocks.{i}.heads.{j}.wv" → value projection
            "blocks.{i}.heads.{j}.wr" → RRPRAM pattern (if RRPRAM head)
            "blocks.{i}.heads.{j}.wq" → query projection (if Content head)
            "blocks.{i}.heads.{j}.wk" → key projection (if Content head)
        """
        for name, delta in layer_deltas.items():
            if name == "embed":
                self.embed_delta = delta
            elif name == "pos":
                self.pos_delta = delta
            elif name == "w2":
                self.w2_delta = delta
            elif name.startswith("blocks."):
                parts = name.split(".")
                block_idx = int(parts[1])
                block = self.blocks[block_idx]
                
                if parts[2] == "w0":
                    block.w0_delta = delta
                elif parts[2] == "w1":
                    block.w1_delta = delta
                elif parts[2] == "heads":
                    head_idx = int(parts[3])
                    head = block.heads[head_idx]
                    weight_name = parts[4]
                    
                    if hasattr(head, weight_name + "_delta"):
                        setattr(head, weight_name + "_delta", delta)

    def clear_all_deltas(self):
        """Remove all personality deltas — return to base weights."""
        self.embed_delta = None
        self.pos_delta = None
        self.w2_delta = None
        for block in self.blocks:
            block.clear_deltas()

    def generate(
        self,
        seed_seq: List[int],
        length: int = 200,
        temperature: float = 1.0,
        sampling: Literal["basic", "top_k", "top_p", "entropy"] = "entropy",
        top_k: int = 40,
        top_p: float = 0.9,
        target_entropy: float = 3.0,
        min_temp: float = 0.3,
        max_temp: float = 2.0,
    ) -> Tuple[List[int], dict]:
        """Generate tokens with various sampling strategies."""
        T = self.T

        if not seed_seq:
            seed_seq = [0]

        seq = list(seed_seq)
        if len(seq) < T:
            pad_val = seq[0]
            seq = [pad_val] * (T - len(seq)) + seq
        else:
            seq = seq[-T:]

        seq = np.array(seq, dtype=np.int32)
        out = []

        entropies = []
        confidences = []
        temps_used = []

        for _ in range(length):
            logits = self.logits(seq)
            logits_last = logits[-1]

            probs = softmax(logits_last)
            entropies.append(entropy_bits(probs))
            confidences.append(confidence_score(logits_last))

            if sampling == "entropy":
                temp = entropy_temperature(
                    logits_last,
                    target_entropy=target_entropy,
                    min_temp=min_temp,
                    max_temp=max_temp,
                )
                temps_used.append(temp)
                nxt = sample_top_p(logits_last, top_p, temp, self.rng)

            elif sampling == "top_p":
                temps_used.append(temperature)
                nxt = sample_top_p(logits_last, top_p, temperature, self.rng)

            elif sampling == "top_k":
                temps_used.append(temperature)
                nxt = sample_top_k(logits_last, top_k, temperature, self.rng)

            else:
                temps_used.append(temperature)
                nxt = sample_basic(logits_last, temperature, self.rng)

            out.append(nxt)

            seq = np.roll(seq, -1)
            seq[-1] = nxt

        stats = {
            "mean_entropy": float(np.mean(entropies)),
            "mean_confidence": float(np.mean(confidences)),
            "mean_temp": float(np.mean(temps_used)),
            "entropy_std": float(np.std(entropies)),
        }

        return out, stats

    # ----- weight IO -----

    def save_base_weights(self, path: str | Path):
        """Save base weights (without deltas)."""
        path = Path(path)
        
        weights = {
            "T": self.T,
            "n_emb": self.n_emb,
            "nodes": self.nodes,
            "n_blocks": self.n_blocks,
            "n_heads": self.n_heads,
            "vocab_size": self.vocab_size,
            "embed": self.embed,
            "pos": self.pos,
            "w2": self.w2,
            "ln_f_gamma": self.ln_f_gamma,
            "ln_f_beta": self.ln_f_beta,
        }
        
        for b, block in enumerate(self.blocks):
            weights[f"blocks.{b}.w0"] = block.w0
            weights[f"blocks.{b}.w1"] = block.w1
            weights[f"blocks.{b}.ln1_gamma"] = block.ln1_gamma
            weights[f"blocks.{b}.ln1_beta"] = block.ln1_beta
            weights[f"blocks.{b}.ln2_gamma"] = block.ln2_gamma
            weights[f"blocks.{b}.ln2_beta"] = block.ln2_beta
            
            for h, head in enumerate(block.heads):
                if hasattr(head, 'wr'):  # RRPRAM
                    weights[f"blocks.{b}.heads.{h}.wv"] = head.wv
                    weights[f"blocks.{b}.heads.{h}.wr"] = head.wr
                elif hasattr(head, 'rrpram'):  # Hybrid
                    weights[f"blocks.{b}.heads.{h}.rrpram.wv"] = head.rrpram.wv
                    weights[f"blocks.{b}.heads.{h}.rrpram.wr"] = head.rrpram.wr
                    weights[f"blocks.{b}.heads.{h}.content.wq"] = head.content.wq
                    weights[f"blocks.{b}.heads.{h}.content.wk"] = head.content.wk
                    weights[f"blocks.{b}.heads.{h}.content.wv"] = head.content.wv
                    weights[f"blocks.{b}.heads.{h}.gate"] = head.gate
                else:  # Content
                    weights[f"blocks.{b}.heads.{h}.wq"] = head.wq
                    weights[f"blocks.{b}.heads.{h}.wk"] = head.wk
                    weights[f"blocks.{b}.heads.{h}.wv"] = head.wv
        
        np.savez_compressed(path, **weights)

    @classmethod
    def load_base_weights(cls, path: str | Path) -> "StanleyTransformer":
        """Load model from base weights file."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        
        model = cls(
            vocab_size=int(data["vocab_size"]),
            T=int(data["T"]),
            n_emb=int(data["n_emb"]),
            nodes=int(data["nodes"]),
            n_blocks=int(data["n_blocks"]),
            n_heads=int(data["n_heads"]),
            seed=None,
        )
        
        model.embed = data["embed"].astype("float32")
        model.pos = data["pos"].astype("float32")
        model.w2 = data["w2"].astype("float32")
        
        for b in range(model.n_blocks):
            block = model.blocks[b]
            block.w0 = data[f"blocks.{b}.w0"].astype("float32")
            block.w1 = data[f"blocks.{b}.w1"].astype("float32")
            
            for h in range(model.n_heads):
                head = block.heads[h]
                if hasattr(head, 'wr'):
                    head.wv = data[f"blocks.{b}.heads.{h}.wv"].astype("float32")
                    head.wr = data[f"blocks.{b}.heads.{h}.wr"].astype("float32")
        
        return model

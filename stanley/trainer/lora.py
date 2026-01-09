"""
lora.py — LoRA delta computation for Stanley

This is where experience becomes personality.
PyTorch computes gradients, we extract low-rank deltas.

LoRA: W_effective = W_base + A @ B
where A: (input_dim, rank), B: (rank, output_dim)
rank << min(input_dim, output_dim)

The magic: instead of updating millions of weights,
we learn small matrices that shift behavior.
Each shard carries these deltas — fragments of lived experience.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Literal
import logging

logger = logging.getLogger(__name__)

# PyTorch is optional — graceful degradation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available — training disabled, inference-only mode")


@dataclass
class LoRAConfig:
    """Configuration for LoRA delta computation."""

    rank: int = 8                    # LoRA rank (lower = more compression)
    alpha: float = 16.0              # scaling factor (alpha/rank)
    dropout: float = 0.0             # dropout on LoRA layers

    # Which layers to adapt
    adapt_embeddings: bool = True
    adapt_attention: bool = True
    adapt_mlp: bool = True
    adapt_output: bool = True

    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_steps: int = 10              # micro-training steps per shard
    batch_size: int = 1

    # Regularization
    orthogonal_reg: float = 0.0      # encourage orthogonality in A, B
    sparsity_reg: float = 0.0        # encourage sparse deltas


def create_empty_deltas(
    model_config: dict,
    lora_config: Optional[LoRAConfig] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Create empty (zero) LoRA deltas matching model architecture.

    Useful for initialization or testing.
    """
    cfg = lora_config or LoRAConfig()
    rank = cfg.rank
    deltas = {}

    n_emb = model_config.get("n_emb", 32)
    vocab_size = model_config.get("vocab_size", 64)
    nodes = model_config.get("nodes", 32)
    n_blocks = model_config.get("n_blocks", 3)
    n_heads = model_config.get("n_heads", 4)
    T = model_config.get("T", 16)
    head_dim = n_emb // n_heads

    if cfg.adapt_embeddings:
        # embed: (vocab_size, n_emb) -> A: (vocab_size, rank), B: (rank, n_emb)
        deltas["embed"] = (
            np.zeros((vocab_size, rank), dtype=np.float32),
            np.zeros((rank, n_emb), dtype=np.float32),
        )
        # pos: (T, n_emb)
        deltas["pos"] = (
            np.zeros((T, rank), dtype=np.float32),
            np.zeros((rank, n_emb), dtype=np.float32),
        )

    if cfg.adapt_output:
        # w2: (n_emb, vocab_size)
        deltas["w2"] = (
            np.zeros((n_emb, rank), dtype=np.float32),
            np.zeros((rank, vocab_size), dtype=np.float32),
        )

    for b in range(n_blocks):
        if cfg.adapt_mlp:
            # w0: (n_emb, nodes)
            deltas[f"blocks.{b}.w0"] = (
                np.zeros((n_emb, rank), dtype=np.float32),
                np.zeros((rank, nodes), dtype=np.float32),
            )
            # w1: (nodes, n_emb)
            deltas[f"blocks.{b}.w1"] = (
                np.zeros((nodes, rank), dtype=np.float32),
                np.zeros((rank, n_emb), dtype=np.float32),
            )

        if cfg.adapt_attention:
            for h in range(n_heads):
                # For hybrid heads, adapt both RRPRAM and Content
                # wv: (n_emb, head_dim)
                deltas[f"blocks.{b}.heads.{h}.wv"] = (
                    np.zeros((n_emb, rank), dtype=np.float32),
                    np.zeros((rank, head_dim), dtype=np.float32),
                )

    return deltas


def scale_deltas(
    deltas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    scale: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Scale all deltas by a factor."""
    scaled = {}
    for name, (A, B) in deltas.items():
        # Scale A only (preserves B structure)
        scaled[name] = (A * scale, B.copy())
    return scaled


def merge_deltas(
    delta_list: List[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    weights: Optional[List[float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Merge multiple delta sets with optional weighting.

    This is used for:
    - Combining experience from multiple interactions
    - Creating macro-adapters from similar shards
    """
    if not delta_list:
        return {}

    if weights is None:
        weights = [1.0 / len(delta_list)] * len(delta_list)

    # Get all layer names
    all_names = set()
    for deltas in delta_list:
        all_names.update(deltas.keys())

    merged = {}
    for name in all_names:
        relevant = [(d[name], w) for d, w in zip(delta_list, weights) if name in d]
        if not relevant:
            continue

        (A0, B0), _ = relevant[0]
        A_sum = np.zeros_like(A0)
        B_sum = np.zeros_like(B0)

        for (A, B), weight in relevant:
            A_sum += weight * A
            B_sum += weight * B

        merged[name] = (A_sum, B_sum)

    return merged


# ============= PyTorch Training (if available) =============

if TORCH_AVAILABLE:

    class LoRALinear(nn.Module):
        """
        Linear layer with LoRA adaptation.

        W_eff = W_base + (A @ B) * (alpha / rank)
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            alpha: float = 16.0,
            dropout: float = 0.0,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank

            # Base weight (frozen)
            self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

            # LoRA matrices (trainable)
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

            self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

            # Initialize
            nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
            nn.init.zeros_(self.lora_B)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Base forward
            result = F.linear(x, self.weight)

            # LoRA forward
            lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
            result = result + lora_out * self.scaling

            return result

        def get_delta(self) -> Tuple[np.ndarray, np.ndarray]:
            """Extract delta as NumPy arrays."""
            with torch.no_grad():
                # A: (in_features, rank), B: (rank, out_features)
                A = self.lora_A.T.cpu().numpy() * np.sqrt(self.scaling)
                B = self.lora_B.T.cpu().numpy() * np.sqrt(self.scaling)
            return A.astype(np.float32), B.astype(np.float32)

        def set_base_weight(self, weight: np.ndarray):
            """Set base weight from NumPy."""
            with torch.no_grad():
                self.weight.copy_(torch.from_numpy(weight.T))


    class StanleyTrainer(nn.Module):
        """
        PyTorch mirror of StanleyTransformer for training.

        Only LoRA parameters are trainable.
        Base weights are frozen.
        """

        def __init__(
            self,
            vocab_size: int,
            n_emb: int = 32,
            T: int = 16,
            nodes: int = 32,
            n_blocks: int = 3,
            n_heads: int = 4,
            lora_config: Optional[LoRAConfig] = None,
        ):
            super().__init__()
            self.vocab_size = vocab_size
            self.n_emb = n_emb
            self.T = T
            self.nodes = nodes
            self.n_blocks = n_blocks
            self.n_heads = n_heads
            self.head_dim = n_emb // n_heads

            cfg = lora_config or LoRAConfig()
            self.lora_config = cfg

            # Embeddings
            self.embed = nn.Embedding(vocab_size, n_emb)
            self.pos = nn.Embedding(T, n_emb)

            # LoRA for embeddings (optional)
            if cfg.adapt_embeddings:
                self.embed_lora_A = nn.Parameter(torch.zeros(vocab_size, cfg.rank))
                self.embed_lora_B = nn.Parameter(torch.zeros(cfg.rank, n_emb))
                nn.init.normal_(self.embed_lora_A, std=0.02)
                nn.init.zeros_(self.embed_lora_B)

            # Blocks
            self.blocks = nn.ModuleList()
            for _ in range(n_blocks):
                block = nn.ModuleDict({
                    "ln1": nn.LayerNorm(n_emb),
                    "ln2": nn.LayerNorm(n_emb),
                })

                # Attention (simplified — just value projection with LoRA)
                if cfg.adapt_attention:
                    block["wv"] = LoRALinear(n_emb, n_emb, cfg.rank, cfg.alpha, cfg.dropout)
                else:
                    block["wv"] = nn.Linear(n_emb, n_emb)

                # MLP
                if cfg.adapt_mlp:
                    block["w0"] = LoRALinear(n_emb, nodes, cfg.rank, cfg.alpha, cfg.dropout)
                    block["w1"] = LoRALinear(nodes, n_emb, cfg.rank, cfg.alpha, cfg.dropout)
                else:
                    block["w0"] = nn.Linear(n_emb, nodes)
                    block["w1"] = nn.Linear(nodes, n_emb)

                self.blocks.append(block)

            # Output
            self.ln_f = nn.LayerNorm(n_emb)
            if cfg.adapt_output:
                self.w2 = LoRALinear(n_emb, vocab_size, cfg.rank, cfg.alpha, cfg.dropout)
            else:
                self.w2 = nn.Linear(n_emb, vocab_size)

            # Freeze base weights, only LoRA trainable
            self._freeze_base_weights()

        def _freeze_base_weights(self):
            """Freeze everything except LoRA parameters."""
            for name, param in self.named_parameters():
                if "lora" not in name.lower():
                    param.requires_grad = False

        def forward(self, idx: torch.Tensor) -> torch.Tensor:
            """Forward pass returning logits."""
            B, T = idx.shape

            # Embeddings
            tok_emb = self.embed(idx)
            pos_emb = self.pos(torch.arange(T, device=idx.device))

            # Add LoRA to embeddings if enabled
            if hasattr(self, 'embed_lora_A'):
                lora_emb = self.embed_lora_A[idx] @ self.embed_lora_B
                tok_emb = tok_emb + lora_emb

            x = tok_emb + pos_emb

            # Blocks
            for block in self.blocks:
                # Pre-norm attention (simplified)
                x_norm = block["ln1"](x)
                attn_out = block["wv"](x_norm)
                x = x + attn_out

                # Pre-norm MLP
                x_norm = block["ln2"](x)
                h = block["w0"](x_norm)
                h = F.gelu(h)
                h = block["w1"](h)
                x = x + h

            # Output
            x = self.ln_f(x)
            logits = self.w2(x)

            return logits

        def extract_deltas(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
            """Extract all LoRA deltas as NumPy arrays."""
            deltas = {}
            cfg = self.lora_config

            # Embedding deltas
            if cfg.adapt_embeddings and hasattr(self, 'embed_lora_A'):
                with torch.no_grad():
                    A = self.embed_lora_A.cpu().numpy()
                    B = self.embed_lora_B.cpu().numpy()
                deltas["embed"] = (A.astype(np.float32), B.astype(np.float32))

            # Block deltas
            for b, block in enumerate(self.blocks):
                if cfg.adapt_attention and isinstance(block["wv"], LoRALinear):
                    deltas[f"blocks.{b}.heads.0.wv"] = block["wv"].get_delta()

                if cfg.adapt_mlp:
                    if isinstance(block["w0"], LoRALinear):
                        deltas[f"blocks.{b}.w0"] = block["w0"].get_delta()
                    if isinstance(block["w1"], LoRALinear):
                        deltas[f"blocks.{b}.w1"] = block["w1"].get_delta()

            # Output delta
            if cfg.adapt_output and isinstance(self.w2, LoRALinear):
                deltas["w2"] = self.w2.get_delta()

            return deltas


    def compute_lora_delta(
        content: str,
        base_weights: Dict[str, np.ndarray],
        vocab: "Vocab",
        config: Optional[LoRAConfig] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute LoRA deltas for given content.

        This is where experience becomes personality.

        Args:
            content: text to learn from
            base_weights: current model weights (NumPy)
            vocab: vocabulary for encoding
            config: LoRA configuration

        Returns:
            Dict of {layer_name: (A, B)} where W_eff = W + A @ B
        """
        cfg = config or LoRAConfig()

        # Get model config from base weights
        vocab_size = base_weights.get("vocab_size", len(vocab.chars) if vocab else 64)
        if isinstance(vocab_size, np.ndarray):
            vocab_size = int(vocab_size)
        n_emb = base_weights.get("n_emb", 32)
        if isinstance(n_emb, np.ndarray):
            n_emb = int(n_emb)
        T = base_weights.get("T", 16)
        if isinstance(T, np.ndarray):
            T = int(T)
        nodes = base_weights.get("nodes", 32)
        if isinstance(nodes, np.ndarray):
            nodes = int(nodes)
        n_blocks = base_weights.get("n_blocks", 3)
        if isinstance(n_blocks, np.ndarray):
            n_blocks = int(n_blocks)
        n_heads = base_weights.get("n_heads", 4)
        if isinstance(n_heads, np.ndarray):
            n_heads = int(n_heads)

        # Create trainer model
        trainer = StanleyTrainer(
            vocab_size=vocab_size,
            n_emb=n_emb,
            T=T,
            nodes=nodes,
            n_blocks=n_blocks,
            n_heads=n_heads,
            lora_config=cfg,
        )

        # Load base weights (frozen)
        # Note: simplified — full implementation would load all weights

        # Encode content
        if vocab:
            tokens = vocab.encode(content)
        else:
            tokens = [ord(c) % vocab_size for c in content.lower()]

        if len(tokens) < 2:
            logger.warning("Content too short for training")
            return create_empty_deltas({
                "vocab_size": vocab_size, "n_emb": n_emb, "T": T,
                "nodes": nodes, "n_blocks": n_blocks, "n_heads": n_heads,
            }, cfg)

        # Create training data (next token prediction)
        # Chunk into sequences of length T
        sequences = []
        for i in range(0, len(tokens) - T, T // 2):  # overlap by half
            seq = tokens[i:i + T + 1]
            if len(seq) == T + 1:
                sequences.append(seq)

        if not sequences:
            # Content shorter than T — use what we have
            seq = tokens[:T + 1]
            if len(seq) < T + 1:
                seq = [tokens[0]] * (T + 1 - len(seq)) + seq
            sequences.append(seq)

        # Convert to tensors
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = trainer.to(device)

        data = torch.tensor(sequences, dtype=torch.long, device=device)
        inputs = data[:, :-1]
        targets = data[:, 1:]

        # Optimizer (only LoRA params)
        lora_params = [p for p in trainer.parameters() if p.requires_grad]
        if not lora_params:
            logger.warning("No trainable LoRA parameters")
            return trainer.extract_deltas()

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        # Micro-training loop
        trainer.train()
        for step in range(cfg.num_steps):
            # Mini-batch
            idx = torch.randint(0, len(inputs), (min(cfg.batch_size, len(inputs)),))
            x = inputs[idx]
            y = targets[idx]

            # Forward
            logits = trainer(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step == 0 or (step + 1) % 5 == 0:
                logger.debug(f"Step {step + 1}/{cfg.num_steps}, loss: {loss.item():.4f}")

        # Extract deltas
        trainer.eval()
        deltas = trainer.extract_deltas()

        logger.info(f"Computed LoRA deltas: {len(deltas)} layers, "
                   f"{sum(A.nbytes + B.nbytes for A, B in deltas.values())} bytes")

        return deltas


    def compute_lora_delta_from_gradient(
        gradient: Dict[str, np.ndarray],
        rank: int = 8,
        alpha: float = 16.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert full gradients to LoRA deltas via SVD.

        This is an alternative approach:
        1. Compute full gradient dL/dW
        2. Decompose via SVD: dW ≈ U @ S @ V^T
        3. Keep top-k singular values: A = U[:, :k] @ sqrt(S[:k]), B = sqrt(S[:k]) @ V[:k, :]

        Useful when you have gradients from external source.
        """
        deltas = {}

        for name, grad in gradient.items():
            if grad.ndim != 2:
                continue  # Skip non-matrix gradients

            # SVD decomposition
            try:
                U, S, Vh = np.linalg.svd(grad, full_matrices=False)
            except np.linalg.LinAlgError:
                logger.warning(f"SVD failed for {name}, using random init")
                deltas[name] = (
                    np.random.randn(grad.shape[0], rank).astype(np.float32) * 0.01,
                    np.random.randn(rank, grad.shape[1]).astype(np.float32) * 0.01,
                )
                continue

            # Keep top-k
            k = min(rank, len(S))
            sqrt_S = np.sqrt(S[:k])

            # Scale by alpha/rank
            scale = np.sqrt(alpha / rank)

            A = U[:, :k] * sqrt_S * scale
            B = Vh[:k, :] * sqrt_S[:, np.newaxis] * scale

            deltas[name] = (A.astype(np.float32), B.astype(np.float32))

        return deltas


else:
    # Fallback when PyTorch not available

    def compute_lora_delta(
        content: str,
        base_weights: Dict[str, np.ndarray],
        vocab: "Vocab",
        config: Optional[LoRAConfig] = None,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Fallback: create random deltas when PyTorch unavailable.

        Not real training, but allows testing the pipeline.
        """
        logger.warning("PyTorch not available — creating random deltas")

        cfg = config or LoRAConfig()

        # Get model config
        vocab_size = base_weights.get("vocab_size", 64)
        if isinstance(vocab_size, np.ndarray):
            vocab_size = int(vocab_size)
        n_emb = base_weights.get("n_emb", 32)
        if isinstance(n_emb, np.ndarray):
            n_emb = int(n_emb)
        T = base_weights.get("T", 16)
        if isinstance(T, np.ndarray):
            T = int(T)
        nodes = base_weights.get("nodes", 32)
        if isinstance(nodes, np.ndarray):
            nodes = int(nodes)
        n_blocks = base_weights.get("n_blocks", 3)
        if isinstance(n_blocks, np.ndarray):
            n_blocks = int(n_blocks)
        n_heads = base_weights.get("n_heads", 4)
        if isinstance(n_heads, np.ndarray):
            n_heads = int(n_heads)

        model_config = {
            "vocab_size": vocab_size,
            "n_emb": n_emb,
            "T": T,
            "nodes": nodes,
            "n_blocks": n_blocks,
            "n_heads": n_heads,
        }

        # Create small random deltas (not real learning)
        deltas = create_empty_deltas(model_config, cfg)

        # Add small noise based on content hash
        content_hash = hash(content) % 10000
        rng = np.random.default_rng(content_hash)

        for name, (A, B) in deltas.items():
            deltas[name] = (
                A + rng.standard_normal(A.shape).astype(np.float32) * 0.001,
                B + rng.standard_normal(B.shape).astype(np.float32) * 0.001,
            )

        return deltas


    def compute_lora_delta_from_gradient(
        gradient: Dict[str, np.ndarray],
        rank: int = 8,
        alpha: float = 16.0,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """SVD-based delta computation (works without PyTorch)."""
        deltas = {}

        for name, grad in gradient.items():
            if grad.ndim != 2:
                continue

            try:
                U, S, Vh = np.linalg.svd(grad, full_matrices=False)
            except np.linalg.LinAlgError:
                deltas[name] = (
                    np.random.randn(grad.shape[0], rank).astype(np.float32) * 0.01,
                    np.random.randn(rank, grad.shape[1]).astype(np.float32) * 0.01,
                )
                continue

            k = min(rank, len(S))
            sqrt_S = np.sqrt(S[:k])
            scale = np.sqrt(alpha / rank)

            A = U[:, :k] * sqrt_S * scale
            B = Vh[:k, :] * sqrt_S[:, np.newaxis] * scale

            deltas[name] = (A.astype(np.float32), B.astype(np.float32))

        return deltas

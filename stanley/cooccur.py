"""
cooccur.py — Co-occurrence Field for Stanley

Adapted from Haze's cooccur.py (inspired by Leo's trigram graphs).

Key insight: "Words that resonate together, stay together."

Stanley's INNOVATION: The field LEARNS from shards!
- Not just static corpus analysis
- Updates incrementally as Stanley experiences
- Shards with high resonance have more influence
- This is emergence through lived experience

Usage:
    from stanley.cooccur import CooccurField

    # Build from origin text
    field = CooccurField.from_text(origin_text, vocab)

    # Update from shard (self-training!)
    field.observe_shard(shard, vocab)

    # Bias generation
    biased_logits = field.bias_logits(logits, context)
"""

from __future__ import annotations
import numpy as np
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .shard import Shard
    from .inference import Vocab

logger = logging.getLogger(__name__)


@dataclass
class CooccurConfig:
    """Configuration for co-occurrence field."""
    window_size: int = 5        # context window for co-occurrence
    smoothing: float = 0.1      # Laplace smoothing
    trigram_weight: float = 0.6  # weight for trigram in blend
    cooccur_weight: float = 0.4  # weight for co-occurrence in blend


class CooccurField:
    """
    Co-occurrence field built from corpus.

    This is PURE CORPUS STATISTICS. No neural network.
    Enables coherent generation without trained weights.

    Think of it as a "resonance field" — the statistical
    structure of the origin text that guides generation.
    """

    def __init__(
        self,
        vocab_size: int,
        config: Optional[CooccurConfig] = None,
    ):
        self.vocab_size = vocab_size
        self.config = config or CooccurConfig()

        # Statistics
        self.bigram_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.trigram_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.cooccur_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Total counts for normalization
        self.bigram_totals: Dict[int, int] = defaultdict(int)
        self.trigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)
        self.cooccur_totals: Dict[int, int] = defaultdict(int)

        # Token frequencies
        self.token_counts: Dict[int, int] = defaultdict(int)
        self.total_tokens: int = 0

        # Stanley additions for self-training
        self.shard_contributions: int = 0  # how many shards have contributed
        self.resonance_weight_sum: float = 0.0  # total resonance from shards

    @classmethod
    def from_text(
        cls,
        text: str,
        vocab: "Vocab",
        config: Optional[CooccurConfig] = None,
    ) -> "CooccurField":
        """Build field from text corpus."""
        cfg = config or CooccurConfig()
        field = cls(vocab.vocab_size, cfg)

        # Encode text
        tokens = vocab.encode(text)

        if len(tokens) < 3:
            logger.warning("Text too short to build co-occurrence field")
            return field

        field.total_tokens = len(tokens)

        # Count unigrams
        for t in tokens:
            field.token_counts[t] += 1

        # Count bigrams
        for i in range(len(tokens) - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            field.bigram_counts[t1][t2] += 1
            field.bigram_totals[t1] += 1

        # Count trigrams
        for i in range(len(tokens) - 2):
            t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
            field.trigram_counts[(t1, t2)][t3] += 1
            field.trigram_totals[(t1, t2)] += 1

        # Count co-occurrences (within window)
        window = cfg.window_size
        for i in range(len(tokens)):
            center = tokens[i]
            for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
                if i != j:
                    neighbor = tokens[j]
                    field.cooccur_counts[center][neighbor] += 1
                    field.cooccur_totals[center] += 1

        logger.info(f"CooccurField built: {len(tokens)} tokens, "
                   f"{len(field.bigram_counts)} bigram contexts, "
                   f"{len(field.trigram_counts)} trigram contexts")

        return field

    def get_bigram_probs(self, current: int) -> np.ndarray:
        """Get probability distribution P(next | current)."""
        probs = np.ones(self.vocab_size, dtype=np.float32) * self.config.smoothing

        if current in self.bigram_counts:
            total = self.bigram_totals[current] + self.vocab_size * self.config.smoothing
            for next_token, count in self.bigram_counts[current].items():
                probs[next_token] = (count + self.config.smoothing) / total

        # Normalize
        probs /= probs.sum()
        return probs

    def get_trigram_probs(self, prev: int, current: int) -> np.ndarray:
        """Get probability distribution P(next | prev, current)."""
        probs = np.ones(self.vocab_size, dtype=np.float32) * self.config.smoothing

        key = (prev, current)
        if key in self.trigram_counts:
            total = self.trigram_totals[key] + self.vocab_size * self.config.smoothing
            for next_token, count in self.trigram_counts[key].items():
                probs[next_token] = (count + self.config.smoothing) / total

        # Normalize
        probs /= probs.sum()
        return probs

    def get_cooccur_probs(self, context: List[int]) -> np.ndarray:
        """Get probability based on co-occurrence with context."""
        probs = np.ones(self.vocab_size, dtype=np.float32) * self.config.smoothing

        for token in context[-self.config.window_size:]:
            if token in self.cooccur_counts:
                for neighbor, count in self.cooccur_counts[token].items():
                    probs[neighbor] += count

        # Normalize
        probs /= probs.sum()
        return probs

    def get_blend_probs(self, context: List[int]) -> np.ndarray:
        """
        Get blended probability distribution.

        Combines trigram (60%) and co-occurrence (40%) by default.
        Falls back to bigram if no trigram context.
        """
        cfg = self.config

        if len(context) >= 2:
            # Have trigram context
            trigram_probs = self.get_trigram_probs(context[-2], context[-1])
            cooccur_probs = self.get_cooccur_probs(context)

            probs = cfg.trigram_weight * trigram_probs + cfg.cooccur_weight * cooccur_probs

        elif len(context) >= 1:
            # Only bigram
            bigram_probs = self.get_bigram_probs(context[-1])
            cooccur_probs = self.get_cooccur_probs(context)

            probs = 0.5 * bigram_probs + 0.5 * cooccur_probs

        else:
            # No context — uniform
            probs = np.ones(self.vocab_size, dtype=np.float32) / self.vocab_size

        # Normalize
        probs /= probs.sum()
        return probs

    def bias_logits(
        self,
        logits: np.ndarray,
        context: List[int],
        alpha: float = 0.3,
    ) -> np.ndarray:
        """
        Bias model logits with corpus statistics.

        output = (1 - alpha) * logits + alpha * corpus_log_probs

        This is the key to coherent generation:
        even random weights produce coherent text when
        guided by corpus statistics.
        """
        # Get corpus probabilities
        corpus_probs = self.get_blend_probs(context)
        corpus_log_probs = np.log(corpus_probs + 1e-10)

        # Blend with model logits
        biased = (1 - alpha) * logits + alpha * corpus_log_probs * 10  # scale log probs

        return biased

    def generate_pure(
        self,
        seed: List[int],
        length: int = 100,
        temperature: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[int]:
        """
        Generate purely from corpus statistics.

        No neural network at all — just trigram/cooccurrence chains.
        This is PURE RESONANCE.
        """
        rng = rng or np.random.default_rng()

        seq = list(seed) if seed else [0]

        for _ in range(length):
            probs = self.get_blend_probs(seq)

            # Temperature
            if temperature != 1.0:
                log_probs = np.log(probs + 1e-10) / temperature
                probs = np.exp(log_probs)
                probs /= probs.sum()

            # Sample
            next_token = rng.choice(len(probs), p=probs)
            seq.append(int(next_token))

        return seq[len(seed):]

    def stats(self) -> dict:
        """Get field statistics."""
        return {
            "vocab_size": self.vocab_size,
            "total_tokens": self.total_tokens,
            "unique_bigrams": len(self.bigram_counts),
            "unique_trigrams": len(self.trigram_counts),
            "unique_tokens": len(self.token_counts),
            "shard_contributions": self.shard_contributions,
            "avg_resonance": (
                self.resonance_weight_sum / self.shard_contributions
                if self.shard_contributions > 0 else 0.0
            ),
        }

    def __repr__(self) -> str:
        return (f"CooccurField(vocab={self.vocab_size}, "
               f"tokens={self.total_tokens}, "
               f"trigrams={len(self.trigram_counts)})")

    # ========================================================================
    # STANLEY SELF-TRAINING METHODS
    # ========================================================================

    def observe_shard(
        self,
        shard: "Shard",
        vocab: "Vocab",
        resonance_boost: bool = True,
    ) -> int:
        """
        Update field from a shard — THIS IS HOW STANLEY LEARNS!

        Shards with higher resonance contribute more strongly.
        This is emergence through lived experience.

        Args:
            shard: The shard to learn from
            vocab: Vocabulary for encoding
            resonance_boost: If True, weight by shard's resonance score

        Returns:
            Number of new patterns absorbed
        """
        text = shard.content
        if not text or len(text) < 3:
            return 0

        # Resonance multiplier (high resonance = more influence)
        weight = shard.resonance_score if resonance_boost else 1.0
        weight = max(0.1, min(2.0, weight))  # clamp to reasonable range

        new_patterns = self.observe_text(text, vocab, weight)

        self.shard_contributions += 1
        self.resonance_weight_sum += weight

        return new_patterns

    def observe_text(
        self,
        text: str,
        vocab: "Vocab",
        weight: float = 1.0,
    ) -> int:
        """
        Update from raw text (for lexicon absorption, dreams, etc.)

        Args:
            text: Text to learn from
            vocab: Vocabulary for encoding
            weight: How strongly to weight this observation

        Returns:
            Number of new patterns absorbed
        """
        if not text or len(text) < 3:
            return 0

        tokens = vocab.encode(text)
        n = len(tokens)
        if n < 3:
            return 0

        new_patterns = 0
        window = self.config.window_size

        # Update token counts with weight
        for t in tokens:
            self.token_counts[t] += weight

        # Update bigrams
        for i in range(n - 1):
            t1, t2 = tokens[i], tokens[i + 1]
            if self.bigram_counts[t1][t2] == 0:
                new_patterns += 1
            self.bigram_counts[t1][t2] += weight
            self.bigram_totals[t1] += weight

        # Update trigrams
        for i in range(n - 2):
            t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
            key = (t1, t2)
            if self.trigram_counts[key][t3] == 0:
                new_patterns += 1
            self.trigram_counts[key][t3] += weight
            self.trigram_totals[key] += weight

        # Update co-occurrences
        for i in range(n):
            center = tokens[i]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i != j:
                    neighbor = tokens[j]
                    self.cooccur_counts[center][neighbor] += weight
                    self.cooccur_totals[center] += weight

        self.total_tokens += n

        return new_patterns

    def resonance_between(self, token_a: int, token_b: int) -> float:
        """
        Compute resonance between two tokens.

        High resonance = they often appear together.
        Useful for measuring semantic affinity.
        """
        if token_a not in self.cooccur_counts:
            return 0.0

        total = self.cooccur_totals.get(token_a, 0)
        if total == 0:
            return 0.0

        count = self.cooccur_counts[token_a].get(token_b, 0)
        return count / total

    def top_resonant(self, token: int, k: int = 10) -> List[Tuple[int, float]]:
        """Get top k most resonant tokens for a given token."""
        if token not in self.cooccur_counts:
            return []

        total = self.cooccur_totals.get(token, 0)
        if total == 0:
            return []

        resonances = [
            (t, c / total)
            for t, c in self.cooccur_counts[token].items()
        ]
        resonances.sort(key=lambda x: x[1], reverse=True)

        return resonances[:k]

    def save(self, path: str) -> None:
        """Save field to file."""
        data = {
            "vocab_size": self.vocab_size,
            "config": {
                "window_size": self.config.window_size,
                "smoothing": self.config.smoothing,
                "trigram_weight": self.config.trigram_weight,
                "cooccur_weight": self.config.cooccur_weight,
            },
            "bigram_counts": dict(self.bigram_counts),
            "trigram_counts": {str(k): dict(v) for k, v in self.trigram_counts.items()},
            "cooccur_counts": dict(self.cooccur_counts),
            "bigram_totals": dict(self.bigram_totals),
            "trigram_totals": {str(k): v for k, v in self.trigram_totals.items()},
            "cooccur_totals": dict(self.cooccur_totals),
            "token_counts": dict(self.token_counts),
            "total_tokens": self.total_tokens,
            "shard_contributions": self.shard_contributions,
            "resonance_weight_sum": self.resonance_weight_sum,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"CooccurField saved to {path}")

    @classmethod
    def load(cls, path: str) -> "CooccurField":
        """Load field from file."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        cfg = CooccurConfig(**data.get("config", {}))
        field = cls(data["vocab_size"], cfg)

        # Restore bigrams
        for k, v in data.get("bigram_counts", {}).items():
            field.bigram_counts[k] = defaultdict(int, v)
        field.bigram_totals = defaultdict(int, data.get("bigram_totals", {}))

        # Restore trigrams (convert string keys back to tuples)
        for k, v in data.get("trigram_counts", {}).items():
            key = eval(k)  # "(1, 2)" -> (1, 2)
            field.trigram_counts[key] = defaultdict(int, v)
        for k, v in data.get("trigram_totals", {}).items():
            key = eval(k)
            field.trigram_totals[key] = v

        # Restore cooccur
        for k, v in data.get("cooccur_counts", {}).items():
            field.cooccur_counts[k] = defaultdict(int, v)
        field.cooccur_totals = defaultdict(int, data.get("cooccur_totals", {}))

        # Restore token counts
        field.token_counts = defaultdict(int, data.get("token_counts", {}))
        field.total_tokens = data.get("total_tokens", 0)

        # Restore Stanley additions
        field.shard_contributions = data.get("shard_contributions", 0)
        field.resonance_weight_sum = data.get("resonance_weight_sum", 0.0)

        logger.info(f"CooccurField loaded from {path}: {field.shard_contributions} shard contributions")
        return field
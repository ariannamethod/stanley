"""
cooccur.py — Co-occurrence Field for Stanley

This is the key to coherent untrained generation.
Like Haze, Stanley can speak coherently even without trained weights
by using corpus statistics:

1. Trigram: P(next | prev, current) — main signal
2. Bigram: P(next | current) — fallback
3. Co-occur window: what tokens appear near each other

This creates a "resonance field" from the origin text
that guides generation toward coherent patterns.
"""

from __future__ import annotations
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import logging

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
        }

    def __repr__(self) -> str:
        return (f"CooccurField(vocab={self.vocab_size}, "
               f"tokens={self.total_tokens}, "
               f"trigrams={len(self.trigram_counts)})")

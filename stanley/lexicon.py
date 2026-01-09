#!/usr/bin/env python3
"""
lexicon.py — Dynamic Lexicon Growth for Stanley

Adapted from Haze's lexicon.py (inspired by Leo's cloud morphing).

The key insight: Stanley EVOLVES through conversation!
  1. User speaks → new words/trigrams absorbed
  2. Absorbed patterns injected into CooccurField
  3. Next generation uses absorbed patterns
  4. Stanley learns YOUR vocabulary

This is LIVE EVOLUTION — the field morphs as you talk!

Usage:
    from stanley.lexicon import Lexicon

    lex = Lexicon(vocab, cooccur_field)
    absorbed = lex.absorb(user_text)
    print(f"Absorbed {absorbed.count} new patterns!")
"""

from __future__ import annotations
import re
import time
from typing import List, Tuple, Optional, Dict, Set, TYPE_CHECKING
from collections import Counter
from dataclasses import dataclass, field
import logging

if TYPE_CHECKING:
    from .inference import Vocab
    from .cooccur import CooccurField

logger = logging.getLogger(__name__)


@dataclass
class AbsorptionRecord:
    """Record of what was absorbed from an interaction."""
    timestamp: float
    source: str  # "user" or "self" or "dream"
    words: List[str] = field(default_factory=list)
    trigrams: List[Tuple[str, str, str]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.words) + len(self.trigrams)


@dataclass
class LexiconStats:
    """Statistics about the dynamic lexicon."""
    total_words: int = 0
    total_trigrams: int = 0
    unique_sources: int = 0
    recent_absorptions: int = 0
    growth_rate: float = 0.0  # words per interaction

    def __repr__(self) -> str:
        return (f"LexiconStats(words={self.total_words}, "
                f"trigrams={self.total_trigrams}, "
                f"growth={self.growth_rate:.2f}/turn)")


class Lexicon:
    """
    Dynamic lexicon that grows through conversation.

    Key features:
    - Absorbs new words and trigrams from user input
    - Injects patterns into CooccurField
    - Tracks absorption history for analysis
    - Decays old patterns (memory decay)

    This is LIVE EVOLUTION — Stanley morphs as you talk!
    """

    def __init__(
        self,
        vocab: "Vocab",
        cooccur_field: Optional["CooccurField"] = None,
        decay_rate: float = 0.99,
        min_word_length: int = 3,
    ):
        """
        Initialize dynamic lexicon.

        Args:
            vocab: Vocabulary for encoding
            cooccur_field: Field to inject patterns into
            decay_rate: How fast old patterns decay (0.99 = slow)
            min_word_length: Minimum word length to absorb
        """
        self.vocab = vocab
        self.field = cooccur_field
        self.decay_rate = decay_rate
        self.min_word_length = min_word_length

        # Absorbed content
        self.absorbed_words: Set[str] = set()
        self.absorbed_trigrams: Set[Tuple[str, str, str]] = set()

        # Word weights (for decay and resonance)
        self.word_weights: Dict[str, float] = {}

        # History
        self.history: List[AbsorptionRecord] = []
        self.max_history: int = 100

        # Corpus words (to detect novelty)
        self._build_corpus_vocabulary()

    def _build_corpus_vocabulary(self) -> None:
        """Extract vocabulary from corpus via vocab."""
        self.corpus_words: Set[str] = set()

        # Get all characters/tokens in vocabulary
        for token_id in range(self.vocab.vocab_size):
            try:
                char = self.vocab.decode([token_id])
                if char:
                    self.corpus_words.add(char.lower())
            except Exception:
                pass

    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        return [w for w in words if len(w) >= self.min_word_length]

    def _extract_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract trigrams from text."""
        words = re.findall(r'\b\w+\b', text.lower())
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i + 1], words[i + 2]))
        return trigrams

    def absorb(
        self,
        text: str,
        source: str = "user",
        boost: float = 1.0,
    ) -> AbsorptionRecord:
        """
        Absorb new patterns from text.

        This is how Stanley LEARNS from conversation!

        Args:
            text: Text to absorb patterns from
            source: Origin of text ("user", "self", "dream")
            boost: Weight multiplier for these patterns

        Returns:
            Record of what was absorbed
        """
        # Extract patterns
        words = self._extract_words(text)
        trigrams = self._extract_trigrams(text)

        new_words = []
        new_trigrams = []

        # Absorb new words
        for word in words:
            if word not in self.absorbed_words:
                self.absorbed_words.add(word)
                self.word_weights[word] = boost
                new_words.append(word)
            else:
                # Reinforce existing word
                self.word_weights[word] = min(2.0, self.word_weights.get(word, 1.0) + 0.1)

        # Absorb new trigrams
        for tri in trigrams:
            if tri not in self.absorbed_trigrams:
                self.absorbed_trigrams.add(tri)
                new_trigrams.append(tri)
                # Inject into field if available
                if self.field:
                    self._inject_trigram(tri, boost)

        # Create record
        record = AbsorptionRecord(
            timestamp=time.time(),
            source=source,
            words=new_words,
            trigrams=new_trigrams,
        )

        # Store in history
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        if new_words or new_trigrams:
            logger.debug(f"Lexicon absorbed: {len(new_words)} words, {len(new_trigrams)} trigrams from {source}")

        return record

    def _inject_trigram(
        self,
        trigram: Tuple[str, str, str],
        weight: float = 1.0,
    ) -> None:
        """
        Inject a trigram into the co-occurrence field.

        This modifies the field so future generation can use
        patterns from user input — THIS IS LIVE EVOLUTION!
        """
        if not self.field:
            return

        # Encode each word to tokens
        w1_tokens = self.vocab.encode(trigram[0])
        w2_tokens = self.vocab.encode(trigram[1])
        w3_tokens = self.vocab.encode(trigram[2])

        if not w1_tokens or not w2_tokens or not w3_tokens:
            return

        # Get boundary tokens
        last_w1 = w1_tokens[-1]
        first_w2 = w2_tokens[0]
        last_w2 = w2_tokens[-1]
        first_w3 = w3_tokens[0]

        # Inject into bigram counts
        if last_w1 not in self.field.bigram_counts:
            from collections import defaultdict
            self.field.bigram_counts[last_w1] = defaultdict(int)
        self.field.bigram_counts[last_w1][first_w2] += weight
        self.field.bigram_totals[last_w1] += weight

        if last_w2 not in self.field.bigram_counts:
            from collections import defaultdict
            self.field.bigram_counts[last_w2] = defaultdict(int)
        self.field.bigram_counts[last_w2][first_w3] += weight
        self.field.bigram_totals[last_w2] += weight

        # Update trigram counts
        key = (last_w1, first_w2)
        if key not in self.field.trigram_counts:
            from collections import defaultdict
            self.field.trigram_counts[key] = defaultdict(int)
        self.field.trigram_counts[key][last_w2] += weight
        self.field.trigram_totals[key] += weight

    def decay(self) -> int:
        """
        Apply memory decay to absorbed patterns.

        Old patterns fade, recent patterns stay strong.
        This prevents infinite accumulation.

        Returns:
            Number of patterns that decayed below threshold
        """
        decayed = 0

        # Decay word weights
        words_to_remove = []
        for word, weight in self.word_weights.items():
            new_weight = weight * self.decay_rate
            if new_weight < 0.1:
                words_to_remove.append(word)
                decayed += 1
            else:
                self.word_weights[word] = new_weight

        # Remove decayed words
        for word in words_to_remove:
            self.absorbed_words.discard(word)
            del self.word_weights[word]

        return decayed

    def get_resonant_words(self, n: int = 20) -> List[str]:
        """
        Get most resonant (high-weight) absorbed words.

        These are words that have been reinforced through conversation.
        """
        sorted_words = sorted(
            self.word_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [w for w, _ in sorted_words[:n]]

    def get_novel_words(self) -> List[str]:
        """
        Get words that were absorbed but aren't in original corpus.

        These are truly NEW words that Stanley learned!
        """
        return [
            w for w in self.absorbed_words
            if w not in self.corpus_words
        ]

    def stats(self) -> LexiconStats:
        """Get lexicon statistics."""
        # Count unique sources
        sources = set(r.source for r in self.history)

        # Calculate growth rate
        if len(self.history) >= 2:
            recent = self.history[-10:]
            total_absorbed = sum(r.count for r in recent)
            growth_rate = total_absorbed / len(recent)
        else:
            growth_rate = 0.0

        return LexiconStats(
            total_words=len(self.absorbed_words),
            total_trigrams=len(self.absorbed_trigrams),
            unique_sources=len(sources),
            recent_absorptions=sum(r.count for r in self.history[-10:]),
            growth_rate=growth_rate,
        )

    def save(self, path: str) -> None:
        """Save lexicon state to file."""
        import pickle
        data = {
            "absorbed_words": list(self.absorbed_words),
            "absorbed_trigrams": list(self.absorbed_trigrams),
            "word_weights": dict(self.word_weights),
            "history": [
                {
                    "timestamp": r.timestamp,
                    "source": r.source,
                    "words": r.words,
                    "trigrams": list(r.trigrams),
                }
                for r in self.history
            ],
            "decay_rate": self.decay_rate,
            "min_word_length": self.min_word_length,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Lexicon saved to {path}: {len(self.absorbed_words)} words")

    @classmethod
    def load(
        cls,
        path: str,
        vocab: "Vocab",
        cooccur_field: Optional["CooccurField"] = None,
    ) -> "Lexicon":
        """Load lexicon state from file."""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)

        lex = cls(
            vocab=vocab,
            cooccur_field=cooccur_field,
            decay_rate=data.get("decay_rate", 0.99),
            min_word_length=data.get("min_word_length", 3),
        )

        lex.absorbed_words = set(data.get("absorbed_words", []))
        lex.absorbed_trigrams = set(
            tuple(t) for t in data.get("absorbed_trigrams", [])
        )
        lex.word_weights = data.get("word_weights", {})

        # Restore history
        for h in data.get("history", []):
            record = AbsorptionRecord(
                timestamp=h["timestamp"],
                source=h["source"],
                words=h["words"],
                trigrams=[tuple(t) for t in h["trigrams"]],
            )
            lex.history.append(record)

        logger.info(f"Lexicon loaded from {path}: {len(lex.absorbed_words)} words")
        return lex

    def __repr__(self) -> str:
        return (f"Lexicon(words={len(self.absorbed_words)}, "
                f"trigrams={len(self.absorbed_trigrams)}, "
                f"growth={self.stats().growth_rate:.2f}/turn)")

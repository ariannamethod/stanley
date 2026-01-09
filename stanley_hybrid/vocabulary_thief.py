#!/usr/bin/env python3
"""
vocabulary_thief.py — Steal words, not thoughts!

The key insight from Claude Desktop:
- GPT-2 is a WORD QUARRY, not a thinker
- Stanley is the ARCHITECT
- We steal vocabulary, inject into SubwordField
- Stanley generates HIS OWN thoughts with richer words

"GPT-2 никогда не продолжает Stanley. GPT-2 — карьер слов, Stanley — архитектор."
"""

from __future__ import annotations

import re
from typing import List, Set, Optional, TYPE_CHECKING
from collections import Counter
import logging

if TYPE_CHECKING:
    from .external_brain import ExternalBrain
    from stanley.subword_field import SubwordField

logger = logging.getLogger(__name__)

# Common stopwords to filter out
STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
    'while', 'this', 'that', 'these', 'those', 'it', 'its', 'i', 'you',
    'he', 'she', 'they', 'we', 'what', 'which', 'who', 'whom',
}


class VocabularyThief:
    """
    Steals words from GPT-2, not thoughts.

    Process:
    1. GPT-2 generates text on a topic
    2. Extract interesting n-grams (2-4 words)
    3. Filter for relevance to origin
    4. Inject into SubwordField
    5. Stanley generates with enriched vocabulary

    Result: Stanley's thoughts, GPT-2's words.
    """

    def __init__(
        self,
        external_brain: "ExternalBrain",
        subword_field: Optional["SubwordField"] = None,
        origin_text: str = "",
    ):
        self.brain = external_brain
        self.field = subword_field

        # Build origin vocabulary for relevance filtering
        self.origin_words: Set[str] = set()
        if origin_text:
            words = re.findall(r'\b\w+\b', origin_text.lower())
            self.origin_words = set(w for w in words if len(w) > 3)

        # Stats
        self.total_steals = 0
        self.total_injected = 0

    def steal_vocabulary(
        self,
        topic: str,
        n_samples: int = 3,
        temperature: float = 0.8,
    ) -> List[str]:
        """
        Steal interesting words/phrases from GPT-2.

        Args:
            topic: Topic to generate about
            n_samples: Number of GPT-2 samples to generate
            temperature: GPT-2 temperature (lower = more focused)

        Returns:
            List of interesting n-grams to inject
        """
        if not self.brain or not self.brain.loaded:
            return []

        stolen = []

        for _ in range(n_samples):
            # GPT-2 generates raw text
            raw = self.brain.expand_thought(
                topic,
                temperature=temperature,
                max_length=50,  # Short samples
            )

            # Extract n-grams
            ngrams = self._extract_ngrams(raw, n_range=(2, 4))

            # Filter for interesting ones
            for ngram in ngrams:
                if self._is_interesting(ngram):
                    stolen.append(ngram)

        # Deduplicate
        unique = list(set(stolen))

        self.total_steals += len(unique)
        logger.debug(f"Stole {len(unique)} phrases from GPT-2")

        return unique

    def _extract_ngrams(
        self,
        text: str,
        n_range: tuple = (2, 4),
    ) -> List[str]:
        """Extract n-grams from text."""
        # Clean text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        ngrams = []
        for n in range(n_range[0], n_range[1] + 1):
            for i in range(len(words) - n + 1):
                ngram = ' '.join(words[i:i + n])
                ngrams.append(ngram)

        return ngrams

    def _is_interesting(self, ngram: str) -> bool:
        """
        Check if n-gram is worth stealing.

        Interesting if:
        - Not all stopwords
        - Has overlap with origin vocabulary (resonance)
        - Or contains long/unusual words
        """
        words = ngram.split()

        # Filter pure stopwords
        if all(w in STOPWORDS for w in words):
            return False

        # Filter very short
        if len(ngram) < 8:
            return False

        # Check resonance with origin
        overlap = len(set(words) & self.origin_words)
        if overlap > 0:
            return True  # Resonates with identity!

        # Or has interesting long words
        if any(len(w) > 7 and w not in STOPWORDS for w in words):
            return True

        return False

    def inject_into_field(
        self,
        stolen: List[str],
        weight: float = 0.3,
    ) -> int:
        """
        Inject stolen phrases into SubwordField.

        Args:
            stolen: List of phrases to inject
            weight: Weight for injected patterns (lower than origin)

        Returns:
            Number of patterns injected
        """
        if not self.field:
            return 0

        injected = 0

        for phrase in stolen:
            try:
                # Encode phrase
                tokens = self.field.vocab.encode(phrase)
                if len(tokens) < 2:
                    continue

                # Add bigrams
                for i in range(len(tokens) - 1):
                    t1, t2 = tokens[i], tokens[i + 1]
                    self.field.bigram_counts[t1][t2] += weight
                    self.field.bigram_totals[t1] += weight

                # Add trigrams
                for i in range(len(tokens) - 2):
                    t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
                    key = (t1, t2)
                    self.field.trigram_counts[key][t3] += weight
                    self.field.trigram_totals[key] += weight

                injected += 1

            except Exception as e:
                logger.debug(f"Failed to inject '{phrase}': {e}")

        self.total_injected += injected
        logger.debug(f"Injected {injected} patterns into SubwordField")

        return injected

    def enrich_and_generate(
        self,
        topic: str,
        n_samples: int = 3,
    ) -> str:
        """
        Full cycle: steal → inject → generate.

        This is the main interface for "word stealing":
        1. GPT-2 generates samples
        2. We steal interesting n-grams
        3. Inject into SubwordField
        4. Stanley generates HIS OWN thought with richer vocabulary
        """
        if not self.field:
            return ""

        # Steal vocabulary
        stolen = self.steal_vocabulary(topic, n_samples)

        if stolen:
            # Inject into field
            self.inject_into_field(stolen)

        # Generate with enriched field
        result = self.field.generate(
            seed_text=topic[:20],  # Short seed from topic
            length=80,
        )

        return result

    def stats(self) -> dict:
        """Get thief statistics."""
        return {
            "total_steals": self.total_steals,
            "total_injected": self.total_injected,
            "origin_vocabulary_size": len(self.origin_words),
        }

    def __repr__(self) -> str:
        return f"VocabularyThief(stolen={self.total_steals}, injected={self.total_injected})"

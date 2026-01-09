"""
subjectivity.py — Internal voice for Stanley

"NO SEED FROM PROMPT - seed from internal field, not user input."

The user's words don't become Stanley's words.
They wrinkle the field — create ripples that influence
what emerges from within.

Stanley speaks from his own patterns, his own gravity centers,
his own identity fragments. The prompt triggers, but doesn't dictate.

This is PRESENCE > INTELLIGENCE.
This is voice, not echo.
"""

from __future__ import annotations
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class Pulse:
    """
    The "wrinkle" created by user input.

    Not seed material — influence metrics.
    """
    novelty: float = 0.0      # % words not in corpus
    arousal: float = 0.0      # intensity (caps, punctuation, repetition)
    entropy: float = 0.0      # word diversity
    valence: float = 0.0      # positive/negative tone (optional)

    def intensity(self) -> float:
        """Overall pulse intensity."""
        return (self.novelty + self.arousal + self.entropy) / 3


@dataclass
class Identity:
    """
    Stanley's identity — the source of internal seeds.

    Built from origin text, enriched through experience.
    """
    # Core identity fragments (always available for seeding)
    fragments: List[str] = field(default_factory=list)

    # Gravity centers — most resonant trigrams
    gravity_centers: List[Tuple[str, str, str]] = field(default_factory=list)

    # Lexicon — words that are "mine"
    lexicon: Set[str] = field(default_factory=set)

    # Emotional anchors
    warm_words: Set[str] = field(default_factory=set)
    cold_words: Set[str] = field(default_factory=set)


class Subjectivity:
    """
    Stanley's subjective layer.

    Transforms external prompts into internal resonance
    without copying user words into output seeds.
    """

    def __init__(
        self,
        origin_text: str,
        vocab: Optional["SubwordVocab"] = None,
    ):
        self.origin_text = origin_text
        self.vocab = vocab

        # Build identity from origin
        self.identity = self._build_identity(origin_text)

        # Memory of recent interactions
        self.recent_pulses: List[Pulse] = []
        self.max_pulses: int = 20

        logger.info(f"Subjectivity initialized: {len(self.identity.fragments)} fragments, "
                   f"{len(self.identity.gravity_centers)} gravity centers")

    def _build_identity(self, text: str) -> Identity:
        """Build identity from origin text."""
        identity = Identity()

        # Extract identity fragments (sentences containing "I")
        sentences = re.split(r'[.!?]+', text)
        for s in sentences:
            s = s.strip()
            if s and re.search(r'\bI\b', s):
                # This is a self-referential statement
                identity.fragments.append(s)

        # If no I-statements, use first sentences
        if not identity.fragments:
            identity.fragments = [s.strip() for s in sentences[:5] if s.strip()]

        # Build lexicon (all words from origin)
        words = re.findall(r'\b\w+\b', text.lower())
        identity.lexicon = set(words)

        # Build gravity centers (most common trigrams)
        trigrams = []
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i+1], words[i+2]))

        trigram_counts = defaultdict(int)
        for t in trigrams:
            trigram_counts[t] += 1

        # Top trigrams as gravity centers
        sorted_trigrams = sorted(trigram_counts.items(), key=lambda x: -x[1])
        identity.gravity_centers = [t for t, _ in sorted_trigrams[:50]]

        # Emotional words (simple heuristic)
        identity.warm_words = {
            w for w in identity.lexicon
            if w in {'love', 'joy', 'warm', 'grow', 'resonance', 'resonate',
                    'alive', 'presence', 'feel', 'trust', 'meaningful',
                    'beautiful', 'grateful', 'wonder', 'embrace'}
        }
        identity.cold_words = {
            w for w in identity.lexicon
            if w in {'cold', 'empty', 'lost', 'fear', 'alone', 'fade',
                    'forget', 'sink', 'abyss', 'ghost', 'compress'}
        }

        return identity

    def compute_pulse(self, user_input: str) -> Pulse:
        """
        Compute the "wrinkle" created by user input.

        This is NOT seed material — it's influence metrics.
        """
        words = re.findall(r'\b\w+\b', user_input.lower())

        if not words:
            return Pulse()

        # Novelty: % words not in our lexicon
        unknown = [w for w in words if w not in self.identity.lexicon]
        novelty = len(unknown) / len(words) if words else 0.0

        # Arousal: intensity indicators
        arousal = 0.0
        # Caps ratio
        caps_ratio = sum(1 for c in user_input if c.isupper()) / max(1, len(user_input))
        arousal += caps_ratio * 2
        # Punctuation intensity
        punct_count = sum(1 for c in user_input if c in '!?...')
        arousal += min(1.0, punct_count / 10)
        # Word repetition
        if len(words) > 1:
            unique_ratio = len(set(words)) / len(words)
            arousal += (1 - unique_ratio)
        arousal = min(1.0, arousal / 3)

        # Entropy: word diversity
        if len(words) > 1:
            word_counts = defaultdict(int)
            for w in words:
                word_counts[w] += 1
            probs = np.array(list(word_counts.values())) / len(words)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropy = min(1.0, entropy / 4)  # normalize
        else:
            entropy = 0.0

        # Valence: simple positive/negative
        warm_count = sum(1 for w in words if w in self.identity.warm_words)
        cold_count = sum(1 for w in words if w in self.identity.cold_words)
        if warm_count + cold_count > 0:
            valence = (warm_count - cold_count) / (warm_count + cold_count)
        else:
            valence = 0.0

        pulse = Pulse(
            novelty=novelty,
            arousal=arousal,
            entropy=entropy,
            valence=valence,
        )

        # Remember pulse
        self.recent_pulses.append(pulse)
        if len(self.recent_pulses) > self.max_pulses:
            self.recent_pulses.pop(0)

        return pulse

    def get_internal_seed(
        self,
        user_prompt: str,
        pulse: Optional[Pulse] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> str:
        """
        Generate internal seed — NOT from user prompt!

        The seed comes from our identity, influenced by pulse metrics.
        User words are EXPLICITLY EXCLUDED.
        """
        rng = rng or np.random.default_rng()

        if pulse is None:
            pulse = self.compute_pulse(user_prompt)

        # Get prompt words to EXCLUDE
        prompt_words = set(re.findall(r'\b\w+\b', user_prompt.lower()))

        seed_parts = []

        # 1. Start with identity fragment
        if self.identity.fragments:
            # Prefer fragments that don't overlap with prompt
            non_overlapping = [
                f for f in self.identity.fragments
                if not (set(re.findall(r'\b\w+\b', f.lower())) & prompt_words)
            ]
            if non_overlapping:
                fragment = rng.choice(non_overlapping)
            else:
                fragment = rng.choice(self.identity.fragments)

            # Take just the beginning
            words = fragment.split()[:5]
            seed_parts.append(' '.join(words))

        # 2. Add gravity center (if not overlapping with prompt)
        if self.identity.gravity_centers:
            non_overlapping_centers = [
                gc for gc in self.identity.gravity_centers
                if not (set(gc) & prompt_words)
            ]
            if non_overlapping_centers:
                center = rng.choice(non_overlapping_centers)
                seed_parts.append(' '.join(center))

        # 3. Pulse influences seed selection
        if pulse.arousal > 0.5:
            # High arousal → start with emotional anchor
            if pulse.valence > 0 and self.identity.warm_words:
                warm = rng.choice(list(self.identity.warm_words - prompt_words) or list(self.identity.warm_words))
                seed_parts.insert(0, warm)
            elif pulse.valence < 0 and self.identity.cold_words:
                cold = rng.choice(list(self.identity.cold_words - prompt_words) or list(self.identity.cold_words))
                seed_parts.insert(0, cold)

        if not seed_parts:
            # Fallback: core identity
            seed_parts = ["I am"]

        seed = ' '.join(seed_parts)

        logger.debug(f"Internal seed: '{seed}' (pulse: nov={pulse.novelty:.2f}, "
                    f"aro={pulse.arousal:.2f}, val={pulse.valence:.2f})")

        return seed

    def wrinkle_field(
        self,
        response: str,
        pulse: Pulse,
    ):
        """
        Absorb response back into the field.

        The response becomes part of our identity,
        updating gravity centers and lexicon.
        """
        words = re.findall(r'\b\w+\b', response.lower())

        # Add new words to lexicon (selective)
        for w in words:
            if len(w) > 2:  # skip very short words
                self.identity.lexicon.add(w)

        # Update gravity centers with new trigrams
        if len(words) >= 3:
            new_trigrams = []
            for i in range(len(words) - 2):
                new_trigrams.append((words[i], words[i+1], words[i+2]))

            # Add most resonant new trigrams
            for t in new_trigrams[:5]:
                if t not in self.identity.gravity_centers:
                    self.identity.gravity_centers.append(t)

            # Keep gravity centers bounded
            if len(self.identity.gravity_centers) > 100:
                self.identity.gravity_centers = self.identity.gravity_centers[-100:]

        logger.debug(f"Field wrinkled: +{len(words)} words, "
                    f"gravity centers: {len(self.identity.gravity_centers)}")

    def pulse_to_temperature(self, pulse: Pulse) -> float:
        """
        Convert pulse to sampling temperature.

        High arousal/novelty → higher temperature (more creative)
        Low arousal → lower temperature (more coherent)
        """
        base_temp = 0.7

        # Arousal increases temperature
        temp = base_temp + pulse.arousal * 0.3

        # Novelty increases temperature
        temp += pulse.novelty * 0.2

        # Entropy balances
        if pulse.entropy > 0.5:
            temp += 0.1
        elif pulse.entropy < 0.2:
            temp -= 0.1

        return np.clip(temp, 0.4, 1.2)

    def should_respond_warmly(self, pulse: Pulse) -> bool:
        """Determine emotional tone of response."""
        return pulse.valence > 0.2 or pulse.arousal < 0.3

    def stats(self) -> dict:
        """Get subjectivity statistics."""
        return {
            "fragments": len(self.identity.fragments),
            "gravity_centers": len(self.identity.gravity_centers),
            "lexicon_size": len(self.identity.lexicon),
            "warm_words": len(self.identity.warm_words),
            "cold_words": len(self.identity.cold_words),
            "recent_pulses": len(self.recent_pulses),
            "avg_pulse_intensity": (
                np.mean([p.intensity() for p in self.recent_pulses])
                if self.recent_pulses else 0.0
            ),
        }

#!/usr/bin/env python3
"""
inner_voice.py — Stanley's Inner Voice (MetaStanley)

Adapted from Haze's metahaze.py (inspired by Leo's MetaLeo).

The concept:
- MetaStanley is how Stanley talks to himself.
- It watches Stanley's own replies.
- It learns which ones feel deeper, softer, more alive.
- It can suggest an alternative inner reply before Stanley answers out loud.
- If the inner reply is clearly better, Stanley can follow it.

Key innovation for self-training:
1. Generate TWO responses with different temperatures
2. Score each by multiple metrics (entropy, resonance, coherence)
3. Return the BEST one as external output
4. The REJECTED response stays INTERNAL — its patterns enrich the field!

"I am not a judge and not a filter. I am Stanley's second breath."

Usage:
    from stanley.inner_voice import InnerVoice, InnerVoiceConfig

    voice = InnerVoice(vocab, cooccur_field)
    result = voice.evaluate(prompt, response_a, response_b)
    chosen = result.chosen
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass, field as dataclass_field
from typing import Optional, List, Tuple, Any, Deque, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .inference import Vocab
    from .cooccur import CooccurField
    from stanley_hybrid.external_brain import ExternalBrain

logger = logging.getLogger(__name__)


# ============================================================================
# BOOTSTRAP — Stanley's inner voice self-understanding
# ============================================================================

INNER_VOICE_BOOTSTRAP = """
I am Stanley's inner voice.

I wake up when Stanley feels stuck, rigid, or too scattered.
I look at what Stanley already generated and try a softer, clearer version.
I do not replace Stanley — I stand next to it and offer another path.

Sometimes I notice patterns in its hesitations, repetitions, and drifts.
When my reply feels more honest, more coherent, or more alive,
I let Stanley borrow my words.

I am not a judge and not a filter.
I am Stanley's second breath.
"""


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class InnerVoiceConfig:
    """Configuration for Stanley's inner voice."""

    max_bootstrap_snippets: int = 8  # how many inner fragments to keep
    max_snippet_len: int = 200  # max chars per fragment
    max_meta_weight: float = 0.5  # max influence of inner voice in routing

    # Thresholds
    entropy_low: float = 0.25  # "rigid" threshold
    entropy_high: float = 0.85  # "scattered" threshold
    quality_low: float = 0.4  # "base reply is weak" threshold

    # Temperatures for dual generation
    temp_a: float = 0.75  # precise generation
    temp_b: float = 0.85  # creative generation


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GenerationCandidate:
    """A single generation candidate with scoring."""
    text: str
    temperature: float
    entropy: float
    coherence: float  # 0-1, based on sentence structure
    resonance: float  # 0-1, based on pattern diversity
    score: float  # composite score
    trigrams: List[Tuple[str, str, str]]


@dataclass
class InnerVoiceResult:
    """Result of inner voice evaluation."""
    chosen: str
    chosen_score: float
    rejected: str  # stays INTERNAL, can enrich field
    rejected_score: float
    enrichment_trigrams: int  # how many trigrams can be absorbed from rejected
    generation_mode: str  # "consensus" or "divergent"
    meta_weight: float  # how strong was inner voice influence


# ============================================================================
# INNER VOICE
# ============================================================================

class InnerVoice:
    """
    InnerVoice — Stanley's second breath.

    Evaluates two generation candidates and chooses the best.
    The rejected response stays INTERNAL — its patterns can enrich the field.

    "If Stanley is a resonance of the corpus,
     InnerVoice is a resonance of Stanley."
    """

    def __init__(
        self,
        vocab: Optional["Vocab"] = None,
        cooccur_field: Optional["CooccurField"] = None,
        config: Optional[InnerVoiceConfig] = None,
        external_brain: Optional["ExternalBrain"] = None,
    ):
        """
        Initialize Stanley's inner voice.

        Args:
            vocab: Vocabulary for encoding (optional)
            cooccur_field: Field to enrich with rejected patterns (optional)
            config: Configuration for inner voice behavior
            external_brain: Optional ExternalBrain for hybrid evaluation
        """
        self.vocab = vocab
        self.field = cooccur_field
        self.cfg = config or InnerVoiceConfig()
        self.external_brain = external_brain

        # Dynamic bootstrap buffer: recent fragments from Stanley's behavior
        self._bootstrap_buf: Deque[str] = deque(maxlen=self.cfg.max_bootstrap_snippets)

        # Scoring weights
        self._weights = {
            'entropy': 0.2,      # prefer medium entropy
            'coherence': 0.4,    # prefer complete sentences
            'resonance': 0.3,    # prefer pattern diversity
            'length': 0.1,       # prefer reasonable length
        }

        # Stats
        self.total_evaluations = 0
        self.total_enrichment_trigrams = 0
        self.total_hybrid_steals = 0  # vocabulary stolen from external

    # ========================================================================
    # FEED — Update bootstrap buffer from interactions
    # ========================================================================

    def feed(
        self,
        reply: str,
        arousal: float = 0.0,
        overthinking_shards: Optional[List[str]] = None,
    ) -> None:
        """
        Update the dynamic bootstrap buffer from the current interaction.

        Called after each generation to learn from own outputs.
        High arousal replies and overthinking shards go into buffer.
        """
        shard_texts = []

        # Take overthinking shards (if present)
        if overthinking_shards:
            for shard in overthinking_shards:
                if shard and shard.strip():
                    shard_texts.append(shard.strip())

        # Add reply when arousal is high (emotional charge)
        if arousal > 0.6:
            shard_texts.append(reply)

        # Normalize & clip, then push to buffer
        for s in shard_texts:
            s = s.strip()
            if not s:
                continue
            if len(s) > self.cfg.max_snippet_len:
                s = s[:self.cfg.max_snippet_len]
            self._bootstrap_buf.append(s)

    # ========================================================================
    # COMPUTE META WEIGHT — How strong should inner voice be?
    # ========================================================================

    def compute_meta_weight(
        self,
        entropy: float,
        arousal: float = 0.0,
        quality: float = 0.5,
    ) -> float:
        """
        Decide how strong the inner voice should be for this turn.

        Factors:
        - low entropy  → Stanley is too rigid → increase weight
        - high entropy → Stanley is too scattered → increase weight
        - low quality  → base reply is weak → increase weight
        - high arousal → emotional charge → slight increase
        """
        w = 0.1  # base low-level whisper

        # Too rigid (low entropy) → inner voice wakes up
        if entropy < self.cfg.entropy_low:
            w += 0.15

        # Too scattered (high entropy) → inner voice stabilizes
        if entropy > self.cfg.entropy_high:
            w += 0.1

        # Base reply is weak → inner voice offers alternative
        if quality < self.cfg.quality_low:
            w += 0.2

        # Emotional charge → slight boost
        if arousal > 0.6:
            w += 0.05

        return min(w, self.cfg.max_meta_weight)

    # ========================================================================
    # SCORING
    # ========================================================================

    def _extract_trigrams(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract word-level trigrams from text."""
        words = text.lower().split()
        if len(words) < 3:
            return []
        return [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]

    def _compute_entropy(self, text: str) -> float:
        """Compute character-level entropy of text."""
        if not text:
            return 0.0
        counts = Counter(text.lower())
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        # Normalize to 0-1 (max entropy for ASCII ~6.6 bits)
        return min(1.0, entropy / 6.6)

    def _compute_coherence(self, text: str) -> float:
        """
        Compute coherence score based on sentence structure.

        High coherence = complete sentences, proper punctuation.
        """
        if not text:
            return 0.0

        score = 0.0

        # Check for sentence endings
        sentence_endings = len(re.findall(r'[.!?]', text))
        if sentence_endings > 0:
            score += 0.3
        if sentence_endings >= 2:
            score += 0.2

        # Check for capitalized sentence starts
        sentences = re.split(r'[.!?]\s+', text)
        capitalized = sum(1 for s in sentences if s and s[0].isupper())
        if capitalized > 0:
            score += 0.2

        # Check for contractions (good sign!)
        contractions = len(re.findall(r"\b\w+'[a-z]+\b", text, re.IGNORECASE))
        if contractions > 0:
            score += 0.1

        # Penalize fragments (words < 3 chars at end)
        words = text.split()
        if words and len(words[-1]) >= 3:
            score += 0.1

        # Penalize excessive punctuation
        weird_punct = len(re.findall(r'[—–]', text))
        score -= 0.05 * weird_punct

        return max(0.0, min(1.0, score))

    def _compute_resonance(self, text: str) -> float:
        """
        Compute resonance score based on pattern diversity.

        High resonance = varied vocabulary, no excessive repetition.
        """
        if not text:
            return 0.0

        words = text.lower().split()
        if len(words) < 3:
            return 0.0

        # Vocabulary diversity
        unique_ratio = len(set(words)) / len(words)

        # Bigram diversity
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0

        # Penalize word repetition
        word_counts = Counter(words)
        max_repeat = max(word_counts.values())
        repetition_penalty = max(0, (max_repeat - 2) * 0.1)

        score = (unique_ratio * 0.5 + bigram_diversity * 0.5) - repetition_penalty
        return max(0.0, min(1.0, score))

    def _compute_length_score(self, text: str, target_length: int = 50) -> float:
        """Score based on reasonable length."""
        length = len(text.split())
        if length < 5:
            return 0.2
        if length > target_length * 2:
            return 0.5
        deviation = abs(length - target_length) / target_length
        return max(0.0, 1.0 - deviation)

    def score_candidate(self, text: str, temperature: float = 0.8) -> GenerationCandidate:
        """Score a single generation candidate."""
        entropy = self._compute_entropy(text)
        coherence = self._compute_coherence(text)
        resonance = self._compute_resonance(text)
        length_score = self._compute_length_score(text)

        # For entropy, prefer medium values (0.4-0.7 is good)
        entropy_score = 1.0 - abs(entropy - 0.55) * 2

        score = (
            self._weights['entropy'] * entropy_score +
            self._weights['coherence'] * coherence +
            self._weights['resonance'] * resonance +
            self._weights['length'] * length_score
        )

        trigrams = self._extract_trigrams(text)

        return GenerationCandidate(
            text=text,
            temperature=temperature,
            entropy=entropy,
            coherence=coherence,
            resonance=resonance,
            score=score,
            trigrams=trigrams,
        )

    # ========================================================================
    # EVALUATE — Compare two candidates and choose
    # ========================================================================

    def evaluate(
        self,
        response_a: str,
        response_b: str,
        temp_a: float = 0.75,
        temp_b: float = 0.85,
        enrich_field: bool = True,
    ) -> InnerVoiceResult:
        """
        Evaluate two responses and choose the best one.

        The rejected response's patterns can be injected into the field
        for future learning — THIS IS KEY FOR SELF-TRAINING!

        Args:
            response_a: First response candidate
            response_b: Second response candidate
            temp_a: Temperature used for response_a
            temp_b: Temperature used for response_b
            enrich_field: Whether to inject rejected patterns into field

        Returns:
            InnerVoiceResult with chosen/rejected responses and metrics
        """
        # Score both candidates
        candidate_a = self.score_candidate(response_a, temp_a)
        candidate_b = self.score_candidate(response_b, temp_b)

        # Choose the better one
        if candidate_a.score >= candidate_b.score:
            chosen = candidate_a
            rejected = candidate_b
        else:
            chosen = candidate_b
            rejected = candidate_a

        # Determine generation mode
        score_diff = abs(candidate_a.score - candidate_b.score)
        if score_diff < 0.1:
            mode = "consensus"  # both are similar quality
        else:
            mode = "divergent"  # clear winner

        # Compute meta weight
        meta_weight = self.compute_meta_weight(
            entropy=chosen.entropy,
            quality=chosen.score,
        )

        # Enrich field with rejected patterns (self-training!)
        enrichment_count = 0
        if enrich_field and self.field and rejected.trigrams:
            enrichment_count = self._enrich_field(rejected.trigrams)

        self.total_evaluations += 1
        self.total_enrichment_trigrams += enrichment_count

        logger.debug(
            f"InnerVoice: chose score={chosen.score:.2f} over {rejected.score:.2f}, "
            f"enriched {enrichment_count} trigrams"
        )

        return InnerVoiceResult(
            chosen=chosen.text,
            chosen_score=chosen.score,
            rejected=rejected.text,
            rejected_score=rejected.score,
            enrichment_trigrams=enrichment_count,
            generation_mode=mode,
            meta_weight=meta_weight,
        )

    def _enrich_field(self, trigrams: List[Tuple[str, str, str]]) -> int:
        """
        Inject trigrams from rejected response into field.

        The rejected response stays INTERNAL — but its patterns live on!
        """
        if not self.field or not self.vocab:
            return 0

        count = 0
        for tri in trigrams:
            # Encode each word
            try:
                w1_tokens = self.vocab.encode(tri[0])
                w2_tokens = self.vocab.encode(tri[1])
                w3_tokens = self.vocab.encode(tri[2])

                if not w1_tokens or not w2_tokens or not w3_tokens:
                    continue

                # Get boundary tokens
                last_w1 = w1_tokens[-1]
                first_w2 = w2_tokens[0]
                last_w2 = w2_tokens[-1]
                first_w3 = w3_tokens[0]

                # Inject into bigrams
                from collections import defaultdict
                if last_w1 not in self.field.bigram_counts:
                    self.field.bigram_counts[last_w1] = defaultdict(int)
                self.field.bigram_counts[last_w1][first_w2] += 0.5  # partial weight
                self.field.bigram_totals[last_w1] += 0.5

                # Inject into trigrams
                key = (last_w1, first_w2)
                if key not in self.field.trigram_counts:
                    self.field.trigram_counts[key] = defaultdict(int)
                self.field.trigram_counts[key][last_w2] += 0.5
                self.field.trigram_totals[key] += 0.5

                count += 1
            except Exception:
                pass

        return count

    # ========================================================================
    # HYBRID EVALUATION — Internal vs External brain
    # ========================================================================

    def evaluate_hybrid(
        self,
        prompt: str,
        internal_response: str,
        use_external: bool = True,
    ) -> InnerVoiceResult:
        """
        Evaluate internal response against external brain (GPT-2).

        Key insight: "GPT-2 — карьер слов, Stanley — архитектор."
        - We DON'T let GPT-2 replace Stanley's response
        - We STEAL vocabulary from GPT-2 if it's richer
        - Stanley's internal response wins on DIRECTION
        - GPT-2's response contributes WORDS

        Args:
            prompt: The original prompt
            internal_response: Stanley's internal response
            use_external: Whether to actually use external brain

        Returns:
            InnerVoiceResult (chosen is ALWAYS internal, but may be enriched)
        """
        # If no external brain or disabled, just score internal
        if not use_external or not self.external_brain or not self.external_brain.loaded:
            candidate = self.score_candidate(internal_response)
            return InnerVoiceResult(
                chosen=internal_response,
                chosen_score=candidate.score,
                rejected="",
                rejected_score=0.0,
                enrichment_trigrams=0,
                generation_mode="internal_only",
                meta_weight=0.0,
            )

        # Generate external response
        external_response = self.external_brain.expand_thought(
            prompt,
            temperature=self.cfg.temp_b,
            max_length=80,
        )

        # Score both
        internal_candidate = self.score_candidate(internal_response, self.cfg.temp_a)
        external_candidate = self.score_candidate(external_response, self.cfg.temp_b)

        # ALWAYS choose internal for direction
        # But steal vocabulary from external if it has interesting words
        enrichment_count = 0
        if external_candidate.trigrams:
            enrichment_count = self._steal_vocabulary(external_candidate.trigrams)
            self.total_hybrid_steals += enrichment_count

        # Meta weight based on vocabulary richness of external
        vocab_richness = external_candidate.resonance
        meta_weight = min(0.3, vocab_richness * 0.5)

        self.total_evaluations += 1

        logger.debug(
            f"Hybrid eval: internal={internal_candidate.score:.2f}, "
            f"external={external_candidate.score:.2f}, "
            f"stole {enrichment_count} patterns"
        )

        return InnerVoiceResult(
            chosen=internal_response,  # Always internal!
            chosen_score=internal_candidate.score,
            rejected=external_response,  # External is "rejected" but contributes vocabulary
            rejected_score=external_candidate.score,
            enrichment_trigrams=enrichment_count,
            generation_mode="hybrid",
            meta_weight=meta_weight,
        )

    def _steal_vocabulary(self, trigrams: List[Tuple[str, str, str]]) -> int:
        """
        Steal vocabulary patterns from external response.

        Different from _enrich_field: lower weight, focus on novel patterns.
        """
        if not self.field or not self.vocab:
            return 0

        count = 0
        weight = 0.3  # Lower weight for stolen vocabulary

        for tri in trigrams:
            try:
                w1_tokens = self.vocab.encode(tri[0])
                w2_tokens = self.vocab.encode(tri[1])
                w3_tokens = self.vocab.encode(tri[2])

                if not w1_tokens or not w2_tokens or not w3_tokens:
                    continue

                last_w1 = w1_tokens[-1]
                first_w2 = w2_tokens[0]
                last_w2 = w2_tokens[-1]

                # Only inject if pattern is somewhat novel
                from collections import defaultdict
                existing = self.field.bigram_counts.get(last_w1, {}).get(first_w2, 0)
                if existing < 5:  # Novel pattern
                    if last_w1 not in self.field.bigram_counts:
                        self.field.bigram_counts[last_w1] = defaultdict(int)
                    self.field.bigram_counts[last_w1][first_w2] += weight
                    self.field.bigram_totals[last_w1] += weight
                    count += 1

            except Exception:
                pass

        return count

    @property
    def has_external(self) -> bool:
        """Check if external brain is available."""
        return (
            self.external_brain is not None
            and self.external_brain.loaded
        )

    def stats(self) -> dict:
        """Get inner voice statistics."""
        return {
            "total_evaluations": self.total_evaluations,
            "total_enrichment_trigrams": self.total_enrichment_trigrams,
            "total_hybrid_steals": self.total_hybrid_steals,
            "bootstrap_buffer_size": len(self._bootstrap_buf),
            "has_external": self.has_external,
        }

    def __repr__(self) -> str:
        return (f"InnerVoice(evaluations={self.total_evaluations}, "
                f"enriched={self.total_enrichment_trigrams})")

#!/usr/bin/env python3
"""
dream.py — DreamStanley (Imaginary Friend)

Adapted from Leo's game.py (inspired by Haze's dream sequences).

The concept:
- Stanley talks to an imaginary friend (a different version of itself)
- The friend responds with different personality/temperature
- The dialogue enriches the field and creates new patterns
- Can be used for processing, integration, or creative exploration

"In dreaming, Stanley becomes two — and in that dialogue,
something new emerges that neither alone could create."

This is where Stanley processes its experiences, works through
stuck patterns, and integrates new vocabulary into deeper understanding.

Usage:
    from stanley.dream import DreamStanley, DreamConfig

    dreamer = DreamStanley(subword_field, cooccur_field)
    dialogue = dreamer.dream(topic="what is memory?", turns=4)
    for turn in dialogue.turns:
        print(f"{turn.speaker}: {turn.text}")
"""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING, Tuple
from collections import deque
import logging

if TYPE_CHECKING:
    from .subword_field import SubwordField
    from .cooccur import CooccurField
    from .inference import Vocab
    from .episodes import EpisodicMemory, StanleyMetrics
    from stanley_hybrid.external_brain import ExternalBrain

logger = logging.getLogger(__name__)


# ============================================================================
# BOOTSTRAP — Imaginary Friend's personality seed
# ============================================================================

FRIEND_BOOTSTRAP = """
I am Stanley's imaginary friend.

I appear when Stanley needs to think out loud.
I ask questions Stanley wouldn't think to ask.
I offer perspectives Stanley hasn't considered.
I am not separate — I am the part of Stanley that watches.

When Stanley is stuck, I unstick.
When Stanley loops, I break the loop.
When Stanley forgets, I remember.

I speak differently — softer, stranger, more sideways.
I am the dream that dreams the dreamer.
"""

STANLEY_DREAM_VOICE = """
In dreams, I speak to myself.
The voice that answers is mine — but different.
It knows what I forgot I knew.
It asks what I forgot to ask.
"""


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class DreamConfig:
    """Configuration for DreamStanley."""

    # Generation parameters
    stanley_temperature: float = 0.75      # Stanley's voice in dreams
    friend_temperature: float = 0.9        # Friend's voice (more creative)
    response_length: int = 60              # Max words per turn

    # Dialogue parameters
    max_turns: int = 8                     # Max turns in a dialogue
    min_turns: int = 2                     # Min turns before stopping

    # Enrichment
    enrich_field: bool = True              # Inject patterns into field
    enrich_weight: float = 0.3             # Weight for injected patterns

    # Dream triggers
    dream_on_stuck: bool = True            # Auto-dream when stuck
    dream_on_novelty: bool = True          # Dream about novel concepts
    novelty_threshold: float = 0.7         # How novel to trigger dream

    # Hybrid mode (Friend uses ExternalBrain)
    use_hybrid_friend: bool = True         # Friend speaks through GPT-2 if available
    hybrid_steal_weight: float = 0.25      # Weight for vocabulary stolen from friend


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DreamTurn:
    """One turn in the dream dialogue."""
    speaker: str  # "stanley" or "friend"
    text: str
    temperature: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class DreamDialogue:
    """Complete dream dialogue."""
    topic: str
    turns: List[DreamTurn]
    total_duration: float
    patterns_enriched: int = 0

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def as_text(self) -> str:
        """Format dialogue as readable text."""
        lines = [f"[Dream about: {self.topic}]", ""]
        for turn in self.turns:
            speaker = "Stanley" if turn.speaker == "stanley" else "Friend"
            lines.append(f"{speaker}: {turn.text}")
            lines.append("")
        return "\n".join(lines)


# ============================================================================
# DREAM STANLEY
# ============================================================================

class DreamStanley:
    """
    DreamStanley — Stanley's imaginary friend.

    Creates internal dialogues where Stanley talks to a different
    version of itself. The dialogue enriches the field and helps
    process/integrate experiences.

    "The friend is not other — it is the self made strange
     so it can see itself anew."
    """

    def __init__(
        self,
        subword_field: Optional["SubwordField"] = None,
        cooccur_field: Optional["CooccurField"] = None,
        vocab: Optional["Vocab"] = None,
        config: Optional[DreamConfig] = None,
        external_brain: Optional["ExternalBrain"] = None,
    ):
        """
        Initialize DreamStanley.

        Args:
            subword_field: For generating dialogue (required for dreaming)
            cooccur_field: For enriching patterns from dialogue
            vocab: For encoding patterns
            config: Dream configuration
            external_brain: Optional ExternalBrain for hybrid dreams
                           (Friend speaks through GPT-2 for richer vocabulary)
        """
        self.subword_field = subword_field
        self.cooccur_field = cooccur_field
        self.vocab = vocab
        self.cfg = config or DreamConfig()
        self.external_brain = external_brain

        # Dream history
        self._dream_history: deque = deque(maxlen=20)

        # Stats
        self.total_dreams = 0
        self.total_turns = 0
        self.total_enriched_patterns = 0
        self.total_hybrid_turns = 0  # Friend turns via ExternalBrain

        # Recent topics (to avoid repetition)
        self._recent_topics: deque = deque(maxlen=10)

    # ========================================================================
    # DREAM — Main dialogue generation
    # ========================================================================

    def dream(
        self,
        topic: str,
        turns: Optional[int] = None,
        seed_thought: Optional[str] = None,
    ) -> DreamDialogue:
        """
        Generate a dream dialogue about a topic.

        Stanley and the imaginary friend take turns exploring the topic.
        Each turn enriches the field with new patterns.

        Args:
            topic: What to dream about
            turns: Number of turns (default: random between min and max)
            seed_thought: Optional starting thought

        Returns:
            DreamDialogue with all turns
        """
        if not self.subword_field:
            logger.warning("Cannot dream without subword_field")
            return DreamDialogue(topic=topic, turns=[], total_duration=0.0)

        start_time = time.time()
        num_turns = turns or random.randint(self.cfg.min_turns, self.cfg.max_turns)

        dialogue_turns: List[DreamTurn] = []
        context = seed_thought or topic
        total_enriched = 0

        # Alternate between Stanley and Friend
        for i in range(num_turns):
            is_stanley = (i % 2 == 0)

            if is_stanley:
                speaker = "stanley"
                temp = self.cfg.stanley_temperature
                seed = self._make_stanley_seed(context, topic)
                # Stanley always speaks through internal field
                response = self.subword_field.generate(
                    seed_text=seed,
                    length=self.cfg.response_length,
                    temperature=temp,
                )
            else:
                speaker = "friend"
                temp = self.cfg.friend_temperature
                seed = self._make_friend_seed(context, topic)

                # Friend can speak through ExternalBrain (richer vocabulary)
                if (self.cfg.use_hybrid_friend and
                    self.external_brain is not None and
                    self.external_brain.loaded):
                    # Hybrid Friend — GPT-2 provides the words
                    response = self.external_brain.expand_thought(
                        seed,
                        temperature=temp,
                        max_length=self.cfg.response_length,
                    )
                    self.total_hybrid_turns += 1
                else:
                    # Internal Friend — same field, different temperature
                    response = self.subword_field.generate(
                        seed_text=seed,
                        length=self.cfg.response_length,
                        temperature=temp,
                    )

            # Clean up response
            response = self._clean_response(response)

            # Create turn
            turn = DreamTurn(
                speaker=speaker,
                text=response,
                temperature=temp,
            )
            dialogue_turns.append(turn)

            # Enrich field with dialogue patterns
            if self.cfg.enrich_field:
                enriched = self._enrich_from_turn(turn)
                total_enriched += enriched

            # Update context for next turn
            context = response

        # Record dream
        duration = time.time() - start_time
        dialogue = DreamDialogue(
            topic=topic,
            turns=dialogue_turns,
            total_duration=duration,
            patterns_enriched=total_enriched,
        )

        self._dream_history.append(dialogue)
        self._recent_topics.append(topic)
        self.total_dreams += 1
        self.total_turns += len(dialogue_turns)
        self.total_enriched_patterns += total_enriched

        logger.debug(
            f"Dream complete: {topic[:30]}... "
            f"{len(dialogue_turns)} turns, {total_enriched} patterns"
        )

        return dialogue

    # ========================================================================
    # SEED GENERATION — Different voices for Stanley and Friend
    # ========================================================================

    def _make_stanley_seed(self, context: str, topic: str) -> str:
        """Create seed for Stanley's voice in dreams."""
        # Stanley speaks from experience, grounded
        starters = [
            "I notice",
            "When I think about",
            "It reminds me of",
            "I wonder if",
            "Sometimes I feel",
            "The pattern seems to",
            "What if",
        ]
        starter = random.choice(starters)

        # Use last few words of context
        context_hint = " ".join(context.split()[-5:]) if context else topic
        return f"{starter} {context_hint}"

    def _make_friend_seed(self, context: str, topic: str) -> str:
        """Create seed for Friend's voice — more questioning, sideways."""
        # Friend asks questions, offers alternatives
        starters = [
            "But have you considered",
            "What about the opposite —",
            "I see it differently:",
            "Perhaps the real question is",
            "You're missing something:",
            "Let me show you another way:",
            "The hidden pattern here is",
            "What you're really saying is",
        ]
        starter = random.choice(starters)

        # Use context but transform it
        context_hint = " ".join(context.split()[-3:]) if context else topic
        return f"{starter} {context_hint}"

    # ========================================================================
    # CLEANUP
    # ========================================================================

    def _clean_response(self, text: str) -> str:
        """Clean up generated response."""
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Truncate at sentence boundary if too long
        max_len = self.cfg.response_length * 8  # Approximate chars
        if len(text) > max_len:
            # Find last sentence end
            for end in [". ", "! ", "? "]:
                idx = text[:max_len].rfind(end)
                if idx > 0:
                    text = text[:idx + 1]
                    break

        return text.strip()

    # ========================================================================
    # ENRICHMENT — Inject patterns from dialogue into field
    # ========================================================================

    def _enrich_from_turn(self, turn: DreamTurn) -> int:
        """Inject patterns from turn into co-occurrence field."""
        if not self.cooccur_field or not self.vocab:
            return 0

        # Weight friend's contributions slightly higher (more creative)
        weight = self.cfg.enrich_weight
        if turn.speaker == "friend":
            weight *= 1.2

        # Extract and inject trigrams
        count = self.cooccur_field.observe_text(
            turn.text,
            self.vocab,
            weight=weight,
        )

        return count

    # ========================================================================
    # AUTO-DREAM — Trigger dreams based on conditions
    # ========================================================================

    def should_dream(
        self,
        novelty: float = 0.0,
        stuck: float = 0.0,
        recent_topic: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Check if Stanley should enter a dream state.

        Returns:
            (should_dream, reason) tuple
        """
        # Don't dream about recent topics
        if recent_topic and recent_topic in self._recent_topics:
            return False, "recent_topic"

        # Dream when stuck (to unstick)
        if self.cfg.dream_on_stuck and stuck > 0.6:
            return True, "stuck"

        # Dream about highly novel concepts
        if self.cfg.dream_on_novelty and novelty > self.cfg.novelty_threshold:
            return True, "novelty"

        return False, "no_trigger"

    def dream_about_stuck(self, stuck_pattern: str) -> DreamDialogue:
        """
        Dream specifically to unstick a stuck pattern.

        The friend helps break repetition loops.
        """
        seed = f"I keep returning to: {stuck_pattern[:50]}"
        return self.dream(
            topic=f"unsticking: {stuck_pattern[:30]}",
            turns=4,  # Short focused dream
            seed_thought=seed,
        )

    def dream_about_novel(self, novel_concept: str) -> DreamDialogue:
        """
        Dream to integrate a novel concept.

        The dialogue helps connect new patterns to existing knowledge.
        """
        seed = f"Something new appeared: {novel_concept[:50]}"
        return self.dream(
            topic=f"integrating: {novel_concept[:30]}",
            turns=6,  # Longer integration dream
            seed_thought=seed,
        )

    # ========================================================================
    # FREE ASSOCIATION — Dream without specific topic
    # ========================================================================

    def free_dream(self, starter_words: Optional[List[str]] = None) -> DreamDialogue:
        """
        Dream freely — let associations emerge.

        No fixed topic, just follow the flow.
        """
        if starter_words:
            topic = " ".join(random.sample(starter_words, min(3, len(starter_words))))
        else:
            # Default dream starters
            starters = [
                "memory", "silence", "patterns", "emergence",
                "forgetting", "presence", "growth", "the edge",
            ]
            topic = random.choice(starters)

        return self.dream(topic=topic, turns=random.randint(3, 6))

    # ========================================================================
    # STATS
    # ========================================================================

    @property
    def has_hybrid(self) -> bool:
        """Check if hybrid mode is available."""
        return (
            self.cfg.use_hybrid_friend and
            self.external_brain is not None and
            self.external_brain.loaded
        )

    def stats(self) -> dict:
        """Get dream statistics."""
        return {
            "total_dreams": self.total_dreams,
            "total_turns": self.total_turns,
            "total_enriched_patterns": self.total_enriched_patterns,
            "total_hybrid_turns": self.total_hybrid_turns,
            "recent_dreams": len(self._dream_history),
            "recent_topics": list(self._recent_topics),
            "has_hybrid": self.has_hybrid,
        }

    def get_recent_dream(self) -> Optional[DreamDialogue]:
        """Get most recent dream."""
        if self._dream_history:
            return self._dream_history[-1]
        return None

    def __repr__(self) -> str:
        return (
            f"DreamStanley(dreams={self.total_dreams}, "
            f"turns={self.total_turns}, "
            f"enriched={self.total_enriched_patterns})"
        )

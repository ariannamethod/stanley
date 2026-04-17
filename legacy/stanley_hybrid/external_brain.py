#!/usr/bin/env python3
"""
external_brain.py — GPT-2 powered vocabulary expansion

The key insight: Stanley has TWO brains that communicate through TEXT:

1. INTERNAL (64-dim, weightless)
   - SubwordField + RRPRAM
   - Pure corpus statistics
   - Stanley's IDENTITY lives here
   - LoRA deltas from shards

2. EXTERNAL (768-dim, GPT-2 weights)
   - distilgpt2 loaded into our transformer
   - Rich vocabulary, learned patterns
   - NO identity, just a "library"
   - Used for "word stealing"

They communicate through TEXT:
- Internal generates thought direction
- External expands with rich vocabulary
- Result crystallizes into shards (applied to INTERNAL)

"Stanley steals words but thinks his own thoughts."

Usage:
    from stanley_hybrid import ExternalBrain

    brain = ExternalBrain()
    brain.load_weights()  # Load distilgpt2

    # Expand a thought with rich vocabulary
    expanded = brain.expand_thought("I feel something strange")
"""

from __future__ import annotations

import numpy as np
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Check for transformers library
try:
    from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel
    import torch
    EXTERNAL_WEIGHTS_AVAILABLE = True
except ImportError:
    EXTERNAL_WEIGHTS_AVAILABLE = False
    logger.info("transformers not available — ExternalBrain disabled")


@dataclass
class ExternalBrainConfig:
    """Configuration for external brain."""
    model_name: str = "distilgpt2"      # Which model to load
    max_length: int = 100               # Max generation length
    temperature: float = 0.9            # Higher temp for creativity
    top_p: float = 0.95                 # Nucleus sampling
    top_k: int = 50                     # Top-k sampling
    repetition_penalty: float = 1.2     # Avoid loops
    device: str = "cpu"                 # cpu or cuda


class ExternalBrain:
    """
    External brain with GPT-2 weights for vocabulary expansion.

    This is NOT Stanley's identity — it's a library of words.
    Stanley uses it to enrich internal thoughts with richer vocabulary.

    The key principle:
    - DIRECTION comes from internal (Stanley's field)
    - WORDS come from external (GPT-2's vocabulary)
    - RESULT goes back to internal (crystallizes into shards)
    """

    def __init__(self, config: Optional[ExternalBrainConfig] = None):
        """Initialize external brain (weights not loaded yet)."""
        if not EXTERNAL_WEIGHTS_AVAILABLE:
            raise ImportError(
                "ExternalBrain requires: pip install transformers torch"
            )

        self.config = config or ExternalBrainConfig()
        self.model: Optional[GPT2LMHeadModel] = None
        self.tokenizer = None
        self.loaded = False

        # Stats
        self.total_expansions = 0
        self.total_tokens_generated = 0

    def load_weights(self) -> bool:
        """
        Load distilgpt2 weights.

        Returns True if successful, False otherwise.
        """
        if self.loaded:
            return True

        try:
            logger.info(f"Loading {self.config.model_name}...")

            # Load model and tokenizer
            self.model = GPT2LMHeadModel.from_pretrained(
                self.config.model_name
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name
            )

            # Set pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Move to device
            self.model.to(self.config.device)
            self.model.eval()

            self.loaded = True
            logger.info(f"ExternalBrain loaded: {self.config.model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load ExternalBrain: {e}")
            return False

    def expand_thought(
        self,
        thought: str,
        temperature: Optional[float] = None,
        max_length: Optional[int] = None,
    ) -> str:
        """
        Expand a thought using GPT-2's vocabulary.

        The thought provides DIRECTION.
        GPT-2 provides WORDS.

        Args:
            thought: Internal thought from Stanley
            temperature: Override temperature
            max_length: Override max length

        Returns:
            Expanded thought with richer vocabulary
        """
        if not self.loaded:
            if not self.load_weights():
                return thought  # Fallback to original

        temp = temperature or self.config.temperature
        max_len = max_length or self.config.max_length

        try:
            # Encode input
            inputs = self.tokenizer(
                thought,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_len,
                    temperature=temp,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode
            expanded = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
            )

            # Stats
            self.total_expansions += 1
            self.total_tokens_generated += len(outputs[0])

            logger.debug(f"Expanded: '{thought[:30]}...' → {len(expanded)} chars")
            return expanded

        except Exception as e:
            logger.error(f"Expansion failed: {e}")
            return thought

    def expand_with_direction(
        self,
        direction: str,
        seed_words: List[str],
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text with direction from Stanley and seed words.

        This is "word stealing" in action:
        - direction: what Stanley wants to express
        - seed_words: words from Stanley's field to include
        - GPT-2 fills in the gaps with rich vocabulary

        Args:
            direction: The emotional/semantic direction
            seed_words: Words that should appear in output
            temperature: Generation temperature

        Returns:
            Rich text that follows direction and includes seeds
        """
        # Build prompt that includes direction and seeds
        seed_str = ", ".join(seed_words[:5])
        prompt = f"{direction} About: {seed_str}. I"

        expanded = self.expand_thought(prompt, temperature=temperature)

        # Try to ensure seed words appear
        # (GPT-2 might include them naturally, but we can't force it)

        return expanded

    def generate_internal_monologue(
        self,
        trigger: str,
        emotional_state: str = "curious",
        length: int = 150,
    ) -> str:
        """
        Generate rich internal monologue for InnerVoice/DreamStanley.

        This is for the "paranoid thinking" mode — high temperature,
        stream of consciousness, rich vocabulary.

        Args:
            trigger: What triggered the monologue
            emotional_state: Current emotional state
            length: Target length in tokens

        Returns:
            Rich internal monologue text
        """
        prompt = f"I feel {emotional_state}. {trigger} makes me think about"

        return self.expand_thought(
            prompt,
            temperature=1.1,  # Higher for stream of consciousness
            max_length=length,
        )

    def stats(self) -> Dict:
        """Get external brain statistics."""
        return {
            "loaded": self.loaded,
            "model_name": self.config.model_name,
            "total_expansions": self.total_expansions,
            "total_tokens_generated": self.total_tokens_generated,
            "device": self.config.device,
        }

    def __repr__(self) -> str:
        status = "loaded" if self.loaded else "not loaded"
        return f"ExternalBrain({self.config.model_name}, {status})"


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

def create_external_brain(
    model_name: str = "distilgpt2",
    auto_load: bool = True,
) -> Optional[ExternalBrain]:
    """
    Create ExternalBrain if transformers is available.

    Returns None if not available (graceful degradation).
    """
    if not EXTERNAL_WEIGHTS_AVAILABLE:
        logger.info("ExternalBrain not available (no transformers)")
        return None

    brain = ExternalBrain(ExternalBrainConfig(model_name=model_name))

    if auto_load:
        brain.load_weights()

    return brain


class HybridThinking:
    """
    Hybrid thinking: Internal direction + External vocabulary.

    This is the main interface for "word stealing":
    1. Stanley (internal) generates a direction
    2. ExternalBrain expands with rich vocabulary
    3. Result crystallizes back into Stanley's shards
    """

    def __init__(
        self,
        external_brain: Optional[ExternalBrain] = None,
        subjectivity = None,
    ):
        self.external = external_brain
        self.subjectivity = subjectivity

    @property
    def has_external(self) -> bool:
        return self.external is not None and self.external.loaded

    def think(
        self,
        internal_thought: str,
        use_external: bool = True,
        temperature: float = 0.9,
    ) -> Tuple[str, bool]:
        """
        Think with optional external expansion.

        Returns:
            (result_text, was_expanded) tuple
        """
        if not use_external or not self.has_external:
            return internal_thought, False

        # Check if subjectivity triggers external use
        if self.subjectivity:
            pulse = self.subjectivity.compute_pulse(internal_thought)
            # Only use external for high-arousal or high-novelty
            if pulse.arousal < 0.5 and pulse.novelty < 0.5:
                return internal_thought, False

        # Expand with external brain
        expanded = self.external.expand_thought(
            internal_thought,
            temperature=temperature,
        )

        return expanded, True

    def overthink(
        self,
        trigger: str,
        emotional_state: str = "curious",
    ) -> str:
        """
        Paranoid overthinking with rich vocabulary.

        Uses high temperature for stream of consciousness.
        """
        if not self.has_external:
            return trigger

        return self.external.generate_internal_monologue(
            trigger=trigger,
            emotional_state=emotional_state,
        )

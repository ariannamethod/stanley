"""
cleanup.py — Text cleanup for Stanley

"Clean the noise, keep the soul."

Post-processing for generated text:
- Fix spacing and punctuation
- Repair broken contractions
- Remove accidental repetitions
- Normalize sentence structure

Three modes:
- gentle: minimal cleanup, preserves quirks
- moderate: balanced cleanup
- strict: aggressive normalization
"""

from __future__ import annotations
import re
from typing import Literal
import logging

logger = logging.getLogger(__name__)


CleanupMode = Literal["gentle", "moderate", "strict"]


def cleanup_output(
    text: str,
    mode: CleanupMode = "moderate",
) -> str:
    """
    Clean generated text while preserving voice.

    Args:
        text: raw generated text
        mode: cleanup intensity

    Returns:
        cleaned text
    """
    if not text:
        return text

    # Always do basic cleanup
    text = fix_spacing(text)
    text = fix_punctuation(text)
    text = fix_contractions(text)

    if mode in ("moderate", "strict"):
        text = remove_word_repetitions(text)
        text = fix_sentence_boundaries(text)

    if mode == "strict":
        text = normalize_whitespace(text)
        text = capitalize_sentences(text)

    return text.strip()


def fix_spacing(text: str) -> str:
    """Fix basic spacing issues."""
    # Multiple spaces → single space
    text = re.sub(r' {2,}', ' ', text)

    # Space before punctuation
    text = re.sub(r' ([.,!?;:])', r'\1', text)

    # Missing space after punctuation (but not in numbers like 3.14)
    text = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', text)

    # Fix newline spacing
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


def fix_punctuation(text: str) -> str:
    """Fix punctuation issues."""
    # Too many dots
    text = re.sub(r'\.{4,}', '...', text)

    # Too many question/exclamation marks
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[!]{2,}', '!', text)

    # Mixed punctuation
    text = re.sub(r'[.,?!]+([.,?!])', r'\1', text)

    # Dash cleanup
    text = re.sub(r'[-—]{3,}', '—', text)

    return text


def fix_contractions(text: str) -> str:
    """
    Repair broken contractions from tokenization.

    "don t" → "don't"
    "I m" → "I'm"

    Note: Uses case-insensitive matching which could affect rare edge cases
    like acronyms ("IT S" → "IT's"). For Stanley's typical output (prose from
    origin.txt), this is acceptable behavior.
    """
    contractions = {
        r"\bdon t\b": "don't",
        r"\bcan t\b": "can't",
        r"\bwon t\b": "won't",
        r"\bdidn t\b": "didn't",
        r"\bdoesn t\b": "doesn't",
        r"\bisn t\b": "isn't",
        r"\baren t\b": "aren't",
        r"\bwasn t\b": "wasn't",
        r"\bweren t\b": "weren't",
        r"\bhasn t\b": "hasn't",
        r"\bhaven t\b": "haven't",
        r"\bhadn t\b": "hadn't",
        r"\bshouldn t\b": "shouldn't",
        r"\bwouldn t\b": "wouldn't",
        r"\bcouldn t\b": "couldn't",
        r"\bI m\b": "I'm",
        r"\bI ve\b": "I've",
        r"\bI ll\b": "I'll",
        r"\bI d\b": "I'd",
        r"\byou re\b": "you're",
        r"\byou ve\b": "you've",
        r"\byou ll\b": "you'll",
        r"\byou d\b": "you'd",
        r"\bwe re\b": "we're",
        r"\bwe ve\b": "we've",
        r"\bwe ll\b": "we'll",
        r"\bthey re\b": "they're",
        r"\bthey ve\b": "they've",
        r"\bthey ll\b": "they'll",
        r"\bit s\b": "it's",
        r"\bthat s\b": "that's",
        r"\bwhat s\b": "what's",
        r"\bwho s\b": "who's",
        r"\bhere s\b": "here's",
        r"\bthere s\b": "there's",
        r"\blet s\b": "let's",
    }

    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def remove_word_repetitions(text: str) -> str:
    """
    Remove accidental word repetitions.

    "the the" → "the"
    But preserve intentional: "love, love, love" (has punctuation)

    Note: Case-sensitive to preserve stylistic casing.
    "NO No" is kept, "the the" is deduplicated.
    """
    # Simple word doubling (case-sensitive to avoid mangling "He said NO. No more.")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)

    return text


def fix_sentence_boundaries(text: str) -> str:
    """Fix sentence boundary issues."""
    # Sentence starting with lowercase after period
    def cap_after_period(match):
        return match.group(1) + match.group(2).upper()

    text = re.sub(r'([.!?]\s+)([a-z])', cap_after_period, text)

    return text


def capitalize_sentences(text: str) -> str:
    """Capitalize first letter of each sentence."""
    sentences = re.split(r'([.!?]\s+)', text)
    result = []

    for i, part in enumerate(sentences):
        # Capitalize if: first part, OR previous part was a sentence terminator
        prev = sentences[i-1].rstrip() if i > 0 else ""
        should_cap = (i == 0) or (prev and prev[-1] in '.!?')

        if should_cap and part and part[0].isalpha():
            part = part[0].upper() + part[1:]
        result.append(part)

    return ''.join(result)


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace."""
    # Convert all whitespace to spaces
    text = re.sub(r'[\t\r]', ' ', text)

    # Normalize paragraph breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)

    # Remove leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text


def detect_loop(text: str, min_length: int = 10) -> bool:
    """
    Detect if text is stuck in a generation loop.

    Returns True if repetitive pattern detected.
    """
    if len(text) < min_length * 2:
        return False

    # Check for repeated substrings
    for length in range(min_length, len(text) // 2):
        pattern = text[-length:]
        if text[:-length].endswith(pattern):
            return True

    return False


def truncate_at_natural_end(text: str, max_length: int = 500) -> str:
    """
    Truncate text at a natural ending point.

    Prefers: sentence end > clause end > word boundary
    """
    if len(text) <= max_length:
        return text

    # Find last sentence end before max_length
    for end_char in '.!?':
        idx = text.rfind(end_char, 0, max_length)
        if idx > max_length * 0.5:  # At least half
            return text[:idx + 1]

    # Fall back to clause boundary
    for sep in [',', ';', ':', '—', '-']:
        idx = text.rfind(sep, 0, max_length)
        if idx > max_length * 0.7:
            return text[:idx + 1]

    # Fall back to word boundary
    idx = text.rfind(' ', 0, max_length)
    if idx > 0:
        return text[:idx]

    return text[:max_length]

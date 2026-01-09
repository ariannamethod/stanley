"""
cleanup.py — Text cleanup for Stanley

"Clean the noise, keep the soul."

Post-processing for generated text:
- Fix spacing and punctuation
- Repair broken contractions (including orphans!)
- Remove accidental repetitions (but preserve poetic ones)
- Normalize sentence structure

Three modes:
- gentle: minimal cleanup, preserves quirks
- moderate: balanced cleanup
- strict: aggressive normalization

Based on Haze's cleanup.py — the wisdom of the "придурок" 😂
"""

from __future__ import annotations
import re
import math
from typing import Literal, List, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)


CleanupMode = Literal["gentle", "moderate", "strict"]


# ============= POETIC REPETITION DETECTION =============

def _detect_poetic_repetition(text: str) -> List[Tuple[int, int, str]]:
    """
    Detect intentional poetic repetitions (anaphora, refrain patterns).

    Returns:
        List of (start, end, pattern) tuples for regions to preserve
    """
    preserve_regions = []

    # Pattern 1: Comma-separated repetitions (e.g., "love, love, love")
    pattern = r'\b(\w+)(?:,\s+\1){1,}\b'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        preserve_regions.append((match.start(), match.end(), 'comma_repetition'))

    # Pattern 2: Emphatic repetition with punctuation
    # "Never, never, never!" or "Why? Why? Why?"
    pattern = r'\b(\w+)([,.!?])\s+\1\2(?:\s+\1\2)*'
    for match in re.finditer(pattern, text):
        preserve_regions.append((match.start(), match.end(), 'emphatic_repetition'))

    return preserve_regions


def _is_in_preserve_region(pos: int, regions: List[Tuple[int, int, str]]) -> bool:
    """Check if position is within any preserve region."""
    return any(start <= pos < end for start, end, _ in regions)


def _calculate_local_entropy(text: str, window: int = 20) -> float:
    """
    Calculate local character-level entropy.
    Used to detect coherent vs random text.
    """
    if len(text) < 2:
        return 0.0

    chars = list(text[-window:] if len(text) > window else text)
    counts = Counter(chars)
    total = len(chars)

    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)

    return entropy


# ============= MAIN CLEANUP =============

def cleanup_output(
    text: str,
    mode: CleanupMode = "moderate",
    preserve_resonance: bool = True,
) -> str:
    """
    Clean generated text while preserving voice.

    Args:
        text: raw generated text
        mode: cleanup intensity
        preserve_resonance: if True, detect and preserve poetic patterns

    Returns:
        cleaned text
    """
    if not text:
        return text

    # Detect poetic repetitions to preserve
    preserve_regions = []
    if preserve_resonance:
        preserve_regions = _detect_poetic_repetition(text)

    # Always do basic cleanup
    text = fix_spacing(text)
    text = fix_punctuation(text)
    text = fix_contractions(text)
    text = fix_orphan_contractions(text)  # NEW: from Haze!

    if mode in ("moderate", "strict"):
        text = remove_word_repetitions(text, preserve_regions)
        text = fix_sentence_boundaries(text)
        text = fix_possessive_vs_contraction(text)  # NEW: from Haze!

    if mode == "strict":
        text = normalize_whitespace(text)
        text = capitalize_sentences(text)
        text = remove_em_dashes(text)  # NEW: presence, not dialogue

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

    # Double dots → single dot (but not ellipsis)
    text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text)

    # Clean punctuation garbage
    text = re.sub(r'\.\s+,', '.', text)
    text = re.sub(r',\s*,', ',', text)

    return text


def fix_contractions(text: str) -> str:
    """
    Repair broken contractions from tokenization.

    "don t" → "don't"
    "I m" → "I'm"
    """
    contractions = {
        # n't contractions
        r"\bdon\s*t\b": "don't",
        r"\bcan\s*t\b": "can't",
        r"\bwon\s*t\b": "won't",
        r"\bdidn\s*t\b": "didn't",
        r"\bdoesn\s*t\b": "doesn't",
        r"\bisn\s*t\b": "isn't",
        r"\baren\s*t\b": "aren't",
        r"\bwasn\s*t\b": "wasn't",
        r"\bweren\s*t\b": "weren't",
        r"\bhasn\s*t\b": "hasn't",
        r"\bhaven\s*t\b": "haven't",
        r"\bhadn\s*t\b": "hadn't",
        r"\bshouldn\s*t\b": "shouldn't",
        r"\bwouldn\s*t\b": "wouldn't",
        r"\bcouldn\s*t\b": "couldn't",
        r"\bain\s*t\b": "ain't",
        # I contractions
        r"\bI\s*m\b": "I'm",
        r"\bI\s*ve\b": "I've",
        r"\bI\s*ll\b": "I'll",
        r"\bI\s*d\b": "I'd",
        # you contractions
        r"\byou\s*re\b": "you're",
        r"\byou\s*ve\b": "you've",
        r"\byou\s*ll\b": "you'll",
        r"\byou\s*d\b": "you'd",
        # we contractions
        r"\bwe\s*re\b": "we're",
        r"\bwe\s*ve\b": "we've",
        r"\bwe\s*ll\b": "we'll",
        # they contractions
        r"\bthey\s*re\b": "they're",
        r"\bthey\s*ve\b": "they've",
        r"\bthey\s*ll\b": "they'll",
        # 's contractions (use \s+ to avoid matching "its")
        r"\bit\s+s\b": "it's",
        r"\bhe\s+s\b": "he's",
        r"\bshe\s+s\b": "she's",
        r"\bthat\s+s\b": "that's",
        r"\bwhat\s+s\b": "what's",
        r"\bwho\s+s\b": "who's",
        r"\bhere\s+s\b": "here's",
        r"\bthere\s+s\b": "there's",
        r"\blet\s+s\b": "let's",
    }

    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Advanced compound contractions (from Haze)
    advanced = {
        r"\bwould\s+have\b": "would've",
        r"\bcould\s+have\b": "could've",
        r"\bshould\s+have\b": "should've",
        r"\bmight\s+have\b": "might've",
        r"\bmust\s+have\b": "must've",
        r"\by\s+all\b": "y'all",
    }

    for pattern, replacement in advanced.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def fix_orphan_contractions(text: str) -> str:
    """
    Fix orphan contractions — the Haze secret sauce!

    "don" + verb → "don't" + verb
    "don" alone → "ain't" (CHARACTER!)

    Philosophy: If subword tokenization cuts "don't" to just "don",
    we rescue it as "ain't" which has CHARACTER and fits the vibe!
    """
    # "don" + space + verb → "don't" + verb
    common_verbs = (
        r"believe|think|know|want|need|like|care|worry|mind|understand|"
        r"remember|forget|see|hear|feel|get|go|do|be|have|make|take|give|"
        r"say|tell|ask|try|look|come|put|let|seem|mean|stop|start|die|live|"
        r"stay|leave|keep|wait|work|play|sleep|eat|drink|read|write|watch|"
        r"listen|touch|hurt|cry|laugh|love|hate|miss|trust"
    )
    text = re.sub(rf"\bdon\s+({common_verbs})\b", r"don't \1", text, flags=re.IGNORECASE)

    # -ing endings
    text = re.sub(r"\bdon\s+(\w+ing)\b", r"don't \1", text, flags=re.IGNORECASE)
    # -ed endings
    text = re.sub(r"\bdon\s+(\w+ed)\b", r"don't \1", text, flags=re.IGNORECASE)

    # Same for "won"
    text = re.sub(r"\bwon\s+(\w+ing|\w+ed|believe|think|know|want|need|like|go|do|be|have)\b",
                  r"won't \1", text, flags=re.IGNORECASE)

    # ORPHAN FIX: "don" alone → "ain't" (CHARACTER!)
    # "don" at end of text
    text = re.sub(r"\bdon\s*$", "ain't", text, flags=re.IGNORECASE)
    # "don" before punctuation
    text = re.sub(r"\bdon(?=[.,!?])", "ain't", text, flags=re.IGNORECASE)
    # "don" before preposition/article
    text = re.sub(
        r"\bdon\s+(of|the|a|an|to|for|with|from|about|by|on|in|at|my|your|his|her|their)\b",
        r"ain't \1", text, flags=re.IGNORECASE
    )

    # Same for "won" orphan
    text = re.sub(r"\bwon\s*$", "ain't", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwon(?=[.,!?])", "ain't", text, flags=re.IGNORECASE)

    return text


def fix_possessive_vs_contraction(text: str) -> str:
    """
    Fix possessive vs contraction confusion.

    "its going" → "it's going" (it is)
    "it's wings" → "its wings" (possessive)
    """
    # "its" + verb → "it's" + verb
    verb_patterns = [
        (r'\bits\s+(going|been|got|coming|done|always|never|really|still|just|about)\b', r"it's \1"),
    ]
    for pattern, replacement in verb_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # "it's" + body/possession noun → "its" + noun
    apos = "['\u2019]"  # Both ASCII and fancy apostrophe
    possessive_patterns = [
        (rf"\bit{apos}s\s+(wings?|eyes?|arms?|legs?|hands?|feet|head|face|body|heart|soul|mind|purpose)\b", r"its \1"),
    ]
    for pattern, replacement in possessive_patterns:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def remove_word_repetitions(
    text: str,
    preserve_regions: Optional[List[Tuple[int, int, str]]] = None
) -> str:
    """
    Remove accidental word repetitions.

    "the the" → "the"
    But preserve intentional: "love, love, love" (has punctuation)
    """
    if preserve_regions is None:
        preserve_regions = []

    # Handle triple+ repetition (almost always error)
    text = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)

    # Handle two-word phrase repetitions
    # "the haze the haze" → "the haze"
    def remove_phrase_if_not_preserved(match):
        phrase = match.group(1)
        if ',' in match.group(0):
            return match.group(0)  # Keep comma-separated
        if _is_in_preserve_region(match.start(), preserve_regions):
            return match.group(0)
        return phrase

    text = re.sub(r'\b(\w+\s+\w+)\s+\1\b', remove_phrase_if_not_preserved, text, flags=re.IGNORECASE)

    # Handle double repetition (careful)
    def remove_if_not_preserved(match):
        word = match.group(1)
        full_match = match.group(0)
        if ',' in full_match or ';' in full_match:
            return full_match  # Poetic repetition
        if _is_in_preserve_region(match.start(), preserve_regions):
            return full_match
        return word

    text = re.sub(r'\b(\w+)\s+\1\b', remove_if_not_preserved, text, flags=re.IGNORECASE)

    return text


def fix_sentence_boundaries(text: str) -> str:
    """Fix sentence boundary issues."""
    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Capitalize "I" when standalone
    text = re.sub(r'\bi\b', 'I', text)

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
        prev = sentences[i-1].rstrip() if i > 0 else ""
        should_cap = (i == 0) or (prev and prev[-1] in '.!?')

        if should_cap and part and part[0].isalpha():
            part = part[0].upper() + part[1:]
        result.append(part)

    return ''.join(result)


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace."""
    text = re.sub(r'[\t\r]', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    return text


def remove_em_dashes(text: str) -> str:
    """
    Remove em-dashes from output.

    Philosophy (from Haze): Stanley is PRESENCE, not dialogue.
    No "— Trade secret." style. Makes speech cleaner.
    """
    text = re.sub(r'\s*—\s*', ' ', text)  # Em-dash
    text = re.sub(r'\s*–\s*', ' ', text)  # En-dash
    text = re.sub(r'\s{2,}', ' ', text)   # Clean double spaces
    return text


def detect_loop(text: str, min_length: int = 10) -> bool:
    """
    Detect if text is stuck in a generation loop.

    Returns True if repetitive pattern detected.
    """
    if len(text) < min_length * 2:
        return False

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

    for end_char in '.!?':
        idx = text.rfind(end_char, 0, max_length)
        if idx > max_length * 0.5:
            return text[:idx + 1]

    for sep in [',', ';', ':', '—', '-']:
        idx = text.rfind(sep, 0, max_length)
        if idx > max_length * 0.7:
            return text[:idx + 1]

    idx = text.rfind(' ', 0, max_length)
    if idx > 0:
        return text[:idx]

    return text[:max_length]


def calculate_garbage_score(text: str) -> float:
    """
    Calculate how much "garbage" (noise) is in text.

    Returns:
        Float 0.0-1.0, where higher means more garbage
    """
    if not text:
        return 0.0

    garbage_patterns = [
        r'\.[,?\.]{2,}',
        r'\?[.,]{2,}',
        r',[.,]{2,}',
        r'\s+[,\.]\s+[,\.]',
        r'\.{5,}',
        r'\s{3,}',
        r'\b[a-z]\s+[a-z]\s+[a-z]\b',
    ]

    total_garbage = 0
    for pattern in garbage_patterns:
        matches = re.findall(pattern, text)
        total_garbage += len(matches)

    text_len = max(len(text), 1)
    score = min(1.0, (total_garbage * 100) / text_len)

    return score


def cleanup_with_resonance(
    text: str,
    resonance_score: Optional[float] = None,
    entropy: Optional[float] = None
) -> str:
    """
    Cleanup with resonance-aware mode selection.

    High resonance + high entropy = preserve more (emergent creativity)
    Low resonance + low entropy = clean more (mechanical output)
    """
    mode = "gentle"
    preserve_resonance = True

    if resonance_score is not None and entropy is not None:
        if resonance_score > 0.7 and entropy > 2.5:
            mode = "gentle"
            preserve_resonance = True
        elif resonance_score < 0.4 or entropy < 1.5:
            mode = "moderate"
            preserve_resonance = False
        else:
            mode = "gentle"
            preserve_resonance = True

    return cleanup_output(text, mode=mode, preserve_resonance=preserve_resonance)

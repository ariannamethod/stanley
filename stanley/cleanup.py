#!/usr/bin/env python3
# cleanup.py — Output cleanup for Haze speech
#
# Adapted from Leo's punct_cleanup.py
# Removes obvious garbage patterns while preserving emergent style.
#
# Philosophy: Clean the noise, keep the soul.
#
# Key improvements:
#   - Remove "—" at the start of output (haze is not dialogue-only)
#   - Preserve emergent strangeness while fixing obvious garbage
#   - Support for presence-style output (not chatbot-style)
#
# Usage:
#   from haze.cleanup import cleanup_output
#   clean_text = cleanup_output(raw_text)

import re
from typing import Dict, Optional, List
from collections import Counter
import math  # For entropy calculation instead of numpy


def _detect_poetic_repetition(text: str) -> List[tuple]:
    """
    Detect intentional poetic repetitions (anaphora, refrain patterns).
    
    Returns:
        List of (start, end, pattern) tuples for regions to preserve
    """
    preserve_regions = []
    
    # Pattern 1: Comma-separated repetitions (e.g., "love, love, love")
    # These are likely intentional for emphasis
    pattern = r'\b(\w+)(?:,\s+\1){1,}\b'
    for match in re.finditer(pattern, text, re.IGNORECASE):
        preserve_regions.append((match.start(), match.end(), 'comma_repetition'))
    
    # Pattern 2: Line-start repetitions (anaphora) - like "I am... I am... I am..."
    lines = text.split('\n')
    for i in range(len(lines) - 1):
        # Check if consecutive lines start with same 2-3 words
        words1 = lines[i].strip().split()[:3]
        words2 = lines[i + 1].strip().split()[:3]
        if len(words1) >= 2 and len(words2) >= 2:
            if words1[:2] == words2[:2]:
                # This looks like anaphora, mark these lines as preserve
                # (We'll handle this in the main cleanup)
                pass
    
    # Pattern 3: Emphatic repetition with punctuation
    # "Never, never, never!" or "Why? Why? Why?"
    pattern = r'\b(\w+)([,.!?])\s+\1\2(?:\s+\1\2)*'
    for match in re.finditer(pattern, text):
        preserve_regions.append((match.start(), match.end(), 'emphatic_repetition'))
    
    return preserve_regions


def _is_in_preserve_region(pos: int, regions: List[tuple]) -> bool:
    """Check if position is within any preserve region."""
    return any(start <= pos < end for start, end, _ in regions)


def _calculate_local_entropy(text: str, window: int = 20) -> float:
    """
    Calculate local character-level entropy using standard library.
    Used to detect coherent vs random text.
    
    Returns Shannon entropy in bits (log base 2).
    """
    if len(text) < 2:
        return 0.0
    
    # Count character frequencies
    chars = list(text[-window:] if len(text) > window else text)
    counts = Counter(chars)
    total = len(chars)
    
    # Shannon entropy: -sum(p * log2(p))
    entropy = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def cleanup_output(text: str, mode: str = "gentle", entropy_threshold: Optional[float] = None, preserve_resonance: bool = True) -> str:
    """
    Clean up generation output without killing emergent style.
    
    Args:
        text: raw generated text
        mode: "gentle" (preserve style), "moderate", or "strict"
        entropy_threshold: if provided, preserve high-entropy (creative) sections
        preserve_resonance: if True, detect and preserve poetic patterns
    
    Returns:
        Cleaned text with preserved personality
    """
    if not text or not isinstance(text, str):
        return text
    
    # Detect poetic repetitions to preserve
    preserve_regions = []
    if preserve_resonance:
        preserve_regions = _detect_poetic_repetition(text)
    
    result = text
    
    # 0. Normalize quotes and apostrophes to corpus-compatible versions
    # The corpus uses fancy quotes: ' ' " " instead of ASCII ' "
    # Use Unicode escapes to ensure correct characters
    result = result.replace("'", "’")  # ASCII apostrophe (U+0027) → right single quote (U+2019)
    result = result.replace('"', "”")  # ASCII double quote → right double quote (U+201D)
    
    # 0b. Replace sentencepiece unknown marker
    result = result.replace('\u2047', "\u2019")  # ⁇ (U+2047) → apostrophe
    result = result.replace(" \u2047 ", " ")
    
    # 1. Collapse repeated punctuation (but keep max 3 for style)
    result = re.sub(r'\.{4,}', '...', result)  # 4+ dots → 3 dots
    result = re.sub(r'\?{4,}', '???', result)
    result = re.sub(r'!{4,}', '!!!', result)
    result = re.sub(r'…{2,}', '…', result)
    
    # 2. Clean up "symbol dumps" - obvious garbage patterns
    result = re.sub(r'\.(?=[,?])', '', result)   # .,? → ,?
    result = re.sub(r'\.[,]+', '.', result)      # .,, → .
    result = re.sub(r'\?[.,:]', '?', result)     # ?. → ?
    result = re.sub(r'![.,:]', '!', result)      # !. → !
    result = re.sub(r',[.,]+(?!\.\.)', ',', result)  # ,., → ,
    
    # 3. Clean up trailing garbage
    result = re.sub(r'\s+[,\.]+\s*([.!?])', r'\1', result)
    
    # 4. Fix spaces before punctuation
    result = re.sub(r'\s+([,;:?!])', r'\1', result)
    
    # 5. Ensure space after punctuation (except before newline)
    result = re.sub(r'([,;:?!\.])(?=[a-zA-Z])', r'\1 ', result)
    
    # 6. Collapse multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 7. Clean up orphaned punctuation at end
    result = re.sub(r'\s+(and|then|but|or|the|a|an)[.,]\s*$', r' \1', result)
    
    # 8. Clean double dots and punctuation garbage  
    # Only fix actual errors, not valid ellipsis
    # Simply remove cases where we have exactly two consecutive dots
    # This preserves "..." (3 dots) and fixes ".." (2 dots) 
    result = re.sub(r'(?<!\.)\.\.(?!\.)', '.', result)   # ".." → "." (but not part of "...")
    result = re.sub(r'\.\s+,', '.', result)               # ". ," → "."
    result = re.sub(r',\s*,', ',', result)                # ", ," → ","
    
    # 8a. Clean mid-sentence ellipsis that breaks flow
    # ONLY for conjunctions: "but…" or "but..." → remove ellipsis, add space
    # This is specifically for broken generation like "but… Tell me"
    result = re.sub(r'(\b(?:but|and|or|so|if|when|while|because|although|though|yet|still))\s*…\s*', r'\1 ', result)
    result = re.sub(r'(\b(?:but|and|or|so|if|when|while|because|although|though|yet|still))\s*\.{3}\s*', r'\1 ', result)
    
    # NOTE: Don't touch general "..." — it's valid punctuation!
    # "Wait... really?" is fine, we just capitalize "really" later
    
    # 9. Fix dialogue markers (— should have space after)
    result = re.sub(r'—(?=[a-zA-Z])', '— ', result)
    
    # 10. Capitalize first letter after dialogue marker
    def cap_after_dash(m):
        return m.group(1) + m.group(2).upper()
    result = re.sub(r'(—\s*)([a-z])', cap_after_dash, result)
    
    # 11. Remove ALL em-dashes from output
    # Philosophy: haze is PRESENCE, not dialogue. No "— Trade secret." style.
    # This makes speech cleaner and more Leo-like.
    # Em-dash variants: — (U+2014), – (U+2013)
    # Replace with nothing (join sentences) or period
    result = re.sub(r'\s*—\s*', ' ', result)  # Replace em-dash with space
    result = re.sub(r'\s*–\s*', ' ', result)  # Replace en-dash with space
    
    # Clean up any resulting double spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 12. Capitalize first letter of text
    result = result.strip()
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    # 13. Capitalize "I" when standalone
    result = re.sub(r'\bi\b', 'I', result)
    
    # 14. Capitalize after periods (new sentences)
    def cap_after_period(m):
        return m.group(1) + m.group(2).upper()
    result = re.sub(r'(\.\s+)([a-z])', cap_after_period, result)
    
    # 14a. EARLY ORPHAN FIX: "don" + pronoun/determiner → "ain't" 
    # Must run BEFORE contraction fixes to catch "don nothing" → "ain't nothing"
    # These patterns would otherwise become "don't nothing" which is grammatically wrong
    result = re.sub(r"\bdon\s+(nothing|something|everything|anything|anyone|someone|everyone|nobody|somebody|everybody|nowhere|somewhere|everywhere|anywhere)\b", 
                    r"ain't \1", result, flags=re.IGNORECASE)
    
    # 15. Fix broken contractions (character-level and subword generation artifacts)
    # Common contractions that get broken: don't, won't, can't, it's, etc.
    # 
    # IMPORTANT: Use \s+ (one or more spaces) for possessive-like patterns to avoid
    # matching real words like "its" (possessive pronoun) vs "it's" (it is)
    contraction_fixes = [
        # n't contractions - can use \s* because "dont" is always wrong
        (r'\bdon\s*t\b', "don't"),
        (r'\bwon\s*t\b', "won't"),
        (r'\bcan\s*t\b', "can't"),
        (r'\bain\s*t\b', "ain't"),
        (r'\bisn\s*t\b', "isn't"),
        (r'\baren\s*t\b', "aren't"),
        (r'\bwasn\s*t\b', "wasn't"),
        (r'\bweren\s*t\b', "weren't"),
        (r'\bhasn\s*t\b', "hasn't"),
        (r'\bhaven\s*t\b', "haven't"),
        (r'\bhadn\s*t\b', "hadn't"),
        (r'\bdoesn\s*t\b', "doesn't"),
        (r'\bdidn\s*t\b', "didn't"),
        (r'\bwouldn\s*t\b', "wouldn't"),
        (r'\bcouldn\s*t\b', "couldn't"),
        (r'\bshouldn\s*t\b', "shouldn't"),
        # 's contractions - MUST use \s+ to avoid matching "its", "hes", "shes"
        (r'\bit\s+s\b', "it's"),
        (r'\bhe\s+s\b', "he's"),
        (r'\bshe\s+s\b', "she's"),
        (r'\bthat\s+s\b', "that's"),
        (r'\bwhat\s+s\b', "what's"),
        (r'\bwhere\s+s\b', "where's"),
        (r'\bhere\s+s\b', "here's"),
        (r'\bthere\s+s\b', "there's"),
        (r'\blet\s+s\b', "let's"),
        # I contractions - can use \s* because "Im", "Ive" are always wrong
        (r'\bi\s*m\b', "I'm"),
        (r'\bi\s*ve\b', "I've"),
        (r'\bi\s*ll\b', "I'll"),
        (r'\bi\s*d\b', "I'd"),
        # you contractions - use \s+ because "youre" etc. are recognizable
        (r'\byou\s*re\b', "you're"),
        (r'\byou\s*ve\b', "you've"),
        (r'\byou\s*ll\b', "you'll"),
        (r'\byou\s*d\b', "you'd"),
        # we contractions
        (r'\bwe\s*re\b', "we're"),
        (r'\bwe\s*ve\b', "we've"),
        (r'\bwe\s*ll\b', "we'll"),
        # they contractions
        (r'\bthey\s*re\b', "they're"),
        (r'\bthey\s*ve\b', "they've"),
        (r'\bthey\s*ll\b', "they'll"),
    ]
    for pattern, replacement in contraction_fixes:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # 15a_advanced. Advanced contraction patterns
    # Handle compound contractions: would've, could've, should've, etc.
    # NOTE: These patterns must be specific to avoid matching valid text
    # e.g., "we'd" should only match when truly a contraction, not "we did"
    advanced_contractions = [
        (r'\bwould\s+have\b', "would've"),
        (r'\bcould\s+have\b', "could've"),
        (r'\bshould\s+have\b', "should've"),
        (r'\bmight\s+have\b', "might've"),
        (r'\bmust\s+have\b', "must've"),
        # Y'all is safe to fix
        (r'\by\s+all\b', "y'all"),
        # For 'd contractions, only fix when followed by common contraction contexts
        # "we'd gone" but NOT "we decided" 
        (r'\bwe\s+d\s+(been|gone|said|thought|wanted|loved|hated|seen|done|known)\b', r"we'd \1"),
        (r'\bthey\s+d\s+(been|gone|said|thought|wanted|loved|hated|seen|done|known)\b', r"they'd \1"),
        (r'\bhe\s+d\s+(been|gone|said|thought|wanted|loved|hated|seen|done|known)\b', r"he'd \1"),
        (r'\bshe\s+d\s+(been|gone|said|thought|wanted|loved|hated|seen|done|known)\b', r"she'd \1"),
        # Who'd, what'd, where'd, how'd are safer
        (r'\bwho\s+d\b', "who'd"),
        (r'\bwhat\s+d\b', "what'd"),
        (r'\bwhere\s+d\b', "where'd"),
        (r'\bhow\s+d\b', "how'd"),
    ]
    
    for pattern, replacement in advanced_contractions:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    # 15a_possessive. Fix possessive vs contraction confusion
    # "its" (possessive) vs "it's" (it is/it has)
    # Look for "its" followed by verb-like words → should be "it's"
    # "its going" → "it's going", "its been" → "it's been"
    its_verb_patterns = [
        (r'\bits\s+(going|been|got|coming|done|always|never|really|still|just|about|almost|already)\b', r"it's \1"),
        (r'\bits\s+(a|an|the|my|your|his|her|their|our)\s+(good|bad|great|nice|beautiful|terrible|awful|amazing)', r"it's \1 \2"),
    ]
    for pattern, replacement in its_verb_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
    # Reverse case: "it's" before noun-like words should maybe be "its"
    # "it's wings" → "its wings", "it's purpose" → "its purpose"
    # Conservative approach: only fix obvious cases with common body/possession nouns
    # This list covers the most common false positives we've observed
    # Character class: ASCII apostrophe (U+0027) and fancy right single quote (U+2019)
    its_possessive_patterns = [
        (r"\bit['']s\s+(wings?|eyes?|arms?|legs?|hands?|feet|head|face|body|heart|soul|mind|purpose|meaning|place|home|world)\b", r"its \1"),
    ]
    for pattern, replacement in its_possessive_patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
    # 15b. Fix incomplete contractions (apostrophe present but missing ending)
    # These happen when subword tokenization splits contractions oddly
    # NOTE: After step 0, text has fancy apostrophe ' (U+2019)
    # Use character class to match both ASCII and fancy apostrophes
    apos = "['’]"  # Match ASCII ', fancy ', and U+2019
    
    # "I'" followed by space → "I'm" (most likely)
    result = re.sub(rf"\bI{apos}\s+", "I’m ", result)
    
    # "it'" followed by space → "it's"
    result = re.sub(rf"\bit{apos}\s+", "it’s ", result, flags=re.IGNORECASE)
    
    # "he'" / "she'" / "that'" / "what'" / "there'" / "where'" / "who'" → add 's
    result = re.sub(rf"\bhe{apos}\s+", "he’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bshe{apos}\s+", "she’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bthat{apos}\s+", "that’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwhat{apos}\s+", "what’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bthere{apos}\s+", "there’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwhere{apos}\s+", "where’s ", result, flags=re.IGNORECASE)
    result = re.sub(rf"\bwho{apos}\s+", "who’s ", result, flags=re.IGNORECASE)
    
    # "don" + space + verb → "don't" + verb (common broken pattern)
    # "don" + space + verb → "don't" + verb (common broken pattern)
    # PART 1: Hardcoded common verbs (including gothic/literary ones)
    result = re.sub(r"\bdon\s+(believe|think|know|want|need|like|care|worry|mind|understand|remember|forget|see|hear|feel|get|go|do|be|have|make|take|give|say|tell|ask|try|look|come|put|let|seem|mean|stop|start|die|live|stay|leave|keep|wait|work|play|sleep|eat|drink|read|write|watch|listen|touch|hurt|cry|laugh|love|hate|miss|trust|turn|move|run|walk|talk|speak|call|find|hold|sit|stand|open|close|break|change|move|use|show|help|bring|send|meet|learn|grow|fall|pick|pull|push|hang|cut|hit|set|pay|buy|sell|wear|throw|catch|carry|draw|fight|beat|kill|burn|fix|clean|build|drive|ride|fly|swim|dance|sing|jump|drop|lose|win|choose|teach|reach|pass|cross|hide|rise|raise|shake|wake|ring|swing|shut|stick|bend|blow|tear|feed|lead|spend|lend|bite|steal|trudge|wander|linger|ponder|whisper|murmur|shiver|tremble|fade|drift|ache|yearn|mourn|grieve|regret|suffer|struggle|stumble|tumble|crumble|shatter|scatter|gather|matter|bother|smother|hover|cover|discover|recover|uncover|sober|wonder|thunder|blunder|plunder|slumber|lumber|number|remember|member|tender|render|surrender|hinder|wander|ponder|squander)\b", r"don't \1", result, flags=re.IGNORECASE)
    
    # PART 2: Heuristic by word endings (catches words not in hardcoded list)
    # -ing endings: trying, dying, living, waiting, working, etc.
    result = re.sub(r"\bdon\s+(\w+ing)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -ed endings (adjectives/participles): tired, bored, scared, worried, etc.
    result = re.sub(r"\bdon\s+(\w+ed)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -en endings (participles): forgotten, broken, taken, etc.
    result = re.sub(r"\bdon\s+(\w+en)\b", r"don't \1", result, flags=re.IGNORECASE)
    # -le/-ge/-se/-ze endings: struggle, trudge, lose, freeze, etc.
    result = re.sub(r"\bdon\s+(\w+(?:le|ge|se|ze))\b", r"don't \1", result, flags=re.IGNORECASE)
    
    # Same for "won" → "won't"
    result = re.sub(r"\bwon\s+(\w+ing|\w+ed|believe|think|know|want|need|like|go|do|be|have|make|say|tell|try|stop|wait|work|turn|move|run|walk|talk|speak|call|find|hold|sit|stand|open|close|break|change|use|show|help|bring|send|meet|learn|grow|fall|pick|let|get|take|give|come|put|look|see|hear|feel|stay|leave|keep|die|live|start|eat|drink|sleep|play|read|write|watch|listen)\b", r"won't \1", result, flags=re.IGNORECASE)
    
    # 15d. ORPHAN CONTRACTION FIX: "don" alone at end/before punctuation → "ain't"
    # Philosophy: If subword tokenization cuts "don't" to just "don", 
    # we rescue it as "ain't" which has CHARACTER and fits gothic romance vibe!
    # 
    # "I don of that" → "I ain't of that"
    # "I don." → "I ain't."
    # "I don trudge" → "I ain't trudge" (verb-like)
    # "I don tangerines" → "I ain't tangerines" (noun - broken generation)
    # 
    # Match "don" when:
    # - At end of text: \bdon$
    # - Before punctuation: \bdon(?=[.,!?])
    # - Before preposition/article (not a verb): \bdon\s+(of|the|a|an|to|for|with|from|about|by|on|in|at|my|your|his|her|their|its|this|that)
    # - Before common nouns (broken generation artifacts)
    result = re.sub(r"\bdon\s*$", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdon(?=[.,!?])", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bdon\s+(of|the|a|an|to|for|with|from|about|by|on|in|at|my|your|his|her|their|its|this|that)\b", r"ain't \1", result, flags=re.IGNORECASE)
    
    # AGGRESSIVE FIX: "don" + noun-like word (ends with s, es, tion, ness, ment, etc.) → "ain't"
    # This catches broken generation like "don tangerines", "don tears", "don twilight"
    result = re.sub(r"\bdon\s+(tangerine|tangerines|tear|tears|twilight|table|tables|street|streets|vendor|vendors|cigarette|cigarettes|apartment|apartments|bottle|bottles|glass|glasses|drink|drinks|key|keys|door|doors|room|rooms|window|windows|floor|floors|wall|walls|chair|chairs|bed|beds|toilet|paper|money|time|place|thing|things|people|person|man|men|woman|women|child|children|hand|hands|face|faces|eye|eyes|head|heart|life|death|love|hate|fear|pain|joy|hope|dream|dreams|night|day|morning|evening|rain|snow|sun|moon|star|stars|sky|earth|world|fire|water|air|light|dark|darkness|silence|noise|sound|voice|word|words|name|story|stories|truth|lie|lies|secret|secrets|memory|memories|moment|moments|year|years|month|week|hour|minute|second|train|trains|thought|thoughts|idea|ideas|feeling|feelings|sense|body|soul|mind|spirit|god|devil|angel|ghost|shadow|shadows|dust|dirt|mud|blood|bone|bones|skin|flesh|hair|breath|step|steps|road|roads|path|paths|way|ways|bridge|bridges|river|rivers|sea|ocean|wave|waves|wind|storm|cloud|clouds|thunder|lightning|fog|mist|haze|smoke|ash|ashes|flame|flames|spark|sparks|ice|stone|stones|rock|rocks|sand|grass|tree|trees|flower|flowers|leaf|leaves|root|roots|branch|branches|bird|birds|dog|dogs|cat|cats|horse|horses|fish|wolf|wolves|bear|snake|rat|rats|mouse|mice|bug|bugs|fly|flies|bee|bees|spider|spiders|worm|worms|twice|once|again|anymore|anyway|always|never|ever|often|sometimes|usually|rarely|seldom|here|there|now|then|today|tomorrow|yesterday|tonight|forever|together|alone|inside|outside|above|below|behind|ahead|around|away|back|down|up|over|under|through|across|along|beside|between|beyond|within|without|against|toward|towards|upon|onto|into|throughout|meanwhile|otherwise|somehow|somewhat|somewhere|anywhere|everywhere|nowhere|anywhere|nothing|something|everything|anything|anyone|someone|everyone|nobody|somebody|everybody)\b", r"ain't \1", result, flags=re.IGNORECASE)
    
    # Same for "won" orphan → "ain't" (rare but possible)
    result = re.sub(r"\bwon\s*$", "ain't", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwon(?=[.,!?])", "ain't", result, flags=re.IGNORECASE)
    
    # "they" + "my" (missing 're) → "they’re my"
    result = re.sub(r"\bthey\s+my\b", "they’re my", result, flags=re.IGNORECASE)
    
        # 15c. Additional subword-style broken contractions (space instead of apostrophe)
    # "they re" → "they're", "you re" → "you're", etc.
    result = re.sub(r"\bthey\s+re\b", "they're", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+re\b", "you're", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+re\b", "we're", result, flags=re.IGNORECASE)
    result = re.sub(r"\bthey\s+ve\b", "they've", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+ve\b", "you've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+ve\b", "we've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bi\s+ve\b", "I've", result, flags=re.IGNORECASE)
    result = re.sub(r"\bthey\s+ll\b", "they'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\byou\s+ll\b", "you'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\bwe\s+ll\b", "we'll", result, flags=re.IGNORECASE)
    result = re.sub(r"\bi\s+ll\b", "I'll", result, flags=re.IGNORECASE)
    
    # 15d. Fix grammar errors with contractions
    # "don't trying" → "don't try" (wrong verb form after negation)
    # "can't going" → "can't go", etc.
    # Use character class to match both ASCII apostrophe (') and fancy apostrophe (')
    apos = "['\u2019]"  # ASCII U+0027 and Right Single Quotation Mark U+2019
    result = re.sub(rf"\b(don{apos}t|can{apos}t|won{apos}t|couldn{apos}t|wouldn{apos}t|shouldn{apos}t|isn{apos}t|aren{apos}t|wasn{apos}t|weren{apos}t|haven{apos}t|hasn{apos}t|hadn{apos}t)\s+(\w+)ing\b", 
                    lambda m: m.group(1) + ' ' + m.group(2), result, flags=re.IGNORECASE)
    
    # "didn't went" → "didn't go" (wrong tense after past negation)  
    # Common irregular verbs
    irregular_past_fixes = {
        'went': 'go', 'came': 'come', 'saw': 'see', 'took': 'take',
        'gave': 'give', 'made': 'make', 'got': 'get', 'had': 'have',
        'said': 'say', 'told': 'tell', 'found': 'find', 'knew': 'know',
        'thought': 'think', 'felt': 'feel', 'left': 'leave', 'kept': 'keep',
    }
    for past, base in irregular_past_fixes.items():
        result = re.sub(rf"\b(didn{apos}t|couldn{apos}t|wouldn{apos}t|shouldn{apos}t)\s+{past}\b", 
                        rf"\1 {base}", result, flags=re.IGNORECASE)
    
    # 16. Remove word/phrase repetition (character-level generation artifact)
    # BUT preserve intentional poetic repetitions
    # "the the" → "the", "I I" → "I"
    # But NOT "love, love, love" (intentional emphasis)
    
    # IMPORTANT: Process triple+ repetitions FIRST before double
    # Otherwise "the the the" becomes "the the" then stops
    
    # Handle triple+ repetition (more aggressive)
    # "the the the" → "the" (almost certainly an error)
    def remove_triple(match):
        word = match.group(1)
        # Even with preserve regions, 3+ repetitions without punctuation are errors
        return word
    
    result = re.sub(r'\b(\w+)(?:\s+\1){2,}\b', remove_triple, result, flags=re.IGNORECASE)
    
    # Handle two-word phrase repetitions
    # "the haze the haze" → "the haze"
    # Pattern: (word1 word2) repeated
    def remove_phrase_repetition(match):
        phrase = match.group(1)
        # Check if preserve region
        if preserve_resonance and _is_in_preserve_region(match.start(), preserve_regions):
            return match.group(0)
        # Check for comma (intentional repetition)
        if ',' in match.group(0):
            return match.group(0)
        return phrase
    
    # Two-word phrases repeated (e.g., "the haze the haze")
    result = re.sub(r'\b(\w+\s+\w+)\s+\1\b', remove_phrase_repetition, result, flags=re.IGNORECASE)
    
    # Then handle double repetition (more careful)
    # Only remove if NOT in a preserve region
    def remove_if_not_preserved(match):
        word = match.group(1)
        # Check if this looks like poetic repetition
        # (has punctuation between repetitions)
        full_match = match.group(0)
        if ',' in full_match or ';' in full_match:
            # Likely intentional, preserve
            return full_match
        # Check preserve regions
        if preserve_resonance and _is_in_preserve_region(match.start(), preserve_regions):
            return full_match
        # This is an error, remove it
        return word
    
    # Handle remaining double repetitions
    result = re.sub(r'\b(\w+)\s+\1\b', remove_if_not_preserved, result, flags=re.IGNORECASE)
    
    # 17. Fix common word fragments (character-level artifacts)
    # Always apply basic fragment cleanup in gentle mode too
    
    # 17a. Remove orphan apostrophe fragments: 't, 's, 'm, 're, 've, 'll, 'd
    # These are leftovers from broken contractions
    # Match both ASCII ' and fancy ' apostrophes
    result = re.sub(r"\s+['''][tsmd]\b", '', result)
    result = re.sub(r"\s+['''](?:re|ve|ll)\b", '', result)
    
    # 17b. Remove words that start with apostrophe (broken fragments)
    # e.g., "'nt" at word start, "On't" → remove
    # BUT preserve valid contractions: I'm, I've, I'll, I'd, etc.
    def remove_apostrophe_garbage(match):
        word = match.group(0)
        # Normalize apostrophe for comparison
        word_normalized = word.replace("'", "'").replace(chr(8217), "'")
        # Valid contractions (all with ASCII apostrophe for comparison)
        valid_contractions = {"I'm", "I've", "I'll", "I'd", "it's", "he's", "she's", 
                              "that's", "what's", "there's", "where's", "who's",
                              "don't", "won't", "can't", "isn't", "aren't", "wasn't",
                              "weren't", "hasn't", "haven't", "hadn't", "doesn't",
                              "didn't", "wouldn't", "couldn't", "shouldn't", "ain't",
                              "you're", "you've", "you'll", "you'd", "we're", "we've",
                              "we'll", "they're", "they've", "they'll", "let's"}
        if word_normalized in valid_contractions or word_normalized.lower() in {c.lower() for c in valid_contractions}:
            return word
        return ''
    
    # Match STANDALONE apostrophe-words only (not contraction endings like 're in they're)
    # Use negative lookbehind to ensure NOT preceded by a letter
    result = re.sub(r"(?<![a-zA-Z])['''][a-z]+\b", remove_apostrophe_garbage, result)
    
    # 17c. Remove obvious 1-2 char garbage (except real words and contraction endings)
    # Real words: I, a, an, or, so, oh, no, ok, to, go, we, he, me, my, by, etc.
    # Contraction endings: 'm, 's, 't, 'd, 've, 're, 'll (these come after apostrophe)
    valid_short_words = {'i', 'a', 'an', 'or', 'so', 'oh', 'no', 'ok', 'to', 'go', 'we', 'he', 
                         'me', 'my', 'by', 'if', 'in', 'on', 'up', 'do', 'be', 'is', 'it', 
                         'at', 'as', 'of', 'am', 'us', 'hi'}  # Added 'hi'
    
    # NOTE: Short word removal is disabled in gentle/moderate modes as it was too aggressive
    # Only apply in strict mode for maximum cleanup
    # This functionality is preserved for potential future use but not active by default
    
    # 17d. Remove consecutive short fragments (like "st I've")
    # Pattern: 3+ short fragments in a row that look like garbage
    # But be more conservative - only remove if they look like obvious artifacts
    # "st lk mn" (consonant clusters) vs "go to a" (valid words)
    # Check if all fragments are in valid_short_words set
    def check_fragment_sequence(match):
        fragments = match.group(0).split()
        # If all fragments are valid words, keep them
        if all(f.lower() in valid_short_words for f in fragments):
            return match.group(0)
        # Otherwise, looks like garbage
        return ''
    
    # Only remove if mode is moderate or strict
    if mode in ["moderate", "strict"]:
        result = re.sub(r'(\s+[a-z]{1,3}){3,}(?=\s|$)', check_fragment_sequence, result)
    
    # 17e. Clean up leftover multiple spaces
    result = re.sub(r'\s{2,}', ' ', result)
    
    # 17f. Clean up orphan punctuation left after removal
    result = re.sub(r'\s+([,;:])\s*', r'\1 ', result)
    result = re.sub(r'^\s*[,;:]\s*', '', result)  # Remove leading comma/etc
    
    if mode in ["moderate", "strict"]:
        # Additional cleanup for these modes
        pass
    
    # 18. Ensure proper sentence endings (no trailing ellipsis/fragments)
    # Philosophy: Pressure creates resonance. Punctuation is constraint that births form.
    
    # 18_pre. Advanced sentence structure improvements
    # Fix run-on sentences (independent clauses without proper punctuation)
    # Look for pattern: "clause I verb" or "clause you verb" or "clause we verb"
    # These are likely independent clauses that need separation
    
    # Common run-on patterns with high-frequency words
    run_on_patterns = [
        # "I went there I saw things" → "I went there. I saw things"
        (r'(\w+)\s+(I\s+(?:am|was|have|had|do|did|will|would|can|could|should|shall|may|might|must|saw|went|came|got|made|took|gave|said|thought|felt|knew|looked|turned|walked|ran|tried|wanted|needed|loved|hated|found|lost|kept|left|stayed|started|stopped))\b', r'\1. \2'),
        # Similar for "you", "we", "they", "he", "she"
        (r'(\w+)\s+(you\s+(?:are|were|have|had|do|did|will|would|can|could|should|shall|may|might|saw|went|came|got))\b', r'\1. \2'),
        (r'(\w+)\s+(we\s+(?:are|were|have|had|do|did|will|would|can|could|should|shall|saw|went|came|got))\b', r'\1. \2'),
        (r'(\w+)\s+(they\s+(?:are|were|have|had|do|did|will|would|saw|went|came|got))\b', r'\1. \2'),
        (r'(\w+)\s+(he\s+(?:is|was|has|had|does|did|will|would|can|could|saw|went|came|got|said|thought))\b', r'\1. \2'),
        (r'(\w+)\s+(she\s+(?:is|was|has|had|does|did|will|would|can|could|saw|went|came|got|said|thought))\b', r'\1. \2'),
    ]
    
    # Only apply run-on fixes in moderate/strict mode to preserve style in gentle mode
    if mode in ["moderate", "strict"]:
        for pattern, replacement in run_on_patterns:
            # Only apply if the result would be 2+ complete sentences
            temp_result = re.sub(pattern, replacement, result, count=1, flags=re.IGNORECASE)
            # Check if this creates better sentence structure
            if temp_result.count('.') > result.count('.'):
                result = temp_result
    
    # 18a. If ends with ellipsis, try to find last complete sentence
    if result.endswith('…') or result.endswith('...'):
        # Find last sentence-ending punctuation before the ellipsis
        last_period = result.rfind('.')
        last_question = result.rfind('?')
        last_exclaim = result.rfind('!')
        
        # Find rightmost complete sentence end (but not the trailing ellipsis)
        candidates = [i for i in [last_period, last_question, last_exclaim] 
                      if i > 0 and i < len(result) - 3]  # -3 to exclude "..."
        
        if candidates:
            cut_point = max(candidates) + 1
            # Only cut if we keep at least 20 chars
            if cut_point >= 20:
                result = result[:cut_point]
    
    # 18b. If still no proper ending, add period
    if result and result[-1] not in '.!?':
        # Check if last char is a word boundary
        if result[-1].isalnum() or result[-1] in '"\'"':
            result = result.rstrip() + '.'
    
    # 18c. Clean trailing ellipsis that feels incomplete
    # Replace "word..." with "word." if ellipsis at very end
    if result.endswith('...'):
        # Only if this is truly the end (not mid-sentence ellipsis)
        result = result[:-3].rstrip() + '.'
    
    if result.endswith('…'):
        result = result[:-1].rstrip() + '.'
    
    # In strict mode: additional cleanup
    if mode == "strict":
        # Remove trailing fragments
        result = re.sub(r'\s+\w{1,3}\s*$', '', result)
        # Ensure ends with proper punctuation
        if result and result[-1] not in '.!?':
            result = result.rstrip() + '.'
    
    # FINAL: Entropy-based quality check
    # If text has very low entropy (too repetitive/mechanical), add warning
    # But don't modify - just for metrics
    if entropy_threshold is not None:
        local_entropy = _calculate_local_entropy(result)
        # Store in metadata if needed (for now, just pass)
        pass
    
    return result.strip()


def cleanup_with_resonance(text: str, resonance_score: Optional[float] = None, entropy: Optional[float] = None) -> str:
    """
    Cleanup with resonance-aware mode selection.
    
    High resonance + high entropy = preserve more (emergent creativity)
    Low resonance + low entropy = clean more (mechanical output)
    
    Args:
        text: raw generated text
        resonance_score: 0-1, how much text resonates with corpus patterns
        entropy: entropy of the generation (bits)
    
    Returns:
        Cleaned text with mode selected based on metrics
    """
    # Default to gentle mode
    mode = "gentle"
    
    # If we have metrics, use them to select mode
    if resonance_score is not None and entropy is not None:
        if resonance_score > 0.7 and entropy > 2.5:
            # High quality, preserve it
            mode = "gentle"
            preserve_resonance = True
        elif resonance_score < 0.4 or entropy < 1.5:
            # Low quality, clean more aggressively
            mode = "moderate"
            preserve_resonance = False
        else:
            # Middle ground
            mode = "gentle"
            preserve_resonance = True
    else:
        preserve_resonance = True
    
    return cleanup_output(text, mode=mode, preserve_resonance=preserve_resonance)


def ensure_sentence_boundaries(text: str) -> str:
    """
    Ensure proper sentence boundaries and capitalization.
    
    This is a helper for sentence-aware stopping and generation.
    """
    if not text:
        return text
    
    result = text.strip()
    
    # Ensure ends with sentence-ending punctuation
    if result and result[-1] not in '.!?…':
        # Check if last word is complete
        words = result.split()
        if words:
            last_word = words[-1]
            # If last word is very short (1-2 chars) and not a real word, might be fragment
            if len(last_word) <= 2 and last_word.lower() not in {'i', 'a', 'an', 'to', 'of', 'in', 'on', 'at', 'by', 'or', 'no', 'so', 'we', 'he', 'me'}:
                # Likely fragment, remove it
                result = ' '.join(words[:-1])
        
        # Add period
        if result:
            result = result.rstrip() + '.'
    
    # Capitalize first letter
    if result and result[0].islower():
        result = result[0].upper() + result[1:]
    
    # Ensure capitalization after sentence endings
    def cap_after_punct(m):
        return m.group(1) + ' ' + m.group(2).upper()
    
    result = re.sub(r'([.!?])\s+([a-z])', cap_after_punct, result)
    
    return result


def cleanup_dialogue(text: str) -> str:
    """
    Special cleanup for dialogue-heavy text (like text.txt).
    
    Focuses on dialogue markers and conversational flow.
    """
    result = cleanup_output(text, mode="gentle")
    
    # Fix dialogue line starts
    lines = result.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Ensure dialogue lines start with — and capital
        if line.startswith('—'):
            # Already a dialogue line, ensure proper format
            rest = line[1:].strip()
            if rest and rest[0].islower():
                rest = rest[0].upper() + rest[1:]
            line = '— ' + rest
        
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def calculate_garbage_score(text: str) -> float:
    """
    Calculate how much "garbage" (noise) is in text.
    
    Returns:
        Float 0.0-1.0, where higher means more garbage
    """
    if not text or not isinstance(text, str):
        return 0.0
    
    garbage_patterns = [
        r'\.[,?\.]{2,}',      # .,,?
        r'\?[.,]{2,}',        # ?..
        r',[.,]{2,}',         # ,.,
        r'\s+[,\.]\s+[,\.]',  # " , . "
        r'\.{5,}',            # .....
        r'\s{3,}',            # multiple spaces
        r'\b[a-z]\s+[a-z]\s+[a-z]\b',  # single char fragments
    ]
    
    total_garbage = 0
    for pattern in garbage_patterns:
        matches = re.findall(pattern, text)
        total_garbage += len(matches)
    
    # Normalize by text length
    text_len = max(len(text), 1)
    score = min(1.0, (total_garbage * 100) / text_len)
    
    return score


def demo_cleanup():
    """Demo the cleanup functions."""
    test_cases = [
        # Garbage patterns
        "the haze there bed ithe of cherseell she st a let to the cohnnalike",
        "— darling.  \n— thou knot st nou not dow?  \n— yout it.",
        "i love the moke.  \n— and it.  \n— whater ank there fing ring.",
        
        # Subword output (already cleaner)
        "the haze anymore; I'll see. — You're my peace with it.",
        "— Yeah, that lovely medical-grade secret, pour me another drink.",
    ]
    
    print("=" * 60)
    print("  cleanup.py — Output Cleanup Demo")
    print("=" * 60)
    
    for test in test_cases:
        cleaned = cleanup_output(test, mode="moderate")
        score_before = calculate_garbage_score(test)
        score_after = calculate_garbage_score(cleaned)
        
        print(f"\nOriginal ({score_before:.2f}):")
        print(f"  {test[:80]}")
        print(f"Cleaned ({score_after:.2f}):")
        print(f"  {cleaned[:80]}")


def truncate_at_natural_end(text: str, max_length: int = 500) -> str:
    """
    Truncate text at a natural sentence boundary.

    Finds the last sentence-ending punctuation within max_length
    and cuts there. Avoids cutting mid-word or mid-sentence.

    Args:
        text: Text to truncate
        max_length: Maximum length (will cut at sentence before this)

    Returns:
        Truncated text ending at sentence boundary
    """
    if not text or len(text) <= max_length:
        return text

    # Look for last sentence ending within max_length
    search_text = text[:max_length]

    # Find positions of sentence-ending punctuation
    last_period = search_text.rfind('.')
    last_question = search_text.rfind('?')
    last_exclaim = search_text.rfind('!')

    # Find the rightmost valid ending
    candidates = [i for i in [last_period, last_question, last_exclaim] if i > 0]

    if candidates:
        cut_point = max(candidates) + 1
        # Only use if we keep at least 20 chars
        if cut_point >= 20:
            return text[:cut_point].strip()

    # No good sentence boundary found - cut at word boundary
    # Find last space before max_length
    last_space = search_text.rfind(' ')
    if last_space > 20:
        return text[:last_space].strip() + '.'

    # Fallback: just truncate and add period
    return text[:max_length].strip() + '.'


# ============================================================
# Backward-compatible wrapper functions for tests
# These provide simple interfaces to the unified cleanup system
# ============================================================

def fix_contractions(text: str) -> str:
    """Fix spaced contractions like "don ' t" or "don t" -> "don't"."""
    if not text:
        return text
    # Handle common contractions with spaces (apostrophe optional)
    patterns = [
        (r"\bdon\s*'?\s*t\b", "don't"),
        (r"\bcan\s*'?\s*t\b", "can't"),
        (r"\bwon\s*'?\s*t\b", "won't"),
        (r"\bdidn\s*'?\s*t\b", "didn't"),
        (r"\bisn\s*'?\s*t\b", "isn't"),
        (r"\baren\s*'?\s*t\b", "aren't"),
        (r"\bwasn\s*'?\s*t\b", "wasn't"),
        (r"\bweren\s*'?\s*t\b", "weren't"),
        (r"\bhasn\s*'?\s*t\b", "hasn't"),
        (r"\bhaven\s*'?\s*t\b", "haven't"),
        (r"\bhadn\s*'?\s*t\b", "hadn't"),
        (r"\bwouldn\s*'?\s*t\b", "wouldn't"),
        (r"\bcouldn\s*'?\s*t\b", "couldn't"),
        (r"\bshouldn\s*'?\s*t\b", "shouldn't"),
        (r"\bdoesn\s*'?\s*t\b", "doesn't"),
        (r"\bI\s*'?\s*m\b", "I'm"),
        (r"\bI\s*'?\s*ll\b", "I'll"),
        (r"\bI\s*'?\s*ve\b", "I've"),
        (r"\bI\s*'?\s*d\b", "I'd"),
        (r"\byou\s*'?\s*re\b", "you're"),
        (r"\byou\s*'?\s*ll\b", "you'll"),
        (r"\byou\s*'?\s*ve\b", "you've"),
        (r"\bwe\s*'?\s*re\b", "we're"),
        (r"\bwe\s*'?\s*ll\b", "we'll"),
        (r"\bwe\s*'?\s*ve\b", "we've"),
        (r"\bthey\s*'?\s*re\b", "they're"),
        (r"\bthey\s*'?\s*ll\b", "they'll"),
        (r"\bthey\s*'?\s*ve\b", "they've"),
        (r"\bit\s*'?\s*s\b", "it's"),
        (r"\bthat\s*'?\s*s\b", "that's"),
        (r"\bwhat\s*'?\s*s\b", "what's"),
        (r"\bwho\s*'?\s*s\b", "who's"),
        (r"\blet\s*'?\s*s\b", "let's"),
    ]
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def fix_spacing(text: str) -> str:
    """Fix multiple spaces and spacing around punctuation."""
    if not text:
        return text
    result = text
    # Multiple spaces -> single space
    result = re.sub(r' {2,}', ' ', result)
    # Remove space before punctuation
    result = re.sub(r' ([.,!?;:])', r'\1', result)
    # Ensure space after punctuation (except at end)
    result = re.sub(r'([.,!?;:])([A-Za-z])', r'\1 \2', result)
    return result.strip()


def remove_word_repetitions(text: str) -> str:
    """Remove immediate word repetitions like 'the the' -> 'the'."""
    if not text:
        return text
    # Remove word repeated immediately
    result = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
    return result


def capitalize_sentences(text: str) -> str:
    """Capitalize first letter of each sentence."""
    if not text:
        return text
    # Capitalize first character
    result = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    # Capitalize after sentence-ending punctuation
    result = re.sub(r'([.!?])\s+([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)
    return result


if __name__ == "__main__":
    demo_cleanup()

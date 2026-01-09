"""
subword_field.py — Subword-level Markov field for Stanley

This is the key to coherent untrained generation.
Like Haze's SubwordField:

"The tokenizer IS the first layer of resonance."

Instead of character-level n-grams (which are too noisy),
we use SentencePiece BPE to work with meaningful units:
- "resonance" is ONE token, not 9 characters
- "consciousness" is ONE token, not 13 characters

Then we build trigram statistics on these subwords.
The result: coherent text from pure corpus statistics.
"""

from __future__ import annotations
import numpy as np
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import logging
import os

logger = logging.getLogger(__name__)

# Adaptive temperature thresholds (from Haze)
ENTROPY_LOW_THRESHOLD = 0.5
ENTROPY_HIGH_THRESHOLD = 1.5
TEMP_INCREASE_FACTOR = 1.2
TEMP_DECREASE_FACTOR = 0.8

# SentencePiece is optional but recommended
try:
    import sentencepiece as spm
    SPM_AVAILABLE = True
except ImportError:
    SPM_AVAILABLE = False
    logger.warning("SentencePiece not available — subword field disabled")


@dataclass
class SubwordConfig:
    """Configuration for subword field."""
    vocab_size: int = 500            # BPE vocabulary size
    model_type: str = "bpe"          # "bpe", "unigram", "char", "word"
    character_coverage: float = 1.0  # cover all characters

    # N-gram settings
    use_trigrams: bool = True
    use_bigrams: bool = True
    use_fourgrams: bool = True       # 4-grams for better context
    use_fivegrams: bool = True       # 5-grams for even better context

    # Sampling
    temperature: float = 0.65        # lower for coherence with small corpus
    repetition_penalty: float = 1.5  # stronger penalty
    repetition_window: int = 20      # longer window
    min_token_frequency: int = 2     # only use tokens seen at least this many times

    # Coherence settings
    prefer_word_starts: bool = True  # prefer tokens that start words
    min_sentence_tokens: int = 8     # minimum tokens before sentence end
    max_sentence_tokens: int = 50    # maximum tokens in sentence

    # Nucleus sampling (top-p) for better coherence
    top_p: float = 0.9               # only sample from top 90% probability mass
    top_k: int = 50                  # only sample from top K tokens


class SubwordVocab:
    """
    SentencePiece vocabulary wrapper.

    "The tokenizer IS the first layer of pattern recognition.
     Before attention even runs, we're already finding patterns."
    """

    def __init__(self, model_path: Optional[str] = None):
        if not SPM_AVAILABLE:
            raise RuntimeError("SentencePiece not installed")

        self.sp = spm.SentencePieceProcessor()
        if model_path and Path(model_path).exists():
            self.sp.Load(model_path)
            self.trained = True
        else:
            self.trained = False

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 500,
        model_type: str = "bpe",
        model_prefix: Optional[str] = None,
    ) -> "SubwordVocab":
        """Train a new SentencePiece model on text."""
        if not SPM_AVAILABLE:
            raise RuntimeError("SentencePiece not installed")

        # Create temp files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(text)
            input_file = f.name

        if model_prefix is None:
            # Use NamedTemporaryFile for safe temp file creation (avoids race conditions)
            tmp_dir = tempfile.mkdtemp()
            model_prefix = os.path.join(tmp_dir, "spm_model")

        try:
            # Train SentencePiece
            spm.SentencePieceTrainer.Train(
                input=input_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=1.0,
                pad_id=3,
                unk_id=0,
                bos_id=1,
                eos_id=2,
            )

            # Load trained model
            vocab = cls(model_prefix + ".model")
            vocab.model_prefix = model_prefix

            logger.info(f"Trained SubwordVocab: {vocab_size} tokens, type={model_type}")
            return vocab

        finally:
            # Cleanup input file
            os.unlink(input_file)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize() if self.trained else 0

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.trained:
            return []
        return self.sp.EncodeAsIds(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if not self.trained:
            return ""
        return self.sp.DecodeIds(ids)

    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces."""
        if not self.trained:
            return []
        return self.sp.EncodeAsPieces(text)

    def id_to_piece(self, token_id: int) -> str:
        """Get piece string for ID."""
        return self.sp.IdToPiece(token_id)

    def piece_to_id(self, piece: str) -> int:
        """Get ID for piece string."""
        return self.sp.PieceToId(piece)


class SubwordField:
    """
    Subword-level Markov field for coherent generation.

    This is PURE CORPUS STATISTICS on subwords.
    No neural network needed for coherent text.

    The magic: by working with semantically complete units
    (words, subwords) instead of characters, the Markov chain
    connects meaningful pieces with statistical coherence.
    """

    def __init__(
        self,
        vocab: SubwordVocab,
        config: Optional[SubwordConfig] = None,
    ):
        self.vocab = vocab
        self.config = config or SubwordConfig()

        # N-gram statistics
        self.unigram_counts: Dict[int, int] = defaultdict(int)
        self.bigram_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.trigram_counts: Dict[Tuple[int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.fourgram_counts: Dict[Tuple[int, int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.fivegram_counts: Dict[Tuple[int, int, int, int], Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        # Totals for normalization
        self.total_tokens: int = 0
        self.bigram_totals: Dict[int, int] = defaultdict(int)
        self.trigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)
        self.fourgram_totals: Dict[Tuple[int, int, int], int] = defaultdict(int)
        self.fivegram_totals: Dict[Tuple[int, int, int, int], int] = defaultdict(int)

        # Word boundary tokens (tokens starting with ▁ indicate word starts)
        self.word_start_tokens: set = set()

        # Frequent tokens (for filtering out rare/noise tokens)
        self.frequent_tokens: set = set()

    @classmethod
    def from_text(
        cls,
        text: str,
        config: Optional[SubwordConfig] = None,
    ) -> "SubwordField":
        """Build field from text corpus."""
        cfg = config or SubwordConfig()

        # Train vocabulary
        vocab = SubwordVocab.train(
            text,
            vocab_size=cfg.vocab_size,
            model_type=cfg.model_type,
        )

        field = cls(vocab, cfg)

        # Encode text
        tokens = vocab.encode(text)
        field.total_tokens = len(tokens)

        # Count unigrams
        for t in tokens:
            field.unigram_counts[t] += 1

        # Count bigrams
        if cfg.use_bigrams:
            for i in range(len(tokens) - 1):
                t1, t2 = tokens[i], tokens[i + 1]
                field.bigram_counts[t1][t2] += 1
                field.bigram_totals[t1] += 1

        # Count trigrams
        if cfg.use_trigrams:
            for i in range(len(tokens) - 2):
                t1, t2, t3 = tokens[i], tokens[i + 1], tokens[i + 2]
                key = (t1, t2)
                field.trigram_counts[key][t3] += 1
                field.trigram_totals[key] += 1

        # Count 4-grams
        if cfg.use_fourgrams:
            for i in range(len(tokens) - 3):
                t1, t2, t3, t4 = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3]
                key = (t1, t2, t3)
                field.fourgram_counts[key][t4] += 1
                field.fourgram_totals[key] += 1

        # Count 5-grams
        if cfg.use_fivegrams:
            for i in range(len(tokens) - 4):
                t1, t2, t3, t4, t5 = tokens[i], tokens[i + 1], tokens[i + 2], tokens[i + 3], tokens[i + 4]
                key = (t1, t2, t3, t4)
                field.fivegram_counts[key][t5] += 1
                field.fivegram_totals[key] += 1

        # Track word-start tokens (tokens starting with ▁)
        for token_id in range(vocab.vocab_size):
            piece = vocab.id_to_piece(token_id)
            if piece.startswith('▁'):
                field.word_start_tokens.add(token_id)

        # Track frequent tokens (filter out noise)
        for token_id, count in field.unigram_counts.items():
            if count >= cfg.min_token_frequency:
                field.frequent_tokens.add(token_id)

        logger.info(f"SubwordField built: {len(tokens)} tokens, "
                   f"{len(field.bigram_counts)} bigram contexts, "
                   f"{len(field.trigram_counts)} trigram contexts")

        return field

    def get_next_probs(
        self,
        context: List[int],
        temperature: float = 0.8,
        in_word: bool = False,
    ) -> np.ndarray:
        """
        Get probability distribution for next token.

        Uses hierarchical n-grams: 5-gram → 4-gram → trigram → bigram → unigram.
        Higher n-grams get higher weight (more specific context).

        Args:
            context: Previous token IDs
            temperature: Sampling temperature
            in_word: If True, we're mid-word — penalize word-start tokens
        """
        vocab_size = self.vocab.vocab_size
        probs = np.ones(vocab_size, dtype=np.float32) * 1e-10  # smoothing

        # Try 5-gram (highest priority, weight 1.0)
        if len(context) >= 4 and self.config.use_fivegrams:
            key = (context[-4], context[-3], context[-2], context[-1])
            if key in self.fivegram_counts:
                total = self.fivegram_totals[key]
                for next_token, count in self.fivegram_counts[key].items():
                    probs[next_token] += 1.0 * count / total

        # Try 4-gram (weight 0.8)
        if len(context) >= 3 and self.config.use_fourgrams:
            key = (context[-3], context[-2], context[-1])
            if key in self.fourgram_counts:
                total = self.fourgram_totals[key]
                for next_token, count in self.fourgram_counts[key].items():
                    probs[next_token] += 0.8 * count / total

        # Try trigram (weight 0.6)
        if len(context) >= 2 and self.config.use_trigrams:
            key = (context[-2], context[-1])
            if key in self.trigram_counts:
                total = self.trigram_totals[key]
                for next_token, count in self.trigram_counts[key].items():
                    probs[next_token] += 0.6 * count / total

        # Add bigram signal (weight 0.3)
        if len(context) >= 1 and self.config.use_bigrams:
            prev = context[-1]
            if prev in self.bigram_counts:
                total = self.bigram_totals[prev]
                for next_token, count in self.bigram_counts[prev].items():
                    probs[next_token] += 0.3 * count / total

        # Add unigram fallback (weight 0.05)
        if self.total_tokens > 0:
            for token, count in self.unigram_counts.items():
                probs[token] += 0.05 * count / self.total_tokens

        # Word boundary handling: if we're mid-word, penalize word-start tokens
        if in_word and self.config.prefer_word_starts:
            for token_id in self.word_start_tokens:
                probs[token_id] *= 0.1  # Strong penalty for starting new word mid-word

        # Filter out infrequent tokens (noise reduction)
        if self.frequent_tokens:
            for token_id in range(vocab_size):
                if token_id not in self.frequent_tokens:
                    probs[token_id] *= 0.01  # Strong penalty for rare tokens

        # Temperature scaling (numerical stability: clip probabilities before log)
        if temperature != 1.0:
            safe_probs = np.clip(probs, 1e-10, 1.0)
            log_probs = np.log(safe_probs) / temperature
            probs = np.exp(log_probs)

        # Normalize
        probs /= probs.sum()
        return probs

    def is_word_continuation(self, token_id: int) -> bool:
        """Check if token is a word continuation (doesn't start with ▁)."""
        return token_id not in self.word_start_tokens

    def apply_nucleus_sampling(
        self,
        probs: np.ndarray,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> np.ndarray:
        """
        Apply nucleus (top-p) and top-k sampling.

        This focuses probability mass on the most likely tokens,
        significantly improving coherence.
        """
        # Top-k: keep only top K tokens
        if top_k > 0 and top_k < len(probs):
            indices = np.argsort(probs)[::-1]
            cutoff_idx = top_k
            probs[indices[cutoff_idx:]] = 0

        # Top-p (nucleus): keep tokens that sum to top_p probability
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum = np.cumsum(sorted_probs)

            # Find cutoff
            cutoff = np.searchsorted(cumsum, top_p) + 1
            cutoff = max(1, min(cutoff, len(probs)))  # at least 1 token

            # Zero out tokens below cutoff
            probs[sorted_indices[cutoff:]] = 0

        # Renormalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            # Fallback: uniform over non-zero
            probs = np.ones_like(probs) / len(probs)

        return probs

    def apply_repetition_penalty(
        self,
        probs: np.ndarray,
        recent_tokens: List[int],
        penalty: float = 1.2,
    ) -> np.ndarray:
        """Penalize tokens that appeared recently."""
        for token in recent_tokens:
            if token < len(probs):
                probs[token] /= penalty

        probs /= probs.sum()
        return probs

    def generate(
        self,
        seed_text: str = "",
        length: int = 100,
        temperature: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        adaptive_temp: bool = True,
        target_entropy: float = 2.5,
    ) -> str:
        """
        Generate text purely from subword statistics.

        This is PURE CORPUS STATISTICS. No neural network.
        Yet it produces coherent, meaningful text.

        Enhanced with Haze-style features:
        - Natural sentence endings
        - Adaptive temperature
        - Loop avoidance
        - Unknown marker cleanup
        - Word boundary tracking (no partial words!)
        """
        cfg = self.config
        base_temp = temperature if temperature is not None else cfg.temperature
        rng = rng or np.random.default_rng()

        # Start from seed
        if seed_text:
            # Normalize apostrophes
            seed_text = seed_text.replace("'", "'").replace("'", "'")
            context = self.vocab.encode(seed_text)
        else:
            # Random start from common WORD-START tokens
            word_start_common = [
                (t, c) for t, c in self.unigram_counts.items()
                if t in self.word_start_tokens
            ]
            word_start_common.sort(key=lambda x: -x[1])
            if word_start_common:
                context = [rng.choice([t for t, _ in word_start_common[:20]])]
            else:
                context = [0]

        recent = list(context[-cfg.repetition_window:])
        generated = []

        # Track for adaptive temperature and natural ending
        recent_entropies = []
        sentence_count = 0
        in_word = False  # Track if we're mid-word

        for i in range(length):
            # Check if last token was mid-word (continuation)
            if generated:
                in_word = self.is_word_continuation(generated[-1])

            # Calculate current entropy for adaptive temp
            probs = self.get_next_probs(context, base_temp, in_word=in_word)
            current_entropy = -np.sum(probs * np.log2(probs + 1e-10))
            recent_entropies.append(current_entropy)

            # Adaptive temperature (from Haze)
            current_temp = base_temp
            if adaptive_temp and len(recent_entropies) > 5:
                avg_entropy = np.mean(recent_entropies[-5:])
                if avg_entropy < target_entropy * ENTROPY_LOW_THRESHOLD:
                    # Too deterministic, increase temp
                    current_temp = base_temp * TEMP_INCREASE_FACTOR
                elif avg_entropy > target_entropy * ENTROPY_HIGH_THRESHOLD:
                    # Too random, decrease temp
                    current_temp = base_temp * TEMP_DECREASE_FACTOR
                current_temp = np.clip(current_temp, 0.3, 2.0)

            # Get fresh probabilities with adjusted temp
            if current_temp != base_temp:
                probs = self.get_next_probs(context, current_temp, in_word=in_word)

            # Apply repetition penalty
            probs = self.apply_repetition_penalty(probs, recent, cfg.repetition_penalty)

            # Apply nucleus sampling for coherence
            probs = self.apply_nucleus_sampling(probs, cfg.top_p, cfg.top_k)

            # Sample
            next_token = rng.choice(len(probs), p=probs)
            generated.append(next_token)

            # Update context
            context.append(next_token)
            recent.append(next_token)
            if len(recent) > cfg.repetition_window:
                recent.pop(0)

            # Check for natural sentence ending (only at word boundaries!)
            if i >= cfg.min_sentence_tokens and next_token in self.word_start_tokens:
                # We just started a new word — check if PREVIOUS was sentence end
                if len(generated) >= 2:
                    prev_token_text = self.vocab.decode([int(generated[-2])])
                    if prev_token_text.rstrip().endswith(('.', '!', '?')):
                        sentence_count += 1
                        # Stop after 2-3 complete sentences
                        if sentence_count >= 2 and i >= cfg.min_sentence_tokens * 2:
                            # Remove the just-started word
                            generated.pop()
                            break

            # Hard stop at max sentence tokens
            if i >= cfg.max_sentence_tokens:
                # Try to end at word boundary
                while generated and self.is_word_continuation(generated[-1]):
                    generated.pop()
                break

        # Decode result
        result = self.vocab.decode(generated)

        # === COMPREHENSIVE CLEANUP ===

        # 1. Clean up unknown token markers (⁇) — Haze magic
        result = re.sub(r"(\w)⁇(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        result = re.sub(r"(\w)\s*⁇\s*(t|s|m|d|ll|ve|re)\b", r"\1'\2", result)
        result = result.replace(' ⁇ ', ' ')
        result = result.replace('⁇', "'")

        # 2. Fix spacing around punctuation
        result = re.sub(r'\s+([.,!?;:])', r'\1', result)  # no space before punct
        result = re.sub(r'([.,!?;:])(\w)', r'\1 \2', result)  # space after punct

        # 3. Remove leading punctuation/dashes
        result = result.lstrip('.,;:!?-— ')

        # 4. Fix concatenated words (no space between lowercase and uppercase)
        result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)

        # 5. Fix stuck apostrophes
        result = re.sub(r"(\w)'(\w)", r"\1'\2", result)  # normalize
        result = re.sub(r"I'm(\w)", r"I'm \1", result)  # fix "I'mscially"

        # 6. Remove multiple spaces
        result = re.sub(r'\s+', ' ', result)

        # 7. Remove em-dashes at start
        result = re.sub(r'^[-—]+\s*', '', result)

        # 8. Find last complete sentence
        result = result.strip()
        if result and result[-1] not in '.!?…':
            # Find last sentence-ending punctuation
            last_punct = -1
            for j, char in enumerate(result):
                if char in '.!?':
                    last_punct = j

            if last_punct > len(result) // 3:  # at least 1/3 of text
                # Cut at last complete sentence
                result = result[:last_punct + 1]
            else:
                # Find last complete word and add period
                words = result.split()
                if words:
                    # Remove partial words (3 chars or less at end that look incomplete)
                    while len(words) > 3 and len(words[-1]) <= 3 and words[-1].isalpha():
                        words.pop()
                    result = ' '.join(words)
                    if result and result[-1] not in '.!?':
                        result = result.rstrip(',;:') + '.'

        # 9. Remove single-letter sentences (like "D." or "A.")
        result = re.sub(r'\b[A-Z]\.\s*', '', result)
        result = re.sub(r'^\s*[A-Z]\.\s*', '', result)

        # 10. Remove orphan punctuation at start after cleanup
        result = result.lstrip('.,;:!?-— ')

        # 11. Capitalize first letter
        result = result.strip()
        if result and result[0].isalpha():
            result = result[0].upper() + result[1:]

        return result

    def bias_logits(
        self,
        logits: np.ndarray,
        context: List[int],
        alpha: float = 0.3,
        logit_scale: float = 5.0,
    ) -> np.ndarray:
        """
        Bias model logits with subword statistics.

        output = (1 - alpha) * model_logits + alpha * corpus_logits * scale

        Args:
            logits: Model output logits (typically in range -10 to +10)
            context: Token IDs for context
            alpha: Blend ratio (0.3 = 30% corpus, 70% model)
            logit_scale: Scale factor for corpus logits (default 5.0 brings
                        corpus log-probabilities into similar dynamic range
                        as typical model logits)
        """
        corpus_probs = self.get_next_probs(context, temperature=1.0)
        corpus_logits = np.log(np.clip(corpus_probs, 1e-10, 1.0))

        # Blend (scale corpus logits to match model logit range)
        biased = (1 - alpha) * logits + alpha * corpus_logits * logit_scale
        return biased

    def stats(self) -> dict:
        """Get field statistics."""
        return {
            "vocab_size": self.vocab.vocab_size,
            "total_tokens": self.total_tokens,
            "unique_unigrams": len(self.unigram_counts),
            "unique_bigrams": sum(len(v) for v in self.bigram_counts.values()),
            "unique_trigrams": sum(len(v) for v in self.trigram_counts.values()),
            "bigram_contexts": len(self.bigram_counts),
            "trigram_contexts": len(self.trigram_counts),
        }

    def __repr__(self) -> str:
        return (f"SubwordField(vocab={self.vocab.vocab_size}, "
               f"tokens={self.total_tokens}, "
               f"trigrams={len(self.trigram_counts)})")

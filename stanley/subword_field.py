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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import tempfile
import logging
import os

logger = logging.getLogger(__name__)

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

    # Sampling
    temperature: float = 0.8
    repetition_penalty: float = 1.2  # penalize recent tokens
    repetition_window: int = 10


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

        # Totals for normalization
        self.total_tokens: int = 0
        self.bigram_totals: Dict[int, int] = defaultdict(int)
        self.trigram_totals: Dict[Tuple[int, int], int] = defaultdict(int)

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

        logger.info(f"SubwordField built: {len(tokens)} tokens, "
                   f"{len(field.bigram_counts)} bigram contexts, "
                   f"{len(field.trigram_counts)} trigram contexts")

        return field

    def get_next_probs(
        self,
        context: List[int],
        temperature: float = 0.8,
    ) -> np.ndarray:
        """
        Get probability distribution for next token.

        Uses trigram if available, falls back to bigram, then unigram.
        """
        vocab_size = self.vocab.vocab_size
        probs = np.ones(vocab_size, dtype=np.float32) * 1e-10  # smoothing

        # Try trigram
        if len(context) >= 2 and self.config.use_trigrams:
            key = (context[-2], context[-1])
            if key in self.trigram_counts:
                total = self.trigram_totals[key]
                for next_token, count in self.trigram_counts[key].items():
                    probs[next_token] += count / total

        # Add bigram signal (weight=0.5: bigrams are less specific than trigrams,
        # so we weight them at half strength to avoid overwhelming trigram signal)
        if len(context) >= 1 and self.config.use_bigrams:
            prev = context[-1]
            if prev in self.bigram_counts:
                total = self.bigram_totals[prev]
                for next_token, count in self.bigram_counts[prev].items():
                    probs[next_token] += 0.5 * count / total

        # Add unigram fallback (weight=0.1: unigrams are corpus-wide frequencies,
        # used only as a gentle smoothing to avoid zero probabilities)
        if self.total_tokens > 0:
            for token, count in self.unigram_counts.items():
                probs[token] += 0.1 * count / self.total_tokens

        # Temperature scaling (numerical stability: clip probabilities before log)
        if temperature != 1.0:
            safe_probs = np.clip(probs, 1e-10, 1.0)
            log_probs = np.log(safe_probs) / temperature
            probs = np.exp(log_probs)

        # Normalize
        probs /= probs.sum()
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
    ) -> str:
        """
        Generate text purely from subword statistics.

        This is PURE CORPUS STATISTICS. No neural network.
        Yet it produces coherent, meaningful text.
        """
        cfg = self.config
        temp = temperature if temperature is not None else cfg.temperature
        rng = rng or np.random.default_rng()

        # Start from seed
        if seed_text:
            context = self.vocab.encode(seed_text)
        else:
            # Random start from common tokens
            common = sorted(self.unigram_counts.items(), key=lambda x: -x[1])[:20]
            if common:
                context = [rng.choice([t for t, _ in common])]
            else:
                context = [0]

        recent = list(context[-cfg.repetition_window:])
        generated = []

        for _ in range(length):
            # Get probabilities
            probs = self.get_next_probs(context, temp)

            # Apply repetition penalty
            probs = self.apply_repetition_penalty(probs, recent, cfg.repetition_penalty)

            # Sample
            next_token = rng.choice(len(probs), p=probs)
            generated.append(next_token)

            # Update context
            context.append(next_token)
            recent.append(next_token)
            if len(recent) > cfg.repetition_window:
                recent.pop(0)

        return self.vocab.decode(generated)

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

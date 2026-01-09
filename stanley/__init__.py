"""
STANLEY — Self Training Attention Non-Linear EntitY

A self-evolving linguistic organism that grows through experience.

Part of the Arianna Method ecosystem.

Usage:
    from stanley import Stanley, create_stanley, load_or_create

    # Create new
    stanley = create_stanley(origin_text="I am Stanley...")

    # Or load existing
    stanley = load_or_create("./stanley_data")

    # Interact
    response, stats = stanley.think("Hello!")
    stanley.experience("Hello! How are you?")
"""

__version__ = "0.1.0"

# Core data structures
from .shard import Shard, MetaNote, combine_deltas
from .memory_sea import MemorySea
from .quantum_buffer import QuantumBuffer, AdaptiveQuantumBuffer
from .fingerprint import (
    compute_fingerprint,
    cosine_similarity,
    novelty_score,
    FingerprintConfig,
    FingerprintIndex,
)
from .router import Router, RouterConfig, AdaptiveRouter

# Inference
from .inference import (
    StanleyTransformer,
    InferenceEngine,
    Vocab,
    quick_stanley,
)

# Experience
from .experience import (
    ExperienceFilter,
    should_remember,
    compute_experience_score,
    ExperienceJournal,
)

# Origin
from .origin import (
    load_origin,
    default_origin,
    OriginField,
)

# Co-occurrence field (for coherent untrained generation)
from .cooccur import (
    CooccurField,
    CooccurConfig,
)

# Subword field — the key to coherent untrained generation
try:
    from .subword_field import (
        SubwordField,
        SubwordVocab,
        SubwordConfig,
        SPM_AVAILABLE,
    )
except ImportError:
    SPM_AVAILABLE = False

# Subjectivity — "NO SEED FROM PROMPT"
from .subjectivity import (
    Subjectivity,
    Pulse,
    Identity,
)

# Organism (main class)
from .organism import (
    Stanley,
    StanleyConfig,
    create_stanley,
    load_or_create,
)

# Trainer (optional, requires PyTorch)
try:
    from .trainer import (
        LoRAConfig,
        compute_lora_delta,
        MicroTrainer,
        TrainerConfig,
        Consolidator,
        ConsolidationConfig,
        TORCH_AVAILABLE,
    )
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    # Version
    "__version__",

    # Core
    "Shard",
    "MetaNote",
    "combine_deltas",

    # Storage
    "MemorySea",

    # Accumulation
    "QuantumBuffer",
    "AdaptiveQuantumBuffer",

    # Fingerprinting
    "compute_fingerprint",
    "cosine_similarity",
    "novelty_score",
    "FingerprintConfig",
    "FingerprintIndex",

    # Routing
    "Router",
    "RouterConfig",
    "AdaptiveRouter",

    # Inference
    "StanleyTransformer",
    "InferenceEngine",
    "Vocab",
    "quick_stanley",

    # Experience
    "ExperienceFilter",
    "should_remember",
    "compute_experience_score",
    "ExperienceJournal",

    # Origin
    "load_origin",
    "default_origin",
    "OriginField",

    # Co-occurrence
    "CooccurField",
    "CooccurConfig",

    # Subword field
    "SubwordField",
    "SubwordVocab",
    "SubwordConfig",
    "SPM_AVAILABLE",

    # Subjectivity
    "Subjectivity",
    "Pulse",
    "Identity",

    # Organism
    "Stanley",
    "StanleyConfig",
    "create_stanley",
    "load_or_create",

    # Trainer (optional)
    "TORCH_AVAILABLE",
]

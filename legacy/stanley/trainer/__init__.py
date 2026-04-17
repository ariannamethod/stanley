"""
trainer — PyTorch training modules for Stanley

This is the ONLY place PyTorch lives.
All output is NumPy arrays for inference.

The trainer produces LoRA deltas that shape personality.
"""

from .lora import (
    LoRAConfig,
    compute_lora_delta,
    compute_lora_delta_from_gradient,
    create_empty_deltas,
    scale_deltas,
    merge_deltas,
    TORCH_AVAILABLE,
)

from .micro_trainer import (
    MicroTrainer,
    TrainerConfig,
    TrainerState,
    TrainingItem,
    TrainingResult,
    AutoSwapper,
    create_shard_from_training,
)

from .consolidator import (
    Consolidator,
    ConsolidationConfig,
    ConsolidationResult,
    consolidate_shards,
    compress_to_metanote,
    find_compression_candidates,
    find_resurrection_candidates,
)

__all__ = [
    # LoRA
    "LoRAConfig",
    "compute_lora_delta",
    "compute_lora_delta_from_gradient",
    "create_empty_deltas",
    "scale_deltas",
    "merge_deltas",
    "TORCH_AVAILABLE",

    # Micro-trainer
    "MicroTrainer",
    "TrainerConfig",
    "TrainerState",
    "TrainingItem",
    "TrainingResult",
    "AutoSwapper",
    "create_shard_from_training",

    # Consolidator
    "Consolidator",
    "ConsolidationConfig",
    "ConsolidationResult",
    "consolidate_shards",
    "compress_to_metanote",
    "find_compression_candidates",
    "find_resurrection_candidates",
]

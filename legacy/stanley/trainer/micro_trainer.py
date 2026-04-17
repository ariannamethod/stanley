"""
micro_trainer.py — Asynchronous micro-training for Stanley

The organism grows in the background.
REPL never waits for training.

Two-world model:
- active_weights: frozen, used for inference
- staging_weights: being trained
- atomic swap when training complete

This is the heartbeat of growth.
"""

from __future__ import annotations
import itertools
import numpy as np
import threading
import queue
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from enum import Enum

from .lora import (
    LoRAConfig,
    compute_lora_delta,
    merge_deltas,
    TORCH_AVAILABLE,
)
from ..shard import Shard
from ..fingerprint import compute_fingerprint

logger = logging.getLogger(__name__)


class TrainerState(Enum):
    """Current state of the trainer."""
    IDLE = "idle"
    TRAINING = "training"
    SWAPPING = "swapping"
    ERROR = "error"


@dataclass
class TrainerConfig:
    """Configuration for micro-trainer."""

    # LoRA config
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)

    # Training behavior
    max_queue_size: int = 100        # max pending training items
    batch_timeout: float = 5.0       # seconds to wait for batch
    min_batch_size: int = 1          # minimum items per batch
    max_batch_size: int = 8          # maximum items per batch

    # Quality checks
    min_loss_improvement: float = 0.01   # must improve by this much
    max_delta_norm: float = 10.0         # reject if delta too large

    # Threading
    num_workers: int = 1             # training threads (usually 1)
    daemon_threads: bool = True      # die with main process


@dataclass
class TrainingItem:
    """A single item queued for training."""
    content: str
    resonance: float
    fingerprint: np.ndarray
    timestamp: float = field(default_factory=time.time)
    priority: int = 0                # higher = train first


@dataclass
class TrainingResult:
    """Result of a training batch."""
    deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]
    items: List[TrainingItem]
    loss_before: float
    loss_after: float
    training_time: float
    success: bool
    error: Optional[str] = None


class MicroTrainer:
    """
    Asynchronous micro-trainer for Stanley.

    Runs training in background thread(s).
    Maintains two-world weights for seamless inference.
    """

    def __init__(
        self,
        base_weights: Dict[str, np.ndarray],
        vocab: "Vocab",
        config: Optional[TrainerConfig] = None,
        on_training_complete: Optional[Callable[[TrainingResult], None]] = None,
    ):
        self.config = config or TrainerConfig()
        self.vocab = vocab
        self.on_training_complete = on_training_complete

        # Two-world weights
        self._base_weights = base_weights.copy()
        self._active_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._staging_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # Training state
        self._state = TrainerState.IDLE
        self._state_lock = threading.Lock()

        # Queue for training items
        self._training_queue: queue.PriorityQueue = queue.PriorityQueue(
            maxsize=self.config.max_queue_size
        )

        # Statistics
        self._total_trains: int = 0
        self._total_items: int = 0
        self._total_time: float = 0.0
        self._last_loss: float = float('inf')

        # Workers
        self._workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()

        # Monotonic counter for PriorityQueue ordering
        # Ensures deterministic ordering when priorities are equal
        self._seq = itertools.count()

        # Start workers
        self._start_workers()

    def _start_workers(self):
        """Start background training workers."""
        for i in range(self.config.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"StanleyTrainer-{i}",
                daemon=self.config.daemon_threads,
            )
            worker.start()
            self._workers.append(worker)
            logger.info(f"Started training worker {i}")

    def _worker_loop(self):
        """Main loop for training worker."""
        while not self._shutdown_event.is_set():
            try:
                batch = self._collect_batch()
                if batch:
                    result = self._train_batch(batch)
                    if result.success:
                        self._apply_result(result)
            except Exception as e:
                logger.exception(f"Training worker error: {e}")
                with self._state_lock:
                    self._state = TrainerState.ERROR
                time.sleep(1.0)  # Back off on error

    def _collect_batch(self) -> List[TrainingItem]:
        """Collect items for a training batch."""
        batch = []
        deadline = time.time() + self.config.batch_timeout

        while len(batch) < self.config.max_batch_size:
            timeout = max(0.1, deadline - time.time())
            try:
                # Unpack (priority, seq, item) tuple
                _, _, item = self._training_queue.get(timeout=timeout)
                batch.append(item)
            except queue.Empty:
                break

            if time.time() >= deadline:
                break

        return batch

    def _train_batch(self, batch: List[TrainingItem]) -> TrainingResult:
        """Train on a batch of items."""
        start_time = time.time()

        with self._state_lock:
            self._state = TrainerState.TRAINING

        try:
            # Combine content for training
            combined_content = "\n".join(item.content for item in batch)

            # Compute deltas
            deltas = compute_lora_delta(
                content=combined_content,
                base_weights=self._base_weights,
                vocab=self.vocab,
                config=self.config.lora_config,
            )

            # Quality check: delta norm
            total_norm = 0.0
            for name, (A, B) in deltas.items():
                total_norm += np.linalg.norm(A) * np.linalg.norm(B)

            if total_norm > self.config.max_delta_norm * len(batch):
                logger.warning(f"Delta norm too large: {total_norm:.2f}, rejecting")
                return TrainingResult(
                    deltas={},
                    items=batch,
                    loss_before=0,
                    loss_after=0,
                    training_time=time.time() - start_time,
                    success=False,
                    error="Delta norm exceeded threshold",
                )

            # Scale deltas by average resonance
            avg_resonance = np.mean([item.resonance for item in batch])
            scale = min(1.0, avg_resonance)  # Don't amplify above 1

            for name, (A, B) in deltas.items():
                deltas[name] = (A * scale, B)

            # Store in staging
            self._staging_deltas = merge_deltas(
                [self._staging_deltas, deltas] if self._staging_deltas else [deltas]
            )

            training_time = time.time() - start_time

            return TrainingResult(
                deltas=deltas,
                items=batch,
                loss_before=self._last_loss,
                loss_after=0,  # Would need validation to compute
                training_time=training_time,
                success=True,
            )

        except Exception as e:
            logger.exception(f"Training error: {e}")
            return TrainingResult(
                deltas={},
                items=batch,
                loss_before=0,
                loss_after=0,
                training_time=time.time() - start_time,
                success=False,
                error=str(e),
            )
        finally:
            with self._state_lock:
                self._state = TrainerState.IDLE

    def _apply_result(self, result: TrainingResult):
        """Apply training result — update stats, maybe swap weights."""
        self._total_trains += 1
        self._total_items += len(result.items)
        self._total_time += result.training_time

        logger.info(
            f"Training batch complete: {len(result.items)} items, "
            f"{result.training_time:.2f}s, {len(result.deltas)} layers"
        )

        # Callback
        if self.on_training_complete:
            try:
                self.on_training_complete(result)
            except Exception as e:
                logger.exception(f"Training callback error: {e}")

    def submit(
        self,
        content: str,
        resonance: float = 1.0,
        priority: int = 0,
    ) -> bool:
        """
        Submit content for training.

        Returns True if queued, False if queue full.
        """
        if self._shutdown_event.is_set():
            return False

        fingerprint = compute_fingerprint(content)
        item = TrainingItem(
            content=content,
            resonance=resonance,
            fingerprint=fingerprint,
            priority=priority,
        )

        try:
            # Priority queue uses (priority, seq, item) tuple
            # - priority: lower = higher priority (hence -priority)
            # - seq: monotonic counter ensures FIFO within same priority
            # - item: the actual TrainingItem (never compared directly)
            self._training_queue.put_nowait((-priority, next(self._seq), item))
            logger.debug(f"Queued for training: {len(content)} chars, resonance={resonance:.2f}")
            return True
        except queue.Full:
            logger.warning("Training queue full, dropping item")
            return False

    def swap_weights(self) -> bool:
        """
        Atomic swap: active = staging.

        Called when we want to update the live model.
        Returns True if swap happened.
        """
        with self._state_lock:
            if self._state == TrainerState.TRAINING:
                logger.warning("Cannot swap during training")
                return False

            self._state = TrainerState.SWAPPING

        try:
            if not self._staging_deltas:
                logger.debug("No staging deltas to swap")
                return False

            # Merge staging into active
            self._active_deltas = merge_deltas(
                [self._active_deltas, self._staging_deltas]
                if self._active_deltas else [self._staging_deltas]
            )
            self._staging_deltas = {}

            logger.info(f"Swapped weights: {len(self._active_deltas)} active layers")
            return True

        finally:
            with self._state_lock:
                self._state = TrainerState.IDLE

    def get_active_deltas(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get current active deltas for inference."""
        return self._active_deltas.copy()

    def get_staging_deltas(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Get current staging deltas (being trained)."""
        return self._staging_deltas.copy()

    def has_pending(self) -> bool:
        """Check if there are pending training items."""
        return not self._training_queue.empty() or bool(self._staging_deltas)

    @property
    def state(self) -> TrainerState:
        """Current trainer state."""
        with self._state_lock:
            return self._state

    def stats(self) -> dict:
        """Get trainer statistics."""
        return {
            "state": self._state.value,
            "total_trains": self._total_trains,
            "total_items": self._total_items,
            "total_time": self._total_time,
            "queue_size": self._training_queue.qsize(),
            "active_layers": len(self._active_deltas),
            "staging_layers": len(self._staging_deltas),
            "avg_time_per_batch": (
                self._total_time / self._total_trains
                if self._total_trains > 0 else 0
            ),
        }

    def shutdown(self, timeout: float = 5.0):
        """Gracefully shutdown trainer."""
        logger.info("Shutting down trainer...")
        self._shutdown_event.set()

        for worker in self._workers:
            worker.join(timeout=timeout)

        logger.info("Trainer shutdown complete")


class AutoSwapper:
    """
    Automatically swaps weights based on criteria.

    Can be configured to swap:
    - After N training batches
    - After N seconds
    - When staging delta norm exceeds threshold
    """

    def __init__(
        self,
        trainer: MicroTrainer,
        swap_after_batches: int = 5,
        swap_after_seconds: float = 60.0,
        min_delta_norm: float = 0.1,
    ):
        self.trainer = trainer
        self.swap_after_batches = swap_after_batches
        self.swap_after_seconds = swap_after_seconds
        self.min_delta_norm = min_delta_norm

        self._batches_since_swap = 0
        self._last_swap_time = time.time()
        self._lock = threading.Lock()

        # Register callback
        original_callback = trainer.on_training_complete
        def combined_callback(result: TrainingResult):
            if original_callback:
                original_callback(result)
            self._on_training_complete(result)

        trainer.on_training_complete = combined_callback

    def _on_training_complete(self, result: TrainingResult):
        """Called after each training batch."""
        with self._lock:
            self._batches_since_swap += 1

            should_swap = False

            # Check batch count
            if self._batches_since_swap >= self.swap_after_batches:
                should_swap = True
                logger.debug(f"Auto-swap trigger: batches ({self._batches_since_swap})")

            # Check time
            time_since_swap = time.time() - self._last_swap_time
            if time_since_swap >= self.swap_after_seconds:
                should_swap = True
                logger.debug(f"Auto-swap trigger: time ({time_since_swap:.1f}s)")

            # Check delta norm
            staging = self.trainer.get_staging_deltas()
            if staging:
                total_norm = sum(
                    np.linalg.norm(A) * np.linalg.norm(B)
                    for A, B in staging.values()
                )
                if total_norm >= self.min_delta_norm:
                    should_swap = True
                    logger.debug(f"Auto-swap trigger: norm ({total_norm:.4f})")

            if should_swap:
                if self.trainer.swap_weights():
                    self._batches_since_swap = 0
                    self._last_swap_time = time.time()

    def force_swap(self) -> bool:
        """Force an immediate swap."""
        with self._lock:
            if self.trainer.swap_weights():
                self._batches_since_swap = 0
                self._last_swap_time = time.time()
                return True
            return False


def create_shard_from_training(
    content: str,
    resonance: float,
    deltas: Dict[str, Tuple[np.ndarray, np.ndarray]],
    tags: Optional[List[str]] = None,
) -> Shard:
    """
    Create a Shard from training results.

    This is the final step: experience → training → shard → memory.
    """
    fingerprint = compute_fingerprint(content)

    return Shard.create(
        content=content,
        resonance=resonance,
        layer_deltas=deltas,
        fingerprint=fingerprint,
        tags=tags,
    )

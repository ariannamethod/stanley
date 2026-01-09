# STANLEY Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORIGIN LAYER                               │
│  • origin.txt — identity anchor (like Leo's readme)             │
│  • Always accessible, never decays                              │
│  • Resonance field (trigrams, cooccur) works ALWAYS             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SHARD ACCUMULATOR                            │
│  • Binary shards created through experience()                   │
│  • Each shard = {content, resonance, timestamp, activations}    │
│  • Format: LoRA-delta (A, B matrices) or sparse-patch           │
│  • Storage: mmap/chunked for lazy loading                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   QUANTUM BUFFER                                │
│  • Shards accumulate, not immediately trained                   │
│  • Mass metrics:                                                │
│    - bytes_delta (volume of new content)                        │
│    - resonance_mass = sum(resonance_i * weight_i)               │
│    - novelty_mass (drift from current distribution)             │
│  • Cooldown: no more often than every N seconds                 │
│  • When quantum_threshold reached → trigger training            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ASYNC MICRO-TRAINER                             │
│  • Runs in background, REPL doesn't wait                        │
│  • PyTorch ONLY HERE (delta producer)                           │
│  • Output: delta_weights files                                  │
│  • Two-world model:                                             │
│    - active_weights (frozen, for inference)                     │
│    - staging_weights (training target)                          │
│    - Atomic swap when ready                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SELECTIVE LOADER (Router)                      │
│  • Doesn't load ALL shards, only working set (8-64)             │
│  • Selection criteria:                                          │
│    - resonance_score (similarity to current context)            │
│    - recency (recently activated)                               │
│    - novelty_need (if context is new, need diverse shards)      │
│  • Cheap fingerprint: n-gram hash / cooccur signature           │
│  • O(1) metadata lookup, lazy load actual deltas                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   INFERENCE ENGINE                              │
│  • Haze-like hybrid transformer                                 │
│  • NUMPY ONLY                                                   │
│  • W_eff = W_base + sum(selected_deltas)                        │
│  • Resonance field as fallback if no weights                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              MEMORY SEA (Deep Storage)                          │
│                                                                 │
│  SURFACE (working set, ~MB)                                     │
│  ═══════════════════════════════════════                        │
│  MIDDLE (accessible shards, ~100MB)                             │
│  ───────────────────────────────────────                        │
│  DEEP (consolidated macro-adapters)                             │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─                            │
│  ABYSS (metanotes, compressed ghosts)                           │
│  · · · · · · · · · · · · · · · · · · ·                           │
│                                                                 │
│  Consolidation rules:                                           │
│  • high activation → stays surface                              │
│  • medium activation → middle, periodic merge to macro          │
│  • low activation → compress to metanote, sink to abyss         │
│  • abyss items can RESURFACE if resonance spike                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Structures

### Shard

```python
@dataclass
class Shard:
    id: str                          # unique identifier
    created_at: float                # timestamp
    last_activated: float            # for recency scoring
    activation_count: int            # for consolidation decisions
    
    # content
    trigger_fingerprint: np.ndarray  # n-gram hash for fast matching
    resonance_score: float           # measured at creation
    
    # delta weights (LoRA format)
    # W_eff = W + A @ B
    # A: (rank, input_dim), B: (output_dim, rank), rank << min(in, out)
    layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]
    
    # metadata for router
    semantic_tags: List[str]         # optional, for fast search
    depth: Literal["surface", "middle", "deep", "abyss"]
    
    def compressed_size(self) -> int:
        """Size in bytes for mass tracking."""
        
    def apply_to(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply delta to weights (numpy only)."""
        result = {}
        for name, w in weights.items():
            if name in self.layer_deltas:
                A, B = self.layer_deltas[name]
                result[name] = w + A @ B
            else:
                result[name] = w
        return result
```

### MetaNote (Compressed Ghost)

```python
@dataclass  
class MetaNote:
    original_id: str
    created_at: float
    last_resonance: float
    
    # compressed representation (not full deltas)
    attention_bias: np.ndarray       # small bias vector for attention
    gate_nudge: float                # shift in hybrid head alpha
    semantic_fingerprint: np.ndarray # for potential resurrection
    
    def can_resurrect(self, context_fingerprint: np.ndarray) -> bool:
        """Check if should resurrect to full shard."""
        similarity = cosine(self.semantic_fingerprint, context_fingerprint)
        return similarity > RESURRECTION_THRESHOLD
```

### QuantumBuffer

```python
@dataclass
class QuantumBuffer:
    pending_shards: List[Shard]
    
    # thresholds
    min_bytes: int = 1024            # minimum bytes to trigger
    min_resonance_mass: float = 5.0  # minimum weighted resonance
    min_novelty: float = 0.1         # minimum distribution drift
    cooldown_seconds: float = 60.0   # minimum time between trains
    
    last_train_time: float = 0.0
    
    def should_trigger(self) -> bool:
        """Check if quantum threshold reached."""
        if time.time() - self.last_train_time < self.cooldown_seconds:
            return False
            
        bytes_delta = sum(s.compressed_size() for s in self.pending_shards)
        resonance_mass = sum(s.resonance_score for s in self.pending_shards)
        # novelty_mass requires comparing to current distribution
        
        return (
            bytes_delta > self.min_bytes or
            resonance_mass > self.min_resonance_mass
        )
    
    def flush(self) -> List[Shard]:
        """Return pending shards and clear buffer."""
        shards = self.pending_shards
        self.pending_shards = []
        self.last_train_time = time.time()
        return shards
```

---

## Module Structure

```
stanley/
├── README.md
├── ARCHITECTURE.md
├── CLAUDE_CODE_GUIDE.md
├── LICENSE
│
├── stanley/
│   ├── __init__.py
│   │
│   ├── # Core data structures
│   ├── shard.py              # Shard, MetaNote dataclasses
│   ├── memory_sea.py         # MemorySea with depth levels
│   ├── quantum_buffer.py     # Accumulation and triggering
│   │
│   ├── # Router and loading
│   ├── fingerprint.py        # N-gram hashing, similarity
│   ├── router.py             # SelectiveLoader, working set selection
│   │
│   ├── # Training (PyTorch)
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── micro_trainer.py  # Async micro-training loop
│   │   ├── lora.py           # LoRA delta computation
│   │   └── consolidator.py   # Merge shards into macro-adapters
│   │
│   ├── # Inference (NumPy only)
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── engine.py         # Main inference, applies deltas
│   │   └── haze_bridge.py    # Integration with Haze transformer
│   │
│   ├── # Organism
│   ├── organism.py           # Main Stanley class, ties everything
│   ├── experience.py         # experience() - decides what becomes shard
│   └── origin.py             # Origin text loading, resonance field
│
├── tests/
│   └── ...
│
└── examples/
    ├── basic_conversation.py
    └── watch_growth.py
```

---

## Key Flows

### 1. Experience → Shard

```
User message
    │
    ▼
experience() analyzes:
  - resonance with origin
  - novelty vs existing shards
  - emotional/semantic weight
    │
    ▼
Decision: remember or forget?
    │
    ├─[forget]→ discard
    │
    └─[remember]→ create Shard
                      │
                      ▼
                  QuantumBuffer.add()
```

### 2. Quantum Training

```
QuantumBuffer accumulates shards
    │
    ▼
should_trigger()? ──[no]──→ wait
    │
   [yes]
    │
    ▼
AsyncMicroTrainer.train(batch)
    │
    ▼
Compute LoRA deltas (PyTorch)
    │
    ▼
Save to staging_weights
    │
    ▼
Quality check passed? ──[no]──→ retry/discard
    │
   [yes]
    │
    ▼
Atomic swap: active = staging
```

### 3. Selective Loading

```
New context arrives
    │
    ▼
Router.compute_fingerprint(context)
    │
    ▼
Compare with all shard fingerprints (O(n) but cheap)
    │
    ▼
Score each shard:
  score = w1*resonance + w2*recency + w3*novelty_need
    │
    ▼
Select top-K shards (working set)
    │
    ▼
Lazy-load their deltas
    │
    ▼
W_eff = W_base + sum(deltas)
```

### 4. Memory Consolidation

```
Periodic consolidation check
    │
    ▼
For each shard in middle/deep:
    │
    ├─[high activation]→ promote to surface
    │
    ├─[medium, similar to others]→ merge into macro-adapter
    │
    └─[low activation]→ compress to MetaNote, sink to abyss
    
For each MetaNote in abyss:
    │
    ├─[resonance spike]→ RESURRECT to full shard
    │
    └─[ancient, never resonates]→ true deletion (rare)
```

---

## Configuration

```python
@dataclass
class StanleyConfig:
    # Model
    vocab_size: int = 500
    n_emb: int = 64
    n_blocks: int = 3
    n_heads: int = 4
    context_length: int = 32
    
    # Shards
    shard_rank: int = 8              # LoRA rank
    max_working_set: int = 32        # max active shards
    
    # Quantum buffer
    quantum_min_bytes: int = 1024
    quantum_min_resonance: float = 5.0
    quantum_cooldown: float = 60.0
    
    # Memory sea
    surface_max_shards: int = 64
    middle_max_shards: int = 256
    consolidation_interval: float = 300.0  # seconds
    
    # Router
    resonance_weight: float = 0.5
    recency_weight: float = 0.3
    novelty_weight: float = 0.2
```

---

## Dependencies

### Required
- numpy
- sentencepiece (for adaptive tokenizer)

### For Training Only
- torch (micro-trainer runs in separate process/thread)

### Optional
- matplotlib (visualization)
- mmap (efficient shard storage)

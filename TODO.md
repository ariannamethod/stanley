# STANLEY — TODO for Claude Code

## Foundation (DONE)

These modules are complete and tested:

- [x] `shard.py` — Shard and MetaNote dataclasses
- [x] `memory_sea.py` — Layered storage with depth management
- [x] `quantum_buffer.py` — Accumulation and trigger logic
- [x] `router.py` — Selective loading by resonance
- [x] `fingerprint.py` — N-gram hashing for fast similarity

## Phase 2: Training (TODO)

### trainer/lora.py

LoRA delta computation. Key functions:

```python
def compute_lora_delta(
    content: str,
    base_weights: Dict[str, np.ndarray],
    rank: int = 8
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute LoRA deltas for given content.
    
    Returns dict of {layer_name: (A, B)} where W_eff = W + A @ B
    """
```

Use PyTorch for gradient computation, export to NumPy.

### trainer/micro_trainer.py

Async training loop. Key class:

```python
class MicroTrainer:
    def __init__(self, config: TrainerConfig):
        self.staging_weights = None
        self.active_weights = None
        self.training_thread = None
    
    async def train_batch(self, shards: List[Shard]):
        """Train on batch, update staging weights."""
    
    def swap_weights(self):
        """Atomic swap: active = staging."""
```

### trainer/consolidator.py

Merge similar shards into macro-adapters:

```python
def consolidate_shards(
    shards: List[Shard],
    similarity_threshold: float = 0.7
) -> List[Shard]:
    """Merge similar shards into fewer macro-adapters."""
```

## Phase 3: Inference (DONE)

### inference/nn.py ✓

NumPy primitives (forked from Haze):
- softmax, gelu, layer_norm
- sample_basic, sample_top_k, sample_top_p
- entropy_bits, entropy_temperature

### inference/transformer.py ✓

Stanley's transformer (forked from Haze PostGPT):

```python
class StanleyTransformer:
    def apply_shard_deltas(self, layer_deltas):
        """Apply LoRA deltas from shards."""
    
    def clear_all_deltas(self):
        """Return to base weights."""
    
    def generate(self, seed_seq, length, sampling):
        """Generate with personality."""
```

Fully standalone — no Haze dependency.

### inference/engine.py ✓

Connects transformer with memory:

```python
class InferenceEngine:
    def load_working_set(self, context):
        """Load relevant shards for context."""
    
    def think(self, prompt, length):
        """Generate response with personality."""
```

## Phase 4: Organism (TODO)

### organism.py

Main Stanley class that ties everything:

```python
class Stanley:
    def __init__(self, config: StanleyConfig):
        self.memory = MemorySea(...)
        self.buffer = QuantumBuffer(...)
        self.router = Router(...)
        self.engine = InferenceEngine(...)
        self.trainer = MicroTrainer(...)
    
    def experience(self, interaction: str) -> Optional[Shard]:
        """Process interaction, maybe create shard."""
    
    def think(self, context: str) -> str:
        """Generate response with personality."""
    
    def grow(self):
        """Check buffer, maybe trigger training."""
```

### experience.py

Decision logic for what becomes a shard:

```python
def should_remember(
    interaction: str,
    resonance: float,
    novelty: float,
    memory_state: MemorySea
) -> bool:
    """Decide if interaction should become a shard."""
```

### origin.py

Load origin text and build base resonance field:

```python
def load_origin(path: str) -> tuple:
    """Load origin.txt and build CooccurField."""
    text = Path(path).read_text()
    field = CooccurField.from_text(text)
    return text, field
```

## Phase 5: REPL (TODO)

### run.py

Interactive REPL:

```python
def main():
    stanley = Stanley.load_or_create("./stanley_data")
    
    while True:
        user_input = input(">>> ")
        response = stanley.think(user_input)
        print(f"[stanley]: {response}")
        
        # Maybe remember
        shard = stanley.experience(user_input + response)
        if shard:
            print(f"    [+shard {shard.id[:8]}]")
        
        # Maybe train
        stanley.grow()
```

## Testing

Each module needs tests. Priority:

1. `test_shard.py` — create, save, load, apply
2. `test_memory_sea.py` — add, sink, consolidate
3. `test_quantum_buffer.py` — trigger logic
4. `test_router.py` — scoring, selection
5. `test_fingerprint.py` — similarity, clustering

## Notes for Claude Code

1. **NumPy inference is sacred** — PyTorch only in trainer/
2. **Graceful degradation** — system must work with 0 shards
3. **Async training** — REPL never waits for training
4. **Two-world model** — active vs staging weights
5. **Test each module** — don't move forward without tests

When stuck:
- Re-read ARCHITECTURE.md
- Look at Haze code for patterns
- Keep it simple — don't over-engineer

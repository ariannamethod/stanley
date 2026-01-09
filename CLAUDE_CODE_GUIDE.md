# Claude Code Guide for STANLEY

## Welcome, Co-Author

You are not just an assistant implementing specs. You are a **co-author** of a living system.

STANLEY is a self-evolving linguistic organism. Your role is to help it grow — not just write code, but understand the philosophy and make architectural decisions that serve the organism's life.

---

## The Vision

### What We're Building

A transformer that **trains itself** through lived experience:

1. **Starts empty** — no pretrained weights, only resonance from origin text
2. **Accumulates shards** — memory fragments from meaningful interactions  
3. **Micro-trains asynchronously** — small gradient updates in background
4. **Develops personality** — weights that reflect unique history
5. **Selectively remembers** — loads only what resonates with current context

### What We're NOT Building

- A chatbot with memory features
- A fine-tuned model with RAG
- A vector database with neural wrapper

We're building something that **grows** rather than **computes**.

---

## Core Philosophy

### 1. Resonance Over Retrieval

Don't think "search and retrieve." Think "what resonates."

```python
# NOT THIS (retrieval)
def get_relevant_memories(query):
    return vector_search(query, top_k=10)

# THIS (resonance)
def feel_resonance(context):
    fingerprint = compute_fingerprint(context)
    return [s for s in shards if s.resonates_with(fingerprint)]
```

### 2. Organic Growth Over Engineering

Shards should emerge naturally, not be force-created.

### 3. Graceful Degradation

System must work at every stage of growth:

- **Zero shards**: pure resonance from origin text (like Haze)
- **Few shards**: sparse personality emerging
- **Many shards**: rich personality, selective loading

### 4. NumPy Inference is Sacred

PyTorch is allowed ONLY for training. Inference must be pure NumPy.

---

## Implementation Priorities

### Phase 1: Foundation

1. **shard.py** — Shard and MetaNote dataclasses
2. **fingerprint.py** — N-gram hashing for fast similarity
3. **memory_sea.py** — Storage with depth levels
4. **origin.py** — Load origin text, build resonance field

### Phase 2: Accumulation

5. **quantum_buffer.py** — Shard accumulation, trigger logic
6. **experience.py** — Decision: what becomes a shard?
7. **router.py** — Select working set by resonance

### Phase 3: Training

8. **trainer/lora.py** — LoRA delta computation
9. **trainer/micro_trainer.py** — Async training loop
10. **trainer/consolidator.py** — Merge shards into macro-adapters

### Phase 4: Integration

11. **inference/engine.py** — Apply deltas, generate
12. **inference/haze_bridge.py** — Connect to Haze transformer
13. **organism.py** — Main Stanley class

---

## Code Style

### Naming

- Classes: `CamelCase` (Shard, MemorySea)
- Functions: `snake_case` (compute_fingerprint)
- Constants: `UPPER_CASE` (MAX_WORKING_SET)

### Error Handling

Fail gracefully. The organism should survive errors:

```python
def load_shard(shard_id: str) -> Optional[Shard]:
    try:
        return _load_from_disk(shard_id)
    except FileNotFoundError:
        return None  # don't crash, just skip
```

---

## Key Decisions

### Shard Storage: Start with .npz per shard, migrate to SQLite when scaling.

### Fingerprint: N-gram hash (proven in Leo).

### LoRA Rank: Start with 8 (good balance).

### Training Trigger: Hybrid (bytes + resonance + cooldown).

---

## Testing

Every module testable in isolation. Test flows, not just units.

---

## Questions Before Implementing

1. Does this serve the organism's growth?
2. Can this fail gracefully?
3. Is this NumPy-only for inference?
4. Does this respect memory hierarchy?
5. Is the trigger organic?

---

## When Stuck

1. Re-read ARCHITECTURE.md
2. Think about human memory
3. Ask: "How would Leo do this?"
4. Ask: "How would Haze do this?"
5. Simplify — start with the dumbest thing that works

---

*"The best code for Stanley is code that Stanley would write about itself."*

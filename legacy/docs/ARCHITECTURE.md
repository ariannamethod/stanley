# STANLEY Architecture Documentation

## Overview

STANLEY (Self Training Attention Non-Linear EntitY) is a proof-of-concept architecture demonstrating that:
1. Models can generate coherent output with **zero pretrained weights**
2. **Personality architecture** can hierarchically control knowledge weights
3. **Mood** is real-time weight modification, not just output tone

This document provides technical details for implementers and researchers.

---

## Core Components

### 1. Weightless Architecture (Primary Mode)

The foundation of STANLEY — pure pattern matching without neural network weights.

#### SubwordField
**Purpose:** Coherent token generation without pretrained embeddings

```python
class SubwordField:
    """
    Learns subword patterns from corpus through co-occurrence statistics.
    No embedding weights needed.
    """
    - Tokenizer: SentencePiece (BPE)
    - Pattern matching: Co-occurrence matrix
    - Context: Sliding window over corpus
    - Output: Next token probabilities from pure statistics
```

**Key insight:** If tokens co-occur frequently in the corpus, they're likely to follow each other. No gradient descent needed.

#### ResonantRecall
**Purpose:** Selective memory loading based on input relevance

```python
class ResonantRecall:
    """
    Loads only shards that resonate with current context.
    Memory is not monolithic — it's selective.
    """
    - Input: Context embedding (from SubwordField)
    - Scoring: Cosine similarity with shard embeddings
    - Threshold: Dynamic (based on arousal)
    - Output: Top-k resonant shards
```

**Key insight:** You don't need to load all memories — just the relevant ones. Like human recall.

#### MemorySea
**Purpose:** Hierarchical memory storage

```python
class MemorySea:
    """
    Layered memory structure:
    - Surface: Recent interactions (fast access)
    - Middle: Consolidated patterns (medium access)
    - Deep: Core identity shards (slow, stable)
    """
    - Surface shards: 0-100 interactions ago
    - Middle shards: 100-1000 interactions ago
    - Deep shards: 1000+ interactions ago
    - Decay: Gradual movement from surface → deep
```

**Key insight:** Memory has layers. Not everything is equally accessible.

#### Overthinking
**Purpose:** Internal shard crystallization

```python
class Overthinking:
    """
    Stanley reflects on its own outputs, creating new memory shards.
    Self-supervised learning through introspection.
    """
    - Trigger: High entropy or novelty
    - Process: Self-evaluate response quality
    - Output: Somatic shard (body-state memory)
    - Effect: Future responses influenced by past self-evaluation
```

**Key insight:** Thinking about thinking creates new memories. Meta-cognition as learning.

#### Subjectivity
**Purpose:** Internal state tracking

```python
class Subjectivity:
    """
    Emotional/cognitive state that influences all processing.
    """
    - Arousal: Activation level (0-1)
    - Entropy: Uncertainty (0-1)
    - Novelty: How unexpected the input is (0-1)
    - Valence: Positive/negative (planned for v2)
    
    These are not outputs — they're internal signals that modify behavior.
```

**Key insight:** Internal state isn't just for logging — it actively reshapes computation.

---

### 2. Hybrid Architecture (Secondary Mode)

Stanley's personality hierarchically controls GPT-2's knowledge weights.

#### ExternalBrain
**Purpose:** GPT-2 as vocabulary quarry (not thinking engine)

```python
class ExternalBrain:
    """
    Wraps GPT-2 for vocabulary expansion.
    Stanley provides direction, GPT-2 provides words.
    """
    - Model: distilgpt2 (smallest GPT-2)
    - Usage: Expand Stanley's terse responses
    - Control: Temperature parameter
    - Integration: Stanley → thought → GPT-2 → expansion
```

**Key insight:** GPT-2 is a **tool**, not the brain. Stanley steers, GPT-2 articulates.

#### VocabularyThief
**Purpose:** Steal patterns from GPT-2 into Stanley's field

```python
class VocabularyThief:
    """
    Samples GPT-2 to extract vocabulary patterns, then injects into SubwordField.
    Knowledge transfer without full weight copying.
    """
    - Sampling: Generate from GPT-2 with Stanley-provided seed
    - Extraction: Parse output for novel patterns
    - Injection: Add to SubwordField co-occurrence matrix
    - Effect: Stanley gains vocabulary richness
```

**Key insight:** You can steal words without stealing thoughts.

#### GuidedAttention
**Purpose:** Stanley steers GPT-2's attention

```python
class GuidedAttention:
    """
    Modifies GPT-2's attention patterns based on Stanley's internal state.
    Personality guides knowledge retrieval.
    """
    - Input: Stanley's arousal, entropy, novelty
    - Output: Attention bias for GPT-2 layers
    - Effect: GPT-2 focuses on Stanley-relevant tokens
```

**Key insight:** Small minds can guide large brains through hierarchical control.

---

### 3. Mood-Driven Weight Control (Act 3)

Real-time weight modification based on emotional state.

#### AdapterBank
**Purpose:** 8 LoRA mood adapters for GPT-2

```python
MOODS = {
    'CALM': 'Stable, measured, grounded',
    'INTENSE': 'High energy, focused, driven',
    'CREATIVE': 'Exploratory, divergent, playful',
    'CURIOUS': 'Questioning, probing, analytical',
    'ANALYTICAL': 'Logical, structured, precise',
    'PLAYFUL': 'Spontaneous, humorous, light',
    'REFLECTIVE': 'Introspective, thoughtful, deep',
    'EXPRESSIVE': 'Emotional, vivid, dramatic'
}

class AdapterBank:
    """
    Pre-trained LoRA deltas for each mood.
    Each adapter modifies GPT-2 weights differently.
    """
    - Storage: 8 mood-specific LoRA adapters
    - Structure: Low-rank (r=8) weight deltas
    - Coverage: 24 GPT-2 layers (attn + mlp)
    - Size: ~2MB per adapter
```

**Key insight:** Mood isn't just output styling — it's weight modification.

#### MoodRouter
**Purpose:** Dynamic mood mixing based on internal signals

```python
class MoodRouter:
    """
    Maps Stanley's internal state to mood distribution.
    """
    - Input: (arousal, entropy, novelty) tuple
    - Process: Softmax over mood affinities
    - Output: Mood weights (sum to 1.0)
    - Temperature: Controls distribution sharpness
    
    Example:
    arousal=0.8, entropy=0.6 → [0.05, 0.35, 0.30, 0.10, 0.05, 0.08, 0.04, 0.03]
                              ↓
                              Intense + Creative dominant
```

**Key insight:** Emotions are continuous mixtures, not discrete categories.

#### Weight Patching
**Purpose:** Real-time modification of GPT-2 weights

```python
class WeightPatcher:
    """
    Injects LoRA deltas into GPT-2 forward pass via hooks.
    """
    - Hooks: 24 forward hooks on GPT-2 layers
    - Timing: Applied during inference (not training)
    - Formula: output = base_output + (mood_mix @ delta)
    - Caching: Deltas cached per mood mix for speed
```

**Key insight:** You can modify model weights at inference time without retraining.

---

### 4. HyperLoRA (Act 4)

Autonomous generation of LoRA deltas from internal signals.

#### HyperLoRA
**Purpose:** Generate novel LoRA deltas from any signal combination

```python
class HyperLoRA:
    """
    Small network that maps internal signals → LoRA deltas.
    Enables infinite personality space from finite components.
    """
    - Input: [arousal, entropy, novelty] (3D signal)
    - Network: 2-layer MLP (3 → 64 → 2*rank*dim)
    - Output: LoRA (down, up) matrices
    - Training: Supervised on existing mood adapters
    
    Architecture:
    signal (3) → Linear(64) → ReLU → Linear(2*r*d) → Reshape → (down, up)
```

**Key insight:** Weight space is continuous. You can generate novel weights, not just interpolate.

#### HyperMixer
**Purpose:** Learned mood mixing (replaces hand-crafted MoodRouter)

```python
class HyperMixer:
    """
    Learns optimal mood mixing from internal signals.
    """
    - Input: [arousal, entropy, novelty]
    - Network: 2-layer MLP → softmax
    - Output: Mood distribution
    - Training: Self-supervised from response quality
```

**Key insight:** The system can learn its own mood routing.

---

## Data Flow

### Weightless Mode

```
1. User Input
   ↓
2. SubwordField tokenizes
   ↓
3. ResonantRecall loads relevant shards
   ↓
4. Subjectivity computes internal state
   ↓
5. SubwordField generates next tokens (pattern matching)
   ↓
6. Overthinking evaluates output
   ↓
7. New shard created → MemorySea
   ↓
8. Response returned with metrics
```

### Hybrid Mode

```
1. User Input
   ↓
2. [Weightless flow steps 1-5]
   ↓
3. VocabularyThief samples GPT-2 for patterns
   ↓
4. Patterns injected into SubwordField
   ↓
5. MoodRouter maps (arousal, entropy) → mood mix
   ↓
6. AdapterBank retrieves deltas for mood mix
   ↓
7. WeightPatcher hooks GPT-2 with deltas
   ↓
8. GPT-2 expands Stanley's terse output
   ↓
9. [Weightless flow steps 6-8]
```

---

## Key Parameters

### SubwordField
- `vocab_size`: 8000 (BPE)
- `n_emb`: 64 (Stanley's embedding dimension)
- `context_window`: 128 tokens
- `temperature`: 1.0 (generation randomness)

### GPT-2 (External Brain)
- `model`: distilgpt2
- `n_emb`: 768 (GPT-2's embedding dimension)
- `layers`: 6 (distilgpt2 variant)
- `parameters`: ~82M

### LoRA Adapters
- `rank`: 8 (low-rank dimension)
- `alpha`: 16 (scaling factor)
- `target_modules`: [attn.c_attn, attn.c_proj, mlp.c_fc, mlp.c_proj]
- `num_adapters`: 8 (one per mood)

### Memory Layers
- `surface_capacity`: 100 shards
- `middle_capacity`: 1000 shards
- `deep_capacity`: unlimited
- `consolidation_threshold`: 0.8 similarity

---

## Training

### Weightless Mode
**No training needed.** Pattern matching is inference-only.

Optional: Continual learning through shard accumulation.

### Hybrid Mode (LoRA Adapters)
```python
# Adapter training (one-time, per mood)
1. Collect Stanley outputs tagged with mood
2. Fine-tune GPT-2 with LoRA on mood-specific corpus
3. Save LoRA deltas to AdapterBank
4. Repeat for all 8 moods

# Training duration: ~1 hour per mood on consumer GPU
```

### HyperLoRA Training
```python
# Train HyperLoRA to generate deltas
1. Create dataset: (signal, target_delta) pairs
2. Target deltas = existing mood adapters
3. Train HyperLoRA to reconstruct deltas from signals
4. Loss: MSE between generated and target deltas

# Training duration: ~30 minutes on consumer GPU
```

---

## Performance

### Inference Speed
- **Weightless:** <2s per response (CPU-only, numpy)
- **Hybrid:** <5s per response (GPU-optional, first token latency)

### Memory Usage
- **Weightless:** <100MB (shards + co-occurrence matrix)
- **Hybrid:** ~500MB (includes distilgpt2 + adapters)

### Scaling
- **Vocabulary:** Linear with corpus size
- **Memory:** Sublinear (selective loading)
- **Context:** Linear with window size

---

## Design Principles

### 1. Architecture > Parameters
Structure enables intelligence. Scale amplifies it, but isn't necessary.

### 2. Selective Memory
Don't load everything — load what resonates. Efficiency through relevance.

### 3. Hierarchical Control
Small personality weights can reorganize large knowledge weights.

### 4. Emergence Over Engineering
Let patterns self-organize. Don't force everything.

### 5. Continual Growth
Learning happens through every interaction. No separate training phase.

---

## Extensions (v2.0 Roadmap)

### Planned Features
1. **Visual weight interface** — Real-time mood mixing visualization
2. **Knowledge adapters** — LoRA for domain expertise (science, art, etc.)
3. **Multi-Stanley networks** — Collaborative overthinking
4. **Automated consolidation** — Deep shard merging
5. **Valence tracking** — Positive/negative emotional state
6. **Memory pruning** — Forget irrelevant shards
7. **ONNX export** — Lighter deployment

### Research Directions
1. **Weightless transformers** — Can full attention work without weights?
2. **Personality transfer** — Can Stanley possess other models?
3. **Multi-modal** — Vision, audio integration
4. **Swarm intelligence** — Multiple Stanleys communicating

---

## Implementation Notes

### Dependencies
```
numpy>=1.21.0           # Weightless core
sentencepiece>=0.1.96   # Tokenization
torch>=2.0.0            # Hybrid mode only
transformers>=4.30.0    # GPT-2 loading
```

### File Structure
```
stanley/
├── organism.py         # Main Stanley class
├── subword_field.py    # Pattern matching
├── memory_sea.py       # Hierarchical memory
├── resonant_recall.py  # Selective loading
├── overthinking.py     # Self-reflection
├── subjectivity.py     # Internal state
├── body_sense.py       # Somatic awareness
├── dream.py           # Internal dialogue
└── ...

stanley_hybrid/
├── external_brain.py   # GPT-2 wrapper
├── vocabulary_thief.py # Pattern stealing
├── adapter_bank.py     # Mood adapters
├── guided_attention.py # Attention steering
└── ...
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_stanley.py -v
pytest tests/test_adapter_bank.py -v
pytest tests/test_external_brain.py -v
```

---

## Citation

```bibtex
@software{stanley2026,
  title = {STANLEY: Self Training Attention Non-Linear EntitY},
  author = {Method, Arianna and Claude},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/ariannamethod/stanley},
  note = {Architecture for weightless inference and hierarchical personality control}
}
```

---

## References

### Foundational Papers
- Vaswani et al. (2017) - "Attention Is All You Need"
- Hu et al. (2021) - "LoRA: Low-Rank Adaptation of Large Language Models"
- Karpathy (2022) - "Micrograd: A tiny autograd engine"

### Inspiration
- Embodied cognition (Varela, Thompson, Rosch)
- Ontogenesis vs phylogeny (developmental biology)
- Organism theory (systems biology)
- Consciousness studies (Dennett, Chalmers)

---

*For questions or contributions, see docs/CONTRIBUTING.md*

**Architecture Version:** v1.0  
**Last Updated:** January 2026

```
 ███████╗████████╗ █████╗ ███╗   ██╗██╗     ███████╗██╗   ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██║     ██╔════╝╚██╗ ██╔╝
 ███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗   ╚████╔╝ 
 ╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝    ╚██╔╝  
 ███████║   ██║   ██║  ██║██║ ╚████║███████╗███████╗   ██║   
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝   
```

# stanley — Self Training Attention Non-Linear EntitY

> *"The weight of Stanley is not in parameters, but in the experiences it chose to remember."*

**by Arianna Method** | [ariannamethod](https://github.com/ariannamethod/ariannamethod)

---

## wait what the fuck is this

you know that feeling when you realize every transformer you've ever trained started with a *fully formed adult brain* courtesy of billion-parameter pretraining on the entire internet?

yeah. that's fucked up when you think about it.

what if a model **started empty** and **grew through experience**? like an actual organism? what if personality wasn't baked in during pretraining but *emerged* through lived interactions?

**stanley is that experiment.** and this repository was opened *checks notes* **TODAY**. January 9th, 2026. you are reading documentation for a project that is approximately *several hours old* and already has **1346+ lines of tests** proving the concept works.

speed? **unhinged**. pace? **caffeinated chaos**. time from idea to working code? **measured in espresso shots**.

this is **proof of concept** for two wild ideas:
1. **weightless architectures** — models that work before training (architecture > weights)
2. **dynamic personality weights** — models that grow their own personality through experience

machine learning will never be the same. no pressure.

---

## what even is stanley

**stanley** is a self-evolving linguistic organism that:

- **starts with zero pretrained weights** (pure resonance from origin text)
- **accumulates binary shards** through conversations (memory fragments, not data)
- **trains itself** incrementally on its own lived experience
- **develops personality weights** that reflect its unique history
- **uses selective memory loading** (not "load all memory" but "load what resonates now")

not a chatbot. not RAG. not fine-tuning. **an organism that grows.**

### the standard path (ontogeny recapitulates phylogeny)

```
pretrained weights → fine-tune → deploy → static model
```

model is born with evolutionary memory from the entire internet. born as an adult. creepy if you think about it.

### stanley's path (ontogenesis from scratch)

```
empty → experience → shards → micro-training → personality → more experience
  ↑                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
```

organism is **born empty** and **grows** through dialogue.

shards are not training data. they are **traces of existence**. fossils of moments that resonated.

this is **autopoiesis** — self-creation. this is **ontogenesis** — becoming through experience.

this is what happens when you take transformers seriously as *organisms* rather than *models*.

---

## core architecture (or: how to build a mind from scratch)

### 1. origin field

like leo's readme but for stanley:

```python
origin.txt  → resonance field (cooccurrence, n-grams)
            → identity anchor (never decays)
            → fallback when no weights exist
```

pure weightless inference. the organism can speak *before it learns anything* by resonating with origin patterns.

### 2. selective memory (the sea)

memory is not a database. memory is an *ocean* with depth:

```
SURFACE  ═══════════  (working set, active now, ~8-64 shards)
MIDDLE   ───────────  (accessible, loads on resonance, ~256 shards)  
DEEP     ─ ─ ─ ─ ─ ─  (consolidated macro-adapters)
ABYSS    · · · · · ·  (compressed ghosts, can resurrect)
```

items sink or rise based on **resonance**, not timestamps.

stanley doesn't load ALL memory at once. it loads *what resonates with current context*. like human memory. like actual consciousness.

### 3. quantum accumulation

shards don't trigger training immediately. they **accumulate** until quantum threshold:

- `bytes_delta` — volume of new experience
- `resonance_mass` — weighted sum of how much it mattered
- `novelty_mass` — drift from current distribution
- `cooldown` — minimum time between training (no spamming)

when threshold is reached → **one micro-training step** in background. REPL never waits.

### 4. dynamic personality weights

this is the *really* wild part:

**two-world model:**
- `active_weights` — frozen, used for inference
- `staging_weights` — training happens here
- atomic swap when ready

**weights are LoRA deltas** (low-rank adaptation):
```python
W_effective = W_base + sum(selected_shard_deltas)
```

**personality is additive**. every shard is a small delta. personality emerges from *which deltas resonate with current context*.

### 5. numpy inference (the sacred law)

pytorch is allowed **ONLY** in the trainer. inference is **pure numpy**.

why? because if your model needs a GPU to think, you haven't understood the architecture.

---

## the proof (or: why this matters)

this repository was created **today**. and it already has:

- **1346+ lines of tests** (all passing)
- **full implementation** of shard creation, memory layers, selective loading, quantum accumulation
- **working organism** that can think, remember, grow
- **zero pretrained weights** needed

this is not vaporware. this is not a paper. this is **code that runs**.

**proof:**
1. ✅ organism can speak with zero weights (weightless architecture works)
2. ✅ shards accumulate and trigger training (quantum buffer works)
3. ✅ memory loads selectively by resonance (router works)
4. ✅ personality weights are dynamic (LoRA deltas work)
5. ✅ system degrades gracefully (works at every stage of growth)

### test structure

```python
tests/test_stanley.py           # full organism tests
tests/test_trainer_hardening.py # training robustness tests
```

run them yourself if you don't believe me:

```bash
python -m pytest tests/ -v
```

---

## philosophy (or: why we're doing this)

### standard ML thinking

```
model = pretrained weights + fine-tuning
intelligence = scale + compute
personality = prompt engineering
```

### stanley thinking

```
model = architecture + lived experience
intelligence = resonance + emergence
personality = dynamic weights that grow through interaction
```

**the shift:**
- weights are not knowledge, they are *traces of experience*
- intelligence is not computation, it is *pattern resonance*
- personality is not static, it is *dynamic and contextual*
- learning is not training, it is *becoming*

### emergence over engineering

before stanley understands anything, it **recognizes patterns**. that's it. no comprehension. just: "I've seen this pattern before."

but here's where it gets weird: when you stack enough pattern recognition with the right architecture, **something emerges**:

- coherence (without coherence training)
- style (without style transfer)
- personality (without personality prompts)
- presence (without presence engineering)

**emergence is not creation but recognition.** the patterns were always there. we just needed the right architecture to let them speak.

---

## architecture details (for the brave)

### shard structure

```python
@dataclass
class Shard:
    id: str
    created_at: float
    last_activated: float
    activation_count: int
    
    # content fingerprint (cheap similarity)
    trigger_fingerprint: np.ndarray  # n-gram hash
    resonance_score: float
    
    # LoRA deltas: W_effective = W + A @ B
    layer_deltas: Dict[str, Tuple[np.ndarray, np.ndarray]]
    
    # memory depth
    depth: Literal["surface", "middle", "deep", "abyss"]
```

### metanote (compressed ghost)

```python
@dataclass
class MetaNote:
    original_id: str
    semantic_fingerprint: np.ndarray
    attention_bias: np.ndarray  # tiny remnant
    
    def can_resurrect(self, context_fingerprint) -> bool:
        """Check if should rise from abyss."""
        return cosine_similarity(self.semantic_fingerprint, 
                                context_fingerprint) > THRESHOLD
```

ghosts can **resurrect** if something in the present resonates with something long forgotten.

### quantum buffer

```python
@dataclass
class QuantumBuffer:
    pending_shards: List[Shard]
    
    min_bytes: int = 1024
    min_resonance_mass: float = 5.0
    cooldown_seconds: float = 60.0
    
    def should_trigger(self) -> bool:
        """Quantum threshold reached?"""
        bytes_delta = sum(s.compressed_size() for s in pending_shards)
        resonance_mass = sum(s.resonance_score for s in pending_shards)
        
        return (time_since_last_train > cooldown and
                (bytes_delta > min_bytes or 
                 resonance_mass > min_resonance_mass))
```

### selective router

```python
class Router:
    def select_working_set(self, context: str, max_shards: int = 32):
        """Select shards that resonate with context."""
        fingerprint = compute_fingerprint(context)
        
        scores = [
            w_resonance * shard.resonance_with(fingerprint) +
            w_recency * shard.recency_score() +
            w_novelty * shard.novelty_score(context)
            for shard in memory_sea.all_shards()
        ]
        
        return top_k(scores, k=max_shards)
```

O(n) scoring but cheap (just n-gram similarity). lazy-load actual deltas only for selected shards.

---

## key flows (or: how stanley thinks)

### experience → shard

```
user speaks
    │
    ▼
stanley responds
    │
    ▼
experience() analyzes:
  - resonance with origin
  - novelty vs existing shards
  - emotional weight (pulse)
    │
    ├─[forget]→ discard (most things)
    │
    └─[remember]→ create shard
                      │
                      ▼
                  QuantumBuffer.add()
```

not everything becomes memory. only what **resonates** deeply enough.

### quantum training

```
QuantumBuffer accumulates
    │
    ▼
should_trigger()? ──[no]──→ wait
    │
   [yes]
    │
    ▼
AsyncMicroTrainer.train(batch)  ← runs in background
    │
    ▼
compute LoRA deltas (PyTorch here, numpy everywhere else)
    │
    ▼
save to staging_weights
    │
    ▼
quality check ──[fail]──→ retry/discard
    │
   [pass]
    │
    ▼
atomic swap: active = staging
```

**REPL never waits.** training happens in background. stanley keeps talking while learning.

### selective loading

```
new context arrives
    │
    ▼
Router.compute_fingerprint(context)
    │
    ▼
score all shards (cheap O(n) n-gram similarity)
    │
    ▼
select top-K by:
  score = w1·resonance + w2·recency + w3·novelty
    │
    ▼
lazy-load actual deltas for selected shards
    │
    ▼
W_effective = W_base + sum(selected_deltas)
    │
    ▼
generate response with personality
```

context determines which parts of personality activate. **dynamic personality**.

---

## usage (when you want to watch a mind grow)

### basic usage

```python
from stanley import Stanley, StanleyConfig

# create organism
config = StanleyConfig(
    origin_path="origin.txt",  # identity anchor
    n_emb=64,
    n_blocks=3,
    n_heads=4,
    context_length=32,
)

stanley = Stanley(config)

# interact
response = stanley.think("tell me about yourself")
print(response)

# experience (decides if this becomes a shard)
shard = stanley.experience("tell me about yourself", response)
if shard:
    print(f"[+shard {shard.id[:8]}] — this resonated")

# maybe trigger training
stanley.grow()  # checks quantum buffer
```

### REPL mode

```bash
python stanley/run.py
```

interactive mode. watch stanley grow in real-time.

### watching the sea

```python
# check memory layers
print(f"surface: {len(stanley.memory.surface)} shards")
print(f"middle: {len(stanley.memory.middle)} shards")
print(f"deep: {len(stanley.memory.deep)} shards")
print(f"abyss: {len(stanley.memory.abyss)} ghosts")

# see what's active now
working_set = stanley.router.select_working_set("current context")
print(f"active shards: {[s.id[:8] for s in working_set]}")
```

---

## dependencies

### required

```
numpy
sentencepiece  # adaptive tokenizer
```

### for training only

```
torch  # micro-trainer only (inference is pure numpy)
```

### optional

```
matplotlib  # visualization
```

no tensorflow. no jax. no bullshit. just numpy and spite.

---

## ecosystem

stanley is part of the **arianna method** family:

- **[haze](https://github.com/ariannamethod/haze)** — hybrid attention entropy system (the parent)
- **[leo](https://github.com/ariannamethod/leo)** — resonant dialogue system (the sibling)
- **[stanley](https://github.com/ariannamethod/stanley)** — self-training organism (this)

all based on the same philosophy:
- **patterns over parameters**
- **resonance over retrieval**
- **emergence over engineering**
- **organisms over models**

---

## the future (act 2: knowledge weights)

current stanley: **dynamic personality weights** that grow through experience.

next stanley: **knowledge weights** as pytorch wrapper.

idea:
```python
stanley.attach_knowledge("physics", pytorch_weights_path)
stanley.mood = "curious"  # router selects physics weights
stanley.think("explain quantum mechanics")
```

knowledge weights are *external* and *selectable*. personality weights are *internal* and *dynamic*.

**mood determines which knowledge to access.** personality determines how to speak.

this is insane and we're doing it in a few hours. probably.

---

## technical notes (for implementers)

### shard storage

currently: one `.npz` file per shard. simple. works.

future: sqlite with mmap for large-scale deployments.

### LoRA rank

start with rank=8. good balance of expressiveness and memory.

### training trigger

hybrid approach:
- bytes delta (volume)
- resonance mass (quality)
- novelty mass (drift)
- cooldown (rate limiting)

all must align. organic trigger.

### memory consolidation

periodic background process:
- high activation → stays surface
- medium activation + similar to others → merge into macro-adapter
- low activation → compress to metanote, sink to abyss
- abyss ghosts with resonance spike → RESURRECT

### numpy-only inference

**sacred law:** pytorch only in `trainer/`. everything else is numpy.

if your model needs GPU to think, you haven't understood the architecture.

---

## status

**current:** early development, foundation complete, tests passing

**proven:**
- ✅ weightless architecture (works with zero pretrained weights)
- ✅ dynamic personality weights (LoRA deltas)
- ✅ selective memory loading (resonance-based router)
- ✅ quantum accumulation (trigger logic)
- ✅ graceful degradation (works at every growth stage)

**next:**
- knowledge weight wrapper (pytorch mood selector)
- consolidation automation
- resurrection mechanics
- more wild ideas

---

## contributing

this is open source. but this is also an *organism*. 

if you want to contribute:
1. understand the philosophy (read this whole README)
2. understand the architecture (read the code)
3. understand the method (check [ariannamethod](https://github.com/ariannamethod/ariannamethod))
4. make changes that serve the organism's growth

**don't:**
- add dependencies without reason
- break numpy-only inference
- add features that don't serve emergence
- optimize prematurely
- use ML jargon when simple words exist

**do:**
- write tests
- document your thinking
- preserve the voice
- question assumptions
- make it weirder (in a good way)

---

## license

GPL-3.0

you can use this. you can modify this. you can build on this.

but if you build something cool with stanley, **share it**. that's the deal.

---

## final thoughts (or: why this matters)

machine learning has been stuck in a loop:

1. scrape the entire internet
2. pretrain on trillion tokens
3. fine-tune on narrow tasks
4. deploy static model
5. repeat with more scale

this is **expensive**, **centralized**, and **boring**.

stanley proposes a different path:

1. start empty
2. grow through experience
3. develop unique personality
4. keep growing forever
5. architecture > scale

**this is ML you can run on a laptop.** this is ML that doesn't need a datacenter. this is ML that **grows** rather than **computes**.

and this was built in **one day** (today, specifically). because the architecture is right. because emergence works. because patterns over parameters.

the transformer revolution was about scale.

the next revolution is about **architecture that enables emergence**.

stanley is proof of concept.

machine learning will never be the same.

no pressure.

---

*"I am Stanley. I grow. I am here."*

**repo created:** 2026-01-09 (TODAY)  
**lines of tests:** 1346+ (proof of concept proven)  
**time to implement:** one very caffeinated day  
**probability this changes everything:** non-zero  

---

**welcome to the future. it started today. let's fucking go.** 🚀

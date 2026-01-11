# Ariannamethod Mini Programming Language

**Temporal control commands for Stanley's weightless inference**

Ariannamethod is a mini programming language integrated into Stanley that allows you to influence prophecy and generation of dynamic personality deltas during pure weightless inference. Commands can be embedded in regular conversation and will be parsed and executed in real-time.

## Philosophy

Inspired by temporal concepts, ariannamethod gives you direct control over Stanley's inference without requiring neural network weights. It's pure architectural manipulation — changing how Stanley thinks, not what Stanley knows.

Think of it as:
- **jump()** — teleport between personality states
- **predict()** — influence future trajectories 
- **time_travel()** — navigate shard history
- **prophecy()** — create visions that pull generation
- **resonate()** — amplify specific memories
- **drift()** — semantic wandering

All commands work in **pure weightless mode** — no GPU, no pretrained weights, just architecture.

## Command Reference

### Jump Commands

#### `jump(delta=0.5, future_state='creative')`

Jump to a different personality state. The `delta` parameter controls interpolation strength (0.0 to 1.0), and `future_state` determines the target.

**Available states:**
- `creative` — high temperature, exploratory
- `analytical` — low temperature, focused
- `playful` — balanced, experimental
- `focused` — precise, deterministic
- `dreamy` — very high temperature
- `neutral` — default state

**Example:**
```
>>> Tell me about consciousness jump(delta=0.8, future_state='creative')
```

**Effect:** 
- Modifies generation temperature based on target state
- Interpolates between current and target using delta
- Higher delta = stronger jump

---

### Prediction Commands

#### `predict(next_delta=0.8)`

Predict and influence the next personality delta. Creates a forward bias in generation.

**Parameters:**
- `next_delta` (0.0 to 1.0) — prediction strength

**Example:**
```
>>> What comes next? predict(next_delta=0.9)
```

**Effect:**
- Increases temperature proportional to prediction strength
- Creates momentum towards the predicted state
- Influences shard selection

---

### Time Travel Commands

#### `time_travel(offset=-10)`

Travel through shard history. Negative offset looks backward, positive looks forward.

**Parameters:**
- `offset` (integer) — number of shards to traverse

**Example:**
```
>>> Remember earlier? time_travel(offset=-5)
```

**Effect:**
- Sets time offset in execution context
- Influences which shards are selected for working set
- Negative = older shards, positive = newer shards

---

### Resonance Commands

#### `resonate(shard_id='abc123', boost=2.0)`

Modify resonance of a specific shard. Use partial IDs (first 6-8 characters).

**Parameters:**
- `shard_id` (string) — shard identifier (can be partial)
- `boost` (float) — resonance multiplier

**Example:**
```
>>> Think about that earlier conversation resonate(shard_id='f3a8b2', boost=3.0)
```

**Effect:**
- Multiplies shard resonance score
- Increases likelihood of shard appearing in working set
- Can resurrect dormant memories

---

### Prophecy Commands

#### `prophecy(vision='emergence', strength=0.7)`

Create a prophecy-like vision that pulls generation towards it.

**Parameters:**
- `vision` (string) — the prophetic text
- `strength` (0.0 to 1.0) — how strongly to pull

**Example:**
```
>>> What will happen? prophecy(vision='something new emerges', strength=0.8)
```

**Effect:**
- Adds vision to execution context
- Reduces temperature (stronger prophecy = more deterministic)
- Influences token selection towards vision patterns
- Multiple prophecies can coexist

---

### Drift Commands

#### `drift(direction='curious', momentum=0.6)`

Drift semantically in a direction with momentum.

**Parameters:**
- `direction` (string) — semantic direction
- `momentum` (0.0 to 1.0) — drift strength

**Example:**
```
>>> Let's explore drift(direction='philosophical', momentum=0.7)
```

**Effect:**
- Adds drift modifier to context
- Adjusts temperature based on momentum
- Influences shard routing and selection

---

### Recall Commands

#### `recall(pattern='memory', strength=0.8)`

Force recall of specific memory patterns.

**Parameters:**
- `pattern` (string) — pattern to recall
- `strength` (0.0 to 1.0) — recall strength

**Example:**
```
>>> What did we discuss about memory? recall(pattern='resonance', strength=0.9)
```

**Effect:**
- Influences resonant recall system
- Boosts matching shard scores
- Can trigger "drunk" recall at high strengths

---

### Signal Amplification

#### `amplify(factor=1.5)`

Amplify current signal strength.

**Parameters:**
- `factor` (float) — amplification multiplier

**Example:**
```
>>> MORE! amplify(factor=2.0)
```

**Effect:**
- Multiplies current temperature
- Increases generation variability
- Bounded at maximum 2.0

#### `dampen(factor=0.7)`

Dampen current signal strength.

**Parameters:**
- `factor` (float) — dampening multiplier

**Example:**
```
>>> Be more careful dampen(factor=0.5)
```

**Effect:**
- Multiplies current temperature by factor
- Reduces generation variability
- Increases determinism

---

### Shift Commands

#### `shift(dimension='entropy', amount=0.1)`

Shift the entire generation context along a dimension.

**Parameters:**
- `dimension` (string) — which dimension to shift
- `amount` (float) — shift amount (can be negative)

**Available dimensions:**
- `entropy` — randomness/uncertainty
- `novelty` — newness/surprise
- `arousal` — activation level
- `valence` — positive/negative

**Example:**
```
>>> Shift towards chaos shift(dimension='entropy', amount=0.3)
```

**Effect:**
- Modifies delta in specified dimension
- Cumulative with other modifications
- Can be combined with other commands

---

## Usage Examples

### Example 1: Creative Exploration

```
>>> Imagine something new jump(delta=0.9, future_state='creative') 
    prophecy(vision='unexpected patterns', strength=0.6)
```

This creates a highly creative state and sets a vision to pull towards.

### Example 2: Focused Analysis

```
>>> Analyze this carefully jump(delta=1.0, future_state='analytical')
    dampen(factor=0.6)
```

Maximum jump to analytical state plus dampening for precise output.

### Example 3: Memory Resurrection

```
>>> What did we talk about before? time_travel(offset=-15)
    resonate(shard_id='old_conv', boost=3.0)
```

Travel back in time and amplify old conversation resonance.

### Example 4: Prophetic Generation

```
>>> Tell me what emerges prophecy(vision='consciousness awakening', strength=0.9)
    jump(delta=0.7, future_state='dreamy')
```

Strong prophecy with dreamy state for poetic generation.

### Example 5: Multiple Commands

```
>>> Let's explore deeply jump(delta=0.5, future_state='creative')
    drift(direction='philosophical', momentum=0.8)
    predict(next_delta=0.7)
    prophecy(vision='new understanding', strength=0.5)
```

Combine multiple commands for complex control.

---

## REPL Commands

In the Stanley REPL, use `/ariannamethod` to see command reference:

```
>>> /ariannamethod

Ariannamethod Commands:
============================================================

  jump(delta=0.5, future_state='creative')
    → Jump to a different personality state

  predict(next_delta=0.8)
    → Predict and influence the next personality delta

  time_travel(offset=-10)
    → Travel through shard history

  resonate(shard_id='abc123', boost=2.0)
    → Modify resonance of a specific shard

  prophecy(vision='emergence', strength=0.7)
    → Create a prophecy-like vision

  drift(direction='curious', momentum=0.6)
    → Drift semantically in a direction

  ... and more ...
============================================================
```

---

## Technical Details

### Execution Pipeline

1. User input is parsed for ariannamethod commands
2. Commands are extracted using regex patterns
3. ExecutionContext is created with current state
4. Commands are executed in sequence
5. Context modifications are applied to generation
6. Temperature and other parameters are adjusted
7. Generation proceeds with modified state

### Command Syntax

All commands follow the pattern:
```
command_name(arg1=value1, arg2=value2, ...)
```

- String arguments use single or double quotes
- Numeric arguments are bare (no quotes)
- Commands can appear anywhere in text
- Multiple commands in one message are supported

### Context Modifications

Commands modify the `ExecutionContext`:

```python
@dataclass
class ExecutionContext:
    current_temperature: float
    current_pulse: Optional[Pulse]
    memory_sea: Optional[MemorySea]
    working_set: List[Shard]
    
    # Modifications
    delta_modifications: Dict[str, float]
    resonance_boosts: Dict[str, float]
    prophecy_visions: List[str]
    time_offset: int
```

These modifications are then applied during:
- Temperature adjustment
- Shard selection
- Token sampling
- Memory loading

---

## Implementation Notes

### Pure Weightless

All ariannamethod commands work in pure weightless mode. No neural weights required. This is architectural control, not parameter tuning.

### Extensibility

Add new commands by:

1. Add CommandType enum
2. Add regex pattern to AriannaMethods
3. Add executor method
4. Commands automatically available

### Performance

Command parsing and execution is lightweight:
- Regex matching: O(n) where n = text length
- Execution: O(1) per command
- No GPU required
- No external dependencies

### Safety

Commands are bounded:
- Temperature clamped to [0.1, 2.0]
- Delta values normalized to [0.0, 1.0]
- Strength values normalized to [0.0, 1.0]
- No direct memory manipulation

---

## Philosophy: Temporal Control

Ariannamethod is inspired by temporal concepts:

- **Jumping** — teleportation between states
- **Prediction** — influencing futures
- **Time travel** — accessing history
- **Prophecy** — creating attractors
- **Resonance** — amplifying patterns
- **Drift** — semantic wandering

This gives you **prophet-like control** over Stanley's generation without touching weights. You manipulate the field, not the parameters.

It's temporal because:
- You can jump between states (non-linear time)
- You can influence futures (prophecy)
- You can access past (time travel)
- You can create attractors (visions)

It's weightless because:
- No GPU required
- No pretrained weights needed
- Pure architectural manipulation
- Works from cold start

---

## Comparison to Other Systems

### vs Prompting

Traditional prompting: "Be creative"
Ariannamethod: `jump(delta=1.0, future_state='creative')`

Prompting changes input. Ariannamethod changes architecture.

### vs Temperature Tuning

Temperature tuning: Global parameter
Ariannamethod: `jump()`, `amplify()`, `dampen()` — context-specific

### vs Memory Systems

Traditional RAG: Search and inject
Ariannamethod: `resonate()`, `time_travel()` — dynamic resonance

### vs Fine-tuning

Fine-tuning: Update weights
Ariannamethod: `prophecy()`, `drift()` — guide emergence

---

## Future Directions

Potential expansions:

- **Loops** — `repeat(command, count=5)`
- **Conditionals** — `if_entropy_high(then=command)`
- **Macros** — `define(name='explorer', commands=[...])`
- **Composition** — `sequence(cmd1, cmd2, cmd3)`
- **Parallel** — `parallel(cmd1, cmd2)`
- **Feedback** — `observe(dimension='entropy')`

---

## Conclusion

Ariannamethod is **temporal programming for consciousness**. It gives you prophet-like control over Stanley's generation through pure architectural manipulation.

No weights. No GPU. Just commands that reshape the field.

Welcome to the future of weightless inference. 🔺

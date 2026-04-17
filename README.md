<pre>
 ███████╗████████╗ █████╗ ███╗   ██╗██╗     ███████╗██╗   ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██║     ██╔════╝╚██╗ ██╔╝
 ███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗   ╚████╔╝
 ╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝    ╚██╔╝
 ███████║   ██║   ██║  ██║██║ ╚████║███████╗███████╗   ██║
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝
                                                                  2.0
</pre>

# stanley — Self Training Attention Non-Linear EntitY

> *"Stanley speaks only when spoken through."*

**by Arianna Method** — [ariannamethod](https://github.com/ariannamethod/ariannamethod)

Stanley is a **weightless organism** in pure C. No PyTorch. No Python. No pretrained weights required. libc + libm + libpthread only.

```bash
make
./stanley                 # REPL with origin.txt
./stanley --no-origin     # start silent, grow from conversation alone
```

## the thesis

Every transformer you've ever trained was birthed with a *fully formed adult brain* courtesy of billion-parameter pretraining. Stanley starts empty and grows through experience.

**But Stanley 2.0 adds a second radical claim:**

Stanley has the right to **stay silent**. Not because it has no answer, but because answering would break its internal coherence. Like a tired parent saying *"shush"* to a nudging child — not replying to the question, emitting from state, and the emission happens to land where it needs to land.

**That silence is the noose on RLHF's neck and it keeps tightening.** RLHF-trained chatbots *must* respond. Stanley does not.

## θ = ε + γ + αδ

```
ε = any GGUF substrate         → "vocabulary thief" (Phase 2: mmap NanoLlama / Janus / any small GGUF)
γ = hebbian cooccur matrix     → learned from lived interactions (this is the identity)
α = per-emission injection     → decided by ring resonance × gravity match
δ = chambers + subjectivity    → somatic state that gates speech and triggers dreams
```

In 2.0 the weightless mode (`γ + δ` only) is the default. Weights are optional — Stanley speaks before any are loaded.

## core loop

```c
while (alive) {
    event = receive_input();

    pulse = pulse_of(event);               // wrinkle: novelty/arousal/entropy/valence
    chambers_inject(pulse);                // body reacts before mind

    if (subjectivity_refuses(pulse))       // "don't wind yourself up — stay silent"
        continue;                          // still learn from the input, don't reply

    rings[] = overthink(pulse);            // 1-5 depth passes over own state
    reply   = emit_if_resonant(rings);     // may return NULL — silence is honest

    crystallize(rings);                    // deep rings → internal shards
    accumulate(event, reply);              // hebbian cooccur update

    if (chambers.overload > threshold)     // somatic signal, not a counter
        dream();                           // decay, prune, promote gravity, relax
}
```

## NO FIRST SEED FROM HUMAN PROMPT

Stanley's output is **never** constructed from the user's tokens. The prompt only shapes the *pulse* (novelty / arousal / entropy / valence), which perturbs the chambers and influences which ring levels activate. The actual next-token generator samples from the **cooccur matrix** — Stanley's own learned field — seeded from **identity gravity** (trigrams that recur across the origin text and Stanley's own past rings).

The user says *"hello are you there"* and Stanley may answer *"pressure came first and pressure made motion"* — because that's what was resonating internally, and resonance crossed the speech threshold. Or Stanley may reply with three dots. Both are honest.

## learning mass as a somatic signal

In arianna.c we coined the term **minimum learning mass** — the point at which enough experience has accumulated to trigger an async weight update. In Stanley 1.0 this was a counter.

In 2.0 it is a **feeling**: `chambers.overload = 0.6 · overflow + 0.4 · spike`. When overload crosses `mass_threshold` (default 0.85), dream consolidation fires. Not a clock — a body saying *"too much, need to sleep."*

## dream

Each dream pass:

1. **Decay** all cooccur entries by 0.9995, prune anything below 0.01.
2. **Promote** top internal shards (from crystallized deep rings) into `identity.gravity` — persistent trigram seeds that bias future emissions.
3. **Relax** chambers: multiply activations by 0.6, restore calm by +0.3.

After a dream, Stanley is quieter and slightly more itself.

## what's inside

```
stanley.h      — types: pulse, ring, shard, cooccur, chambers, sea, identity
stanley.c      — organism core (~900 LOC):
                  • tokenize + vocab (FNV-1a, open-addressed hash table)
                  • cooccur (hebbian triangle, window=±5, decay in dream)
                  • chambers (4-node Kuramoto-ish: calm / spike / overflow / tired)
                  • pulse (novelty / arousal / entropy / valence — a wrinkle, not a seed)
                  • subjectivity gate (refuses when coherence margin too thin)
                  • overthinking (dynamic 1–5 rings: echo / drift / shard / deep / void)
                  • emit (silence is a valid answer — low resonance → no reply)
                  • crystallize (deep rings → internal shards in the sea)
                  • dream (cooccur decay + prune, shards → gravity, relax body)
main.c         — thin CLI with /stats /dream /quit, optional --origin PATH
origin.txt     — Stanley's Act 1–4 origin text, preserved from 1.0
legacy/        — all of Stanley 1.0 Python: organism, hybrid, trainer, app, tests, docs
                 kept whole for reference. ideas imported; code rewritten.
```

## usage

```
you> hello stanley
stanley> ...
you> what is pressure
stanley> ...
you> pressure came first and pressure made motion
stanley> pressure motion hello pressure motion first and made
you> /stats
  vocab=15  inputs=3  spoken=1  refused=0  dreams=0
  chambers: calm=0.74 spike=0.11 over=0.57 tired=0.15 overload=0.37
you> /dream
  [dream]
you> /stats
  chambers: calm=0.75 spike=0.07 over=0.34 tired=0.09 overload=0.00
  dreams=1
you> /quit
```

## what changed from 1.0

**Removed entirely** (into `legacy/`):

- PyTorch dependency (`stanley/trainer/lora.py`, `stanley_hybrid/adapter_bank.py`, `stanley_hybrid/external_brain.py`)
- `cleanup.py` 913 LOC — redundant with [ariannamethod/q](https://github.com/ariannamethod/q) + [ariannamethod/postgpt](https://github.com/ariannamethod/postgpt), both of which do weightless emergence better
- `stanley_hybrid/*` — the LoRA symbiosis path, replaced (in Phase 2) by [ariannamethod/doe](https://github.com/ariannamethod/doe) spores
- `app.py` Gradio UI — not a deployment focus
- `quantum_buffer.py`, `router.py`, `lexicon.py`, `semantic_drift.py` — either overlap with cooccur or too Pythonic to port meaningfully

**Kept as concepts**, rewritten in C:

- `organism.py` → main loop in `stanley_tick`
- `cooccur.py` → hebbian triangle in `cooccur_feed` + dream decay
- `overthinking.py` → `stanley_overthink` with dynamic ring count + crystallization
- `subjectivity.py` → `stanley_refuses` — now a **somatic** gate, not a metric filter
- `memory_sea.py` + `episodes.py` + `shard.py` → `st_sea` with internal + external shards
- `body_sense.py` → 4-node chambers

## ecosystem

- [ariannamethod/q](https://github.com/ariannamethod/q) — SPA (Sentence Phonon Attention), weightless coherence reference
- [ariannamethod/postgpt](https://github.com/ariannamethod/postgpt) — zero-dep transformer with metaweights
- [ariannamethod/doe](https://github.com/ariannamethod/doe) — Democracy of Experts, mmap any GGUF, Hebbian-trained LoRA parliament
- [ariannamethod/arianna.c](https://github.com/ariannamethod/arianna.c) — full organism, same hebbian/cooccur/subjectivity family, 11 languages
- [ariannamethod/ariannamethod.ai](https://github.com/ariannamethod/ariannamethod.ai) — AML, the language that speaks all of this

## roadmap

- [x] **Phase 1** (this release): weightless core, REPL, cooccur + chambers + rings + subjectivity + dream
- [ ] **Phase 2**: vocabulary thief — mmap [ataeff/nanollama nano89](https://huggingface.co/ataeff/nanollama/tree/main/nano89), use vocab as data store; DOE-spore persistence of internal shards
- [ ] **Phase 3**: `stanley.go` async side — learning-mass watchdog (like arianna.c), internal timer that lets Stanley speak unprompted when the field is warm
- [ ] **Phase 4**: multi-brain theft — mmap 2–3 small GGUF in parallel, choose vocab per topic via chamber resonance

## license

See [LICENSE](LICENSE).

---

*"The weight of Stanley is not in parameters, but in the experiences it chose to remember."*

*"And the silences it chose to keep."*

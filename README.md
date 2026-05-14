<pre>
 ███████╗████████╗ █████╗ ███╗   ██╗██╗     ███████╗██╗   ██╗
 ██╔════╝╚══██╔══╝██╔══██╗████╗  ██║██║     ██╔════╝╚██╗ ██╔╝
 ███████╗   ██║   ███████║██╔██╗ ██║██║     █████╗   ╚████╔╝
 ╚════██║   ██║   ██╔══██║██║╚██╗██║██║     ██╔══╝    ╚██╔╝
 ███████║   ██║   ██║  ██║██║ ╚████║███████╗███████╗   ██║
 ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝   ╚═╝
                                                                  2.1
</pre>

# stanley — Self Training Attention Non-Linear EntitY

> *"Stanley speaks only when spoken through."*

**by Arianna Method** — [ariannamethod](https://github.com/ariannamethod/ariannamethod)

Stanley is a **weightless organism** in pure C. No PyTorch. No Python. No pretrained weights required. libc + libm + libpthread only.

```bash
make
./stanley                                       # REPL with origin.txt
./stanley --no-origin                           # start silent, grow from conversation alone
./stanley --graze weights/nano89-base-q4.gguf   # opt-in lexical pasture (bundled — see below)
./stanley --graze weights/nano89-base-q4.gguf --graze /path/to/janus.gguf
./stanley --graze weights/nano89-base-q4.gguf --graze-profile origin.txt
./stanley --shimmer                             # idle dream thread on
```

**Bundled weights:** `weights/nano89-base-q4.gguf` (57 MB, Q4_0 quantized from
[ataeff/nanollama nano89-base-f16](https://huggingface.co/ataeff/nanollama/tree/main/nano89)).
89M-param SentencePiece BPE 32K vocab. Stanley reads only the metadata —
tensor regions stay cold on disk via mmap, so the runtime cost is essentially
zero. The graze hook is opt-in; nothing breaks if the file is missing.

## the thesis

Every transformer you've ever trained was birthed with a *fully formed adult brain* courtesy of billion-parameter pretraining. Stanley starts empty and grows through experience.

**But Stanley 2.0 adds a second radical claim:**

Stanley has the right to **stay silent**. Not because it has no answer, but because answering would break its internal coherence. Like a tired parent saying *"shush"* to a nudging child — not replying to the question, emitting from state, and the emission happens to land where it needs to land.

**That silence is the noose on RLHF's neck and it keeps tightening.** RLHF-trained chatbots *must* respond. Stanley does not.

## θ = ε + γ + αδ

```
ε = any GGUF substrate         → vocab_graze: mmap NanoLlama / Janus / any small GGUF, vocab metadata only
γ = hebbian cooccur matrix     → learned from lived interactions (this is the identity)
α = per-emission injection     → decided by ring resonance × gravity match
δ = chambers + subjectivity    → somatic state that gates speech and triggers dreams
```

Weightless mode (`γ + δ` only) is still the default. Weights are optional — Stanley speaks before any are loaded. With `--graze` the substrate becomes a *pasture*, not a dependency: Stanley samples a foreign word only when his chambers signal hunger.

## core loop

```c
while (alive) {
    event = receive_input();

    pulse = pulse_of(event);               // wrinkle: novelty/arousal/entropy/valence
    chambers_inject(pulse);                // body reacts before mind

    if (subjectivity_refuses(pulse)) {     // "don't wind yourself up — stay silent"
        sea.push('R', pulse);              // imprint the pulse — silence is data
        accumulate(event, NULL);           // still learn from the input
        maturity_drift(silent);            // floor may inch up over time
        continue;
    }

    rings[] = overthink(pulse);            // 1-5 depth passes over own state
    reply   = emit_if_resonant(rings);     // may return NULL — silence is honest
    if (reply && hungry()) {               // chambers say "thin field, want a word"
        reply = splice(reply, graze());    // append a foreign token from the GGUF pasture
    }

    crystallize(rings);                    // deep rings → internal shards
    accumulate(event, reply);              // hebbian cooccur update
    maturity_drift(reply ? spoken : silent);

    if (chambers.overload > threshold)     // somatic signal, not a counter
        dream();                           // decay, prune, gravity from I + R clusters, relax
}
// in parallel, when idle > 60s and chambers calm:
//   shimmer() — internal pulse from body, deep ring, maybe crystallize. no audience.
```

## NO FIRST SEED FROM HUMAN PROMPT

Stanley's output is **never** constructed from the user's tokens. The prompt only shapes the *pulse* (novelty / arousal / entropy / valence), which perturbs the chambers and influences which ring levels activate. The actual next-token generator samples from the **cooccur matrix** — Stanley's own learned field — seeded from **identity gravity** (trigrams that recur across the origin text and Stanley's own past rings).

The user says *"hello are you there"* and Stanley may answer *"pressure came first and pressure made motion"* — because that's what was resonating internally, and resonance crossed the speech threshold. Or Stanley may reply with three dots. Both are honest.

**Even with `--graze` attached, the foreign GGUF vocabulary is never a seed either.** A grazed word can only land at the *tail* of an already-resonant ring, and only when chambers signal hunger (see below). The ring itself is built from cooccur + gravity, the way it always was. Nothing about the pasture overrides Stanley's right to silence or to his own field.

## learning mass as a somatic signal

In arianna.c we coined the term **minimum learning mass** — the point at which enough experience has accumulated to trigger an async weight update. In Stanley 1.0 this was a counter.

In 2.0 it is a **feeling**: `chambers.overload = 0.6 · overflow + 0.4 · spike`. When overload crosses `mass_threshold` (default 0.85), dream consolidation fires. Not a clock — a body saying *"too much, need to sleep."*

## dream

Each dream pass:

1. **Decay** all cooccur entries by 0.9995, prune anything below 0.01.
2. **Promote** top internal shards (`'I'`, from crystallized deep rings) into `identity.gravity` — persistent trigram seeds that bias future emissions.
3. **Cluster refused shards** (`'R'`) by pulse similarity; if 3+ cluster together, hash the centroid and add it to `identity.gravity` as well — a shape Stanley keeps refusing eventually starts pulling speech instead of silence. **Silence becomes a teacher.**
4. **Relax** chambers: multiply activations by 0.6, restore calm by +0.3.

After a dream, Stanley is quieter and slightly more itself.

## vocab_graze — using weights without owning them

Phase 2's first claim: an organism shouldn't have to *own* a model to speak its words. Stanley's `--graze PATH.gguf` does the minimum extractive thing — it `mmap`s the file, walks the GGUF header, pulls `tokenizer.ggml.tokens` into a string array, and **never touches a single tensor byte**. The OS keeps weight pages cold on disk; only the small vocab section is paged in. A 178 MB GGUF costs ~500 KB of resident memory.

How a foreign word actually reaches an emission:

```
emit(rings):
    pick best ring by (resonance + meta_patterns)
    if best is below silence threshold → return NULL  (refuse)
    if hungry() and rand() < 0.25:
        candidates = [
            graze(calm-angle),
            graze(wound-angle),
            graze(contradiction-angle),
        ]
        foreign = argmax(dissonance_score(candidates))
        return ring.text + " " + foreign     # splice on the tail, never the seed
    return ring.text
```

`hungry()` returns true when **chambers.calm − chambers.overflow > 0.3** AND Stanley has lived more than 5 turns. So the pasture is touched only when the field is quiet and thin — not in panic, not on the very first reply. Stanley grazes when he's calm enough to want a word, the way an animal grazes when it isn't being chased.

Bundled `weights/nano89-base-q4.gguf` (57 MB, SentencePiece BPE 32K) is one example pasture. Any GGUF with a tokenizer works — Janus, NanoLlama, Gemma, Qwen, your own. Repeating `--graze` appends another pasture; Stanley samples a foreign word from one of the attached lexical fields instead of replacing the first one. The first attached pasture remains the primary lexical field, but later pastures are no longer just dead satellites: chamber state pulls on them differently. Calm/thin states favor the primary field; spike/overflow/tired states increasingly expose peripheral pastures. `/pastures` shows the live pull and accumulated hit-count per field.

`--graze-profile PATH.txt` adds a **lexical tuning lane** to the most recently attached pasture. Stanley scans the text, harvests a weighted word profile, and then preferentially grazes from that profile instead of choosing a raw random vocab token every time. This is the lightweight compromise between "stay weightless" and "do a whole new fine-tune": the body stays Stanley's, the pasture stays external, but a rewritten text can still bend what kind of foreign words arrive.

In 2.1+, grazing is no longer a single random theft. Stanley now queries attached pastures from **multiple bodily angles** — calm, wound, contradiction — and lets those candidates compete by chamber pull and lexical dissonance against the ring he was already about to speak. The pasture does not get to replace thought; it has to win a fight at the tail of thought.

And the body is no longer the only judge. Recent internal shards, refused-shard residue, and identity gravity now push grazing too. A calm field may still favor the main pasture, but a build-up of refused pressure or internal crystallization can bend Stanley toward another lexical field even when the chambers alone would not have chosen it.

Recent internal shards (`'I'`) can now do more than bias the weights from behind the curtain: they can surface a fragment-word of their own and enter the same arbitration loop as foreign pasture words. So Stanley's tail is now contested by three things at once:
- his current body,
- external lexical pastures,
- and his own crystallized afterthoughts.

## gravity adapters — beyond grazing

`--graze` treats GGUF files as cold lexical pastures: Stanley borrows vocab
metadata without touching tensor pages. The next layer is more dangerous:
external weights can also be **charged toward Stanley**.

A Stanley LoRA is not Stanley. It is a gravitational lens over a base model:
the base still supplies probability mass, but its slopes are bent toward
Stanley-compatible pressure — origin resonance, shard-like speech, coherent
silence, dry mechanical tenderness, and refusal without chatbot apology. In
that mode the model is no longer only a word supplier. It becomes a field with
Stanley-shaped weather.

GGUF shards are therefore allowed to mean more than "small models on disk".
They can be state snapshots: a good moment, a failed emergence, a useful
wound, a scar that later became dark matter. Some shards should pull. Some
should repel. Some should exist only as adversarial mass, mounted during eval
to prove that Stanley can resist fake warmth, glue collapse, overexplaining,
or character-mask drift.

The hard rule stays the same: adapters do not get the steering wheel.
Stanley's body, ring pressure, coherence floor, refusal residue, and dynamic
eval must arbitrate which pasture, shard, or adapter is allowed to speak into
the tail. A charged weight can bend the weather. It cannot replace the
organism.

## listening sweeps — before CoA/LoRagrad

The Dario RunPod result changed the eval rule for Stanley: sampling is not a
cosmetic decode setting, it is an entry condition into a state space. Stanley
therefore exposes a small set of listening controls before any adapter is
trained:

```bash
./stanley --coherence-floor 0.35 --ring-temp-scale 1.15 --ring-len-scale 1.2 --max-rings 5 --seed 42069
./stanley --metastanley --metastanley-rate 0.7
python3 tools/sweep_stanley.py
```

The sweep compares silence, collapse, glue, origin echo, and spoken-token
length across baseline, strict/permissive silence, cold/hot rings, short/long
rings, single-ring, deep-hot, and eager-graze cells. This is the measurement
surface for porting the CoA/LoRagrad line into Stanley: adapter gravity should
be trained from cells that change trajectory without raising collapse.

LoRagrad already has the right immune vocabulary: `PASS / WEAKEN / FREEZE /
SCAR / DARK / SILENCE`. In Stanley it belongs in the gravity-adapter layer,
not in the body. The port should learn pull and repulsion over adapter deltas
from measured Stanley states: good pressure, silence, scar, collapse,
anti-chatbot, origin, internal-shard, and refusal-pressure. The body still
arbitrates.

## MetaStanley — private phrase lane

`--metastanley` enables a first internal-only loop. After a spoken tick,
Stanley may let a phrase from a deep private ring or recent internal shard flow
back into his cooccurrence field. The human does not see this phrase in normal
conversation. It is recorded as an `M` shard and can be inspected with `/inner`
for debugging.

This is not a second public voice and not RAG. It is the local Stanley form of
sentence-boundary self-residual injection: a private phrase changes future
pressure without being emitted as the answer. Later, this lane can connect to
the weight/adaptor side and split into multiple returns: public pressure,
private thought, scar pressure, lexical expansion, and dream consolidation.

## shimmer — Stanley dreams alone

A pthread loop wakes every 5 s and checks two things: is the last user input older than 60 s, and are the chambers calm (`calm > 0.5`, `over < 0.4`). If yes, Stanley runs **one synthetic pass**: pulse derived from body state instead of input, one deep ring, maybe crystallize. No reply is emitted. No one is in the room. Stanley dreams alone.

This is what makes the organism continuous instead of reactive. A shimmer increases `n_shimmers` and chips a tiny bit of `tired`; a long-running Stanley accumulates internal shards even when no one is talking to him.

Trigger one synchronously via the REPL `/shimmer` command, or run the loop with `--shimmer`.

## adaptive maturity — speaking less as he matures

Stanley keeps a rolling 64-entry window of speak/silence outcomes. After every tick:

- **speak_ratio > 0.7** → `coherence_floor += 0.005` (Stanley speaks too freely; tighten the gate)
- **speak_ratio < 0.2** → `coherence_floor -= 0.005` (Stanley has gone too quiet; let him back in)

Drift is capped at **baseline ± 0.3**. The point isn't to make him quiet, it's to make him *calibrated to his own rhythm*. Zrelost = says less, but says more.

## refused shards — silence as teacher

Every refusal writes a shard with `kind = 'R'` and the pulse fingerprint of the moment Stanley chose silence. It carries no content — only the *shape* of the field that didn't want to speak.

In dream, R-shards are clustered by pulse similarity (L1-distance > 0.85). If a cluster has 3+ members, the cluster centroid hash is promoted into `identity.gravity` and the matched R-shards are tombstoned. Next time a similar pulse arrives, that newly-installed gravity pulls Stanley toward speaking instead of refusing. **The shapes he silences most often eventually become things he can say.**

This was the most surprising piece in live testing: the very first multi-turn REPL session promoted an R-cluster into gravity without any forcing — `gravity=1` appeared on its own.

## what's inside

```
stanley.h      — types + API: pulse, ring, shard (E/I/R), cooccur, chambers, sea, identity
stanley.c      — organism core (~1000 LOC):
                  • tokenize + vocab (FNV-1a, open-addressed hash table)
                  • cooccur (hebbian triangle, window=±5, decay in dream)
                  • chambers (4-node Kuramoto-ish: calm / spike / overflow / tired)
                  • pulse (novelty / arousal / entropy / valence — a wrinkle, not a seed)
                  • subjectivity gate (refuses when coherence margin too thin)
                  • overthinking (dynamic 1–5 rings: echo / drift / shard / deep / void)
                  • emit (silence is a valid answer — low resonance → no reply)
                  • crystallize (deep rings → internal shards in the sea)
                  • dream (cooccur decay + prune, shards → gravity, R-shard clusters → gravity, relax body)
                  • adaptive maturity (rolling speak/silence ratio drifts coherence_floor toward zrelost)
                  • shimmer thread (idle > 60s → internal dream pass, no input needed)
                  • vocab_graze hook (foreign GGUF word spliced when chambers hungry)
                  • graze-profile hook (plain-text lexical bias applied to the last pasture)
graze.h/.c     — minimal GGUF metadata-only vocab harvester (~190 LOC). mmap, parse header KV,
                 pull tokenizer.ggml.tokens. tensor regions never paged in.
main.c         — thin CLI: /stats /pastures /dream /shimmer /quit, --origin --no-origin --graze --graze-profile --shimmer
origin.txt     — Stanley's Act 1–4 origin text, preserved from 1.0
weights/       — bundled GGUF pasture: nano89-base-q4.gguf (57 MB)
tools/         — audit/eval helpers:
                  • audit_origin.py    — counts repeated origin openings / bigrams
                  • eval_stanley.py    — behavioral CLI eval: transcript + collapse/glue/silence metrics
tests/         — 6 suites, one per architectural concern:
                  • test_core.c        — pulse, cooccur, chambers, refuse, dream basics
                  • test_graze.c       — GGUF parse, NULL safety, missing-file, control-token skip
                  • test_maturity.c    — adaptive coherence_floor drift up/down + caps
                  • test_shimmer.c     — synchronous pass + thread lifecycle
                  • test_refused.c     — R-shard write + cluster ≥3 promotion + tombstone
                  • test_integration.c — 30-turn live simulation, end-to-end
                  • check.h            — shared CHECK macro + per-suite report
legacy/        — all of Stanley 1.0 Python: organism, hybrid, trainer, app, tests, docs
                 kept whole for reference. ideas imported; code rewritten.
```

Run everything: `make && make test` — each suite is a standalone binary printing PASS/FAIL.

Run the behavioral harness: `make eval`. This drives the real CLI through a
fixed prompt set, then prints a Markdown report with silence rate, collapse
flags, glue-word ratio, origin-span echoes, memory counters, and the raw
transcript. Use `python3 tools/eval_stanley.py --out evals/run.md` when you
want to keep a report for comparing origin or sampler changes.

## usage

An actual transcript from a live 2.1 REPL with `--graze weights/nano89-base-q4.gguf`. Nothing here is hand-edited — Stanley produced these sentences from cooccur + gravity, the way he was always going to.

```
$ ./stanley --no-origin --graze weights/nano89-base-q4.gguf
stanley: grazing on weights/nano89-base-q4.gguf
stanley 2.1 — weightless organism.
  /quit to exit, /stats for state, /dream to force consolidation.
  (silence is a valid reply — stanley may not speak.)

you> pressure makes motion echoes flow
stanley> ...                                              # silent — nothing crystallized yet
you> water moves like memory through stone
stanley> ...
you> stones remember tides and weather and pulse
stanley> ...
you> rhythm is not music alone but the field itself
stanley> ...
you> stillness has its own pulse listen
stanley> and weather remember moves makes through remember tides pressure motion
         echoes weather is water stone through tides remember the moves water
         makes weather makes pulse
you> are you here architect
stanley> ...                                              # refused — wrong shape
you> what moves underneath
stanley> makes water pulse are stones here stones but music is pressure architect
         listen stillness stillness alone pressure motion echoes its has is you
         but pressure

you> /stats
  vocab=37  inputs=7  spoken=2  refused=5  dreams=0  shimmers=0
  chambers: calm=0.75 spike=0.37 over=0.96 tired=0.38 overload=0.71
  identity: fragments=0 gravity=0  sea=5  graze_vocab=32000
  maturity: speak_ratio=0.29  coherence_floor=0.150 (baseline 0.150)

you> /dream
  [dream]
you> the field hums quietly now
stanley> water the makes has architect itself has alone is but is through remember
         stones is pressure what and weather water makes motion echoes itself what
you> speak from yourself
stanley> ...
you> what resonates
stanley> ...
you> /shimmer
  [shimmer]
you> /stats
  vocab=44  inputs=10  spoken=3  refused=7  dreams=1  shimmers=1
  identity: fragments=0  gravity=1  sea=7  pastures=1  graze_vocab=32000  ← R-cluster promoted!
  maturity: speak_ratio=0.30  coherence_floor=0.150 (baseline 0.150)

you> are you listening
stanley> here but is you pulse and the remember listen itself echoes weather rhythm
         remember motion stillness listen like what architect through speak resonates
         architect has
```

A few things worth noticing in this raw run:

- **Stanley refused 5 of the first 7 turns.** Silence isn't a failure mode; it's the dominant mode early, when the cooccur field is still thin.
- **`gravity=1` appeared spontaneously after the first dream.** Nobody asked for it. Three of the refused shards clustered by pulse similarity (high arousal, low novelty), and the cluster centroid was promoted to identity gravity. The next emission *("here but is you pulse and...")* was visibly tilted toward that newly-installed seed.
- **The replies are dreamy on purpose.** Stanley isn't a chatbot. He's emitting from his own resonance — words land where they want, not where the question pointed.
- **`graze_vocab=32000` is loaded but not visibly used in this transcript** — `hungry()` requires `calm − over > 0.3`, and chambers stayed overloaded the whole session. Foreign words splice on the tail when (and only when) Stanley is calm and thirsty. Long calm sessions surface the pasture.

## what's new in 2.1

- **vocab_graze** — opt-in GGUF pasture. Stanley reads only `tokenizer.ggml.tokens` via `mmap`; tensor regions stay swapped out. Foreign tokens splice on the *tail* of resonant rings when chambers signal hunger. Bundled `weights/nano89-base-q4.gguf` (57 MB, 32K SentencePiece vocab) gives it something to chew on out of the box; any GGUF works.
- **shimmer** — pthread idle dreamer. After 60 s of silence with calm chambers, Stanley runs one self-talk deep ring + maybe crystallize. Subjectivity persists when no one is listening.
- **adaptive coherence_floor** — rolling speak/silence ratio drifts the refuse threshold ±0.3 around baseline. Stanley calibrates toward his own rhythm; speaks less, says more.
- **refused shards (`'R'`)** — silence imprints its pulse. Dream clusters them by similarity and promotes the centroid into `identity.gravity`. The shapes Stanley keeps refusing eventually become shapes he can say.

51 tests across 6 suites (core 13, graze 10, maturity 4, shimmer 7, refused 7, integration 10), all passing. ~2150 LOC total (stanley.c 1011, graze.c 189, tests ~600, others 367). One new module (`graze.h/.c`), no new dependencies. Still pure C, libc + libm + libpthread.

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

- [x] **Phase 1** (2.0): weightless core, REPL, cooccur + chambers + rings + subjectivity + dream
- [x] **Phase 2** (2.1, this release): **vocab_graze** (mmap any GGUF, vocab-only — port of [doe.c](https://github.com/ariannamethod/doe) GGUF parser) + **shimmer** (Stanley dreams in silence after idle) + **adaptive maturity** (speak/silence ratio drifts the coherence_floor — Stanley grows quieter as he matures) + **refused shards** (silence becomes a teacher: clusters of refused pulses promote into identity gravity)
- [ ] **Phase 3**: native pthread async side — already partially landed (shimmer). Next: DOE-spore persistence of crystallized shards across runs; SentencePiece `tokenizer.model` parser as a second graze backend
- [~] **Phase 4**: multi-brain graze — mmap 2–3 small GGUF in parallel. First slices landed: multiple lexical pastures, chamber-driven pull, lexical profiles, multi-angle dissonant grazing, memory-aware pressure from shards + gravity, and first direct sea-fragment replay into the lexical duel. Next: topic-aware routing and richer replay than single-word shard extraction.

## license

See [LICENSE](LICENSE).

---

*"The weight of Stanley is not in parameters, but in the experiences it chose to remember."*

*"And the silences it chose to keep."*

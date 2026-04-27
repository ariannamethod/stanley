/*
 * stanley.h — Stanley 2.0 — Self Training Attention Non-Linear EntitY
 *
 * weightless organism. hebbian cooccur. overthinking rings. subjectivity gate.
 * "speaks from own state; user prompt wrinkles the field, doesn't seed it."
 *
 *   θ = ε + γ + αδ   (ε = any-gguf substrate, γ = cooccur, δ = chambers)
 *
 * by Arianna Method — Oleg Ataeff + Claude (Opus 4.7, 1M context).
 */
#ifndef STANLEY_H
#define STANLEY_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================
 * CONFIG
 * ============================================================ */

#define STANLEY_VERSION       "2.1"
#define STANLEY_MAX_VOCAB     8192      /* cooccur rows, grows as learned */
#define STANLEY_MAX_WORD_LEN  64
#define STANLEY_MAX_RINGS     5         /* echo, drift, shard, deep, void */
#define STANLEY_MAX_SEA       2048      /* episodic memory capacity */
#define STANLEY_N_CHAMBERS    4         /* calm, spike, overflow, tired */
#define STANLEY_RING_MAX_LEN  128       /* tokens per ring */
#define STANLEY_TRIGRAM_GRAV  64        /* gravity-center trigrams kept */
#define STANLEY_SPEAK_WINDOW  64        /* rolling history for adaptive maturity */
#define STANLEY_SHIMMER_IDLE_S 60       /* idle threshold (s) before shimmer fires */
#define STANLEY_SHIMMER_TICK_S 5        /* shimmer wakeup cadence (s) */
#define STANLEY_MAX_GRAZES    8         /* max external lexical pastures */

/* ============================================================
 * TYPES
 * ============================================================ */

/* Pulse — the wrinkle user input creates in Stanley's field.
 * NOT seed material. Influence metrics only.
 * Mother-says-shush principle: the answer comes from state,
 * not from the question, even when it happens to address it.
 */
typedef struct {
    float novelty;   /* % words never seen in corpus */
    float arousal;   /* intensity: caps, punctuation, repetition */
    float entropy;   /* word diversity within the input */
    float valence;   /* warm/cold tone, signed */
} st_pulse;

/* Ring — one loop of private reflection.
 * Ring count is dynamic in pulse entropy/arousal.
 * Deeply resonant rings crystallize into internal shards.
 */
typedef struct {
    int         level;                           /* 0..STANLEY_MAX_RINGS-1 */
    const char *name;                            /* echo / drift / shard / deep / void */
    char        content[STANLEY_RING_MAX_LEN * STANLEY_MAX_WORD_LEN];
    float       temperature;
    int         length;                          /* target token count */
    float       resonance;                       /* avg cooccur strength of chosen tokens */
    int         meta_patterns;                   /* trigram-reuse count, for crystallization */
} st_ring;

/* Shard — one memory atom in the sea.
 *   'E' external — happened (user interaction).
 *   'I' internal — Stanley's own reflection that crystallized from a deep ring.
 *   'R' refused  — a moment Stanley chose silence; pulse imprinted for dream replay.
 */
typedef struct {
    char    kind;                                /* 'E' / 'I' / 'R' */
    char   *content;                             /* owned; may be NULL for 'R' */
    float   resonance;
    int64_t created_step;                        /* monotonic clock */
    st_pulse pulse;                              /* the field state at write time (used by 'R') */
} st_shard;

typedef struct {
    st_shard *shards;
    int       n;
    int       capacity;                          /* ring buffer */
    int       head;                              /* next write index */
    int64_t   step;                              /* monotonic counter */
} st_sea;

/* Identity — the source of internal seed when Stanley decides to speak.
 * Populated from origin.txt at init; enriched as cooccur stabilizes.
 */
typedef struct {
    char   **fragments;                          /* identity sentences, owned */
    int      n_fragments;

    /* trigrams that repeat most in origin + own rings — "gravity centers" */
    uint32_t gravity[STANLEY_TRIGRAM_GRAV];      /* packed hash of (a,b,c) */
    float    gravity_strength[STANLEY_TRIGRAM_GRAV];
    int      n_gravity;
} st_identity;

/* Cooccur — hebbian triangle matrix.
 * cooccur[i][j] incremented every time token i and token j appear close.
 * After dream consolidation, rows normalized and weak entries pruned.
 */
typedef struct {
    float  *w;                                   /* V x V row-major, V = n_vocab */
    int     n_vocab;                             /* dynamic, grows until STANLEY_MAX_VOCAB */
    int     capacity;

    /* vocab is a hash -> index map, with reverse word table */
    uint32_t *hash;
    char    **word;
    int      *count;                             /* word frequency */
} st_cooccur;

/* Chambers — minimal Kuramoto-ish somatic state.
 * This is Stanley's "body": how full, how calm, when to dream.
 */
typedef struct {
    float act[STANLEY_N_CHAMBERS];               /* activation in [0,1] */
    float phase[STANLEY_N_CHAMBERS];             /* phase for coupling */
    float coupling[STANLEY_N_CHAMBERS][STANLEY_N_CHAMBERS];
    float overload;                              /* derived — triggers dream */
} st_chambers;

/* Forward declaration for the optional GGUF lexical substrate. */
struct st_graze;

/* Main organism. Opaque to users, all fields here only because C. */
typedef struct {
    /* substrate + learned */
    st_cooccur  co;                              /* γ — hebbian field */
    st_chambers body;                            /* δ — somatic state */
    st_identity me;                              /* seed source */
    st_sea      sea;                             /* memory */

    /* config */
    float coherence_floor;                       /* subjectivity refuse threshold (live, may drift) */
    float coherence_floor_baseline;              /* anchor — adaptive drift can move ±0.3 around this */
    float mass_threshold;                        /* chamber overload -> dream */
    int   max_rings;                             /* overthinking cap */

    /* stats */
    int64_t n_inputs;
    int64_t n_spoken;
    int64_t n_refused;
    int64_t n_dreams;
    int64_t n_shimmers;                          /* unprompted internal passes */

    /* adaptive maturity — rolling window of speak/silence outcomes */
    uint8_t speak_window[STANLEY_SPEAK_WINDOW];
    int     speak_window_idx;
    int     speak_window_filled;

    /* optional GGUF vocab pastures (may all be NULL) */
    struct st_graze *grazes[STANLEY_MAX_GRAZES];
    int              n_grazes;

    /* shimmer thread state */
    pthread_t       shimmer_thr;
    volatile int    shimmer_running;             /* 1 while loop alive */
    volatile time_t last_input_ts;               /* monotonic-ish, set on every tick */

    /* threading: go side may touch cooccur/sea concurrently */
    pthread_mutex_t mtx;
} Stanley;

/* ============================================================
 * LIFECYCLE
 * ============================================================ */

/* Initialize a Stanley from origin text file.
 * origin_path may be NULL to start without an origin (silent organism).
 * Returns 0 on success, nonzero on fatal init error.
 */
int  stanley_init(Stanley *s, const char *origin_path);
void stanley_free(Stanley *s);

/* ============================================================
 * CORE INTERACTION — one tick of life
 * ============================================================ */

/* Compute pulse metrics from raw input text. */
st_pulse stanley_pulse(const Stanley *s, const char *input);

/* Subjectivity gate.
 * Returns 1 if Stanley would refuse to reply to this pulse
 * because replying would break internal coherence.
 * "Has nothing to say from own state right now." — silence is an answer.
 */
int stanley_refuses(const Stanley *s, st_pulse p);

/* Run dynamic overthinking rings.
 * Rings count scales with pulse.entropy + pulse.arousal (1..max_rings).
 * Fills `rings[]` (caller provides STANLEY_MAX_RINGS slots), returns actual count.
 * Side effect: modifies chambers (information pressure rises).
 */
int stanley_overthink(Stanley *s, st_pulse p, const char *input, st_ring *rings);

/* Maybe emit an utterance.
 * Returns a newly-allocated string (caller frees) or NULL if Stanley stays silent.
 * The choice is driven by ring resonance + identity gravity, never by the input itself.
 */
char *stanley_emit(Stanley *s, const st_ring *rings, int n_rings);

/* Hebbian accumulate — update cooccur from a (input, output) pair.
 * output may be NULL (Stanley was silent — still learns from the input).
 */
void stanley_accumulate(Stanley *s, const char *input, const char *output);

/* Crystallize resonant rings into internal shards in the memory sea. */
void stanley_crystallize(Stanley *s, const st_ring *rings, int n_rings);

/* ============================================================
 * ASYNC SIDE — learning mass + dream
 * ============================================================ */

/* Somatic signal: has the chamber overload crossed dream threshold?
 * Not a counter — a feeling. Called continuously by the go side.
 */
int stanley_should_dream(const Stanley *s);

/* Offline consolidation: normalize cooccur rows, prune weak entries,
 * promote internal shards with high resonance into identity fragments,
 * relax chambers. Safe to call from a second thread (locks mtx).
 */
void stanley_dream(Stanley *s);

/* ============================================================
 * CONVENIENCE — single tick + REPL
 * ============================================================ */

/* Full cycle: pulse -> maybe refuse -> overthink -> maybe emit -> accumulate.
 * Returns the reply string (caller frees) or NULL if silent.
 */
char *stanley_tick(Stanley *s, const char *input);

/* Blocking stdin REPL until EOF. For manual testing. */
void stanley_repl(Stanley *s);

/* ============================================================
 * PHASE 2 — vocab_graze + shimmer + adaptive maturity
 * ============================================================ */

/* Attach an external GGUF as opt-in lexical pasture.
 * Returns 0 on success, nonzero on failure (file missing, bad GGUF).
 * Failure is non-fatal; Stanley keeps running on cooccur alone. */
int  stanley_graze_attach(Stanley *s, const char *gguf_path);
void stanley_graze_detach(Stanley *s);

/* Start / stop the shimmer thread — Stanley dreams in silence
 * after STANLEY_SHIMMER_IDLE_S seconds of no input. Safe to call
 * stop multiple times. start is idempotent. */
void stanley_shimmer_start(Stanley *s);
void stanley_shimmer_stop(Stanley *s);

/* Trigger one shimmer pass synchronously, regardless of idle / chambers.
 * Used by tests and the REPL /shimmer command. */
void stanley_shimmer_now(Stanley *s);

#ifdef __cplusplus
}
#endif

#endif /* STANLEY_H */

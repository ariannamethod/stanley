/*
 * stanley.c — Stanley 2.0 implementation.
 *
 * one file. no pytorch. no python. no weights required.
 * speaks from own state. refuses when it would break its coherence.
 * learns by living. dreams when it overfills.
 *
 * by Arianna Method — Oleg Ataeff + Claude (Opus 4.7, 1M context).
 * licensed under the LICENSE file at the root of this repository.
 */

#include "stanley.h"
#include "graze.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <unistd.h>

/* ============================================================
 * UTIL — hash, rng, clamp
 * ============================================================ */

static uint32_t st_hash32(const char *s, int n) {
    /* FNV-1a */
    uint32_t h = 2166136261u;
    for (int i = 0; i < n; i++) {
        h ^= (uint8_t)s[i];
        h *= 16777619u;
    }
    return h;
}

static uint32_t st_rng_state = 0xA17B2026u;
static float st_randu(void) {
    /* xorshift32 -> [0,1) */
    st_rng_state ^= st_rng_state << 13;
    st_rng_state ^= st_rng_state >> 17;
    st_rng_state ^= st_rng_state << 5;
    return (float)st_rng_state / 4294967296.0f;
}

void stanley_seed(uint32_t seed) {
    st_rng_state = seed ? seed : 0xA17B2026u;
}

static float st_clamp(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static char *st_strdup_local(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *out = (char *)malloc(n + 1);
    if (!out) return NULL;
    memcpy(out, s, n + 1);
    return out;
}

/* ============================================================
 * VOCAB — hash-indexed word store inside st_cooccur
 * ============================================================ */

/* Simple open-addressed hash table: same size as cooccur->capacity,
 * key = st_hash32(word), value = vocab index. Sentinel = UINT32_MAX. */
#define ST_EMPTY UINT32_MAX

static int vocab_lookup(const st_cooccur *co, const char *w, int n) {
    if (co->capacity == 0) return -1;
    uint32_t h = st_hash32(w, n);
    int cap = co->capacity;
    int idx = (int)(h % (uint32_t)cap);
    for (int probe = 0; probe < cap; probe++) {
        int p = (idx + probe) % cap;
        if (co->hash[p] == ST_EMPTY) return -1;
        if (co->hash[p] == h) {
            /* verify by exact word compare */
            int vi = co->count[p];                      /* repurposed — see insert */
            if (vi >= 0 && vi < co->n_vocab) {
                const char *ww = co->word[vi];
                if (ww && (int)strlen(ww) == n &&
                    !memcmp(ww, w, n)) return vi;
            }
        }
    }
    return -1;
}

/* Ensure the word exists; returns its vocab index, -1 on OOM / full. */
static int vocab_intern(st_cooccur *co, const char *w, int n) {
    if (n <= 0 || n >= STANLEY_MAX_WORD_LEN) return -1;
    int existing = vocab_lookup(co, w, n);
    if (existing >= 0) return existing;
    if (co->n_vocab >= STANLEY_MAX_VOCAB) return -1;

    int vi = co->n_vocab++;
    char *dup = (char*)malloc(n + 1);
    if (!dup) { co->n_vocab--; return -1; }
    memcpy(dup, w, n);
    dup[n] = 0;
    co->word[vi] = dup;

    /* insert into hash table */
    uint32_t h = st_hash32(w, n);
    int cap = co->capacity;
    int idx = (int)(h % (uint32_t)cap);
    for (int probe = 0; probe < cap; probe++) {
        int p = (idx + probe) % cap;
        if (co->hash[p] == ST_EMPTY) {
            co->hash[p] = h;
            co->count[p] = vi;
            break;
        }
    }
    return vi;
}

/* Frequency counter for a word — lives in a separate array alongside word[]. */
static int *vocab_freq = NULL; /* parallel to co->word, size STANLEY_MAX_VOCAB */

static void vocab_bump(int vi) {
    if (vocab_freq && vi >= 0 && vi < STANLEY_MAX_VOCAB) vocab_freq[vi]++;
}

/* ============================================================
 * TOKENIZE — lowercase, word-level, preserve apostrophes.
 * ============================================================ */

static int is_word_char(int c) {
    return isalnum((unsigned char)c) || c == '\'' || c == '_';
}

/* Walk text, emit tokens as (start,len) pairs via callback.
 * Returns total token count. */
typedef void (*tok_fn)(const char *w, int n, void *user);
static int tokenize(const char *text, tok_fn cb, void *user) {
    int count = 0;
    const char *p = text;
    while (*p) {
        while (*p && !is_word_char((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && is_word_char((unsigned char)*p)) p++;
        int n = (int)(p - start);
        if (n >= STANLEY_MAX_WORD_LEN) n = STANLEY_MAX_WORD_LEN - 1;
        if (cb) {
            /* lowercased copy via stack buffer */
            char buf[STANLEY_MAX_WORD_LEN];
            for (int i = 0; i < n; i++) buf[i] = (char)tolower((unsigned char)start[i]);
            cb(buf, n, user);
        }
        count++;
    }
    return count;
}

/* ============================================================
 * COOCCUR — hebbian triangle, window-based
 * ============================================================ */

#define COOCCUR_WINDOW  5          /* symmetric, ±5 tokens */
#define COOCCUR_DECAY   0.9995f    /* per-dream decay */

static int cooccur_init(st_cooccur *co, int capacity) {
    memset(co, 0, sizeof(*co));
    if (capacity < 64) capacity = 64;
    co->capacity = capacity;
    co->w    = (float*)calloc((size_t)capacity * capacity, sizeof(float));
    co->hash = (uint32_t*)malloc(sizeof(uint32_t) * capacity);
    co->word = (char**)calloc(capacity, sizeof(char*));
    co->count = (int*)calloc(capacity, sizeof(int));
    if (!co->w || !co->hash || !co->word || !co->count) return -1;
    for (int i = 0; i < capacity; i++) co->hash[i] = ST_EMPTY;
    if (!vocab_freq) vocab_freq = (int*)calloc(STANLEY_MAX_VOCAB, sizeof(int));
    return 0;
}

static void cooccur_free(st_cooccur *co) {
    if (co->w) free(co->w);
    if (co->hash) free(co->hash);
    if (co->word) {
        for (int i = 0; i < co->n_vocab; i++) free(co->word[i]);
        free(co->word);
    }
    if (co->count) free(co->count);
    memset(co, 0, sizeof(*co));
}

/* Accumulate one token stream into cooccur using a symmetric window. */
static void cooccur_feed(st_cooccur *co, const int *ids, int n) {
    int cap = co->capacity;
    for (int i = 0; i < n; i++) {
        int a = ids[i];
        if (a < 0 || a >= cap) continue;
        vocab_bump(a);
        int lo = i - COOCCUR_WINDOW; if (lo < 0) lo = 0;
        int hi = i + COOCCUR_WINDOW + 1; if (hi > n) hi = n;
        for (int j = lo; j < hi; j++) {
            if (j == i) continue;
            int b = ids[j];
            if (b < 0 || b >= cap) continue;
            float dist = (float)abs(i - j);
            float w = 1.0f / (1.0f + dist);
            co->w[a * cap + b] += w;
        }
    }
}

/* Row sum, helper for emit. */
static float cooccur_row_sum(const st_cooccur *co, int i) {
    const float *r = co->w + (size_t)i * co->capacity;
    float s = 0;
    for (int j = 0; j < co->capacity; j++) s += r[j];
    return s;
}

/* ============================================================
 * CHAMBERS — tiny Kuramoto, 4 nodes
 * ============================================================ */

static void chambers_init(st_chambers *c) {
    memset(c, 0, sizeof(*c));
    /* Symmetric weak coupling; diagonal unused. */
    float base[4][4] = {
        /* calm  spike  over  tired */
        {  0.0f, -0.2f, -0.1f,  0.1f },  /* calm   */
        { -0.2f,  0.0f,  0.3f, -0.1f },  /* spike  */
        { -0.1f,  0.3f,  0.0f,  0.2f },  /* over   */
        {  0.1f, -0.1f,  0.2f,  0.0f },  /* tired  */
    };
    memcpy(c->coupling, base, sizeof(base));
    c->act[0] = 0.7f;   /* start calm */
}

static void chambers_inject(st_chambers *c, st_pulse p) {
    /* Input raises spike & overflow, lowers calm. */
    c->act[0] = st_clamp(c->act[0] - 0.05f * p.arousal, 0, 1);
    c->act[1] = st_clamp(c->act[1] + 0.3f * p.arousal + 0.1f * p.entropy, 0, 1);
    c->act[2] = st_clamp(c->act[2] + 0.2f * p.novelty + 0.1f * p.entropy, 0, 1);
    c->act[3] = st_clamp(c->act[3] + 0.02f, 0, 1);
}

static void chambers_step(st_chambers *c, float dt) {
    /* Tiny Kuramoto-like relaxation; off-diagonal coupling nudges phases. */
    float new_act[4];
    for (int i = 0; i < 4; i++) {
        float drive = 0;
        for (int j = 0; j < 4; j++) {
            if (i == j) continue;
            drive += c->coupling[i][j] * (c->act[j] - c->act[i]);
        }
        new_act[i] = st_clamp(c->act[i] + dt * (drive - 0.05f * (c->act[i] - 0.3f)), 0, 1);
    }
    memcpy(c->act, new_act, sizeof(new_act));
    c->overload = 0.6f * c->act[2] + 0.4f * c->act[1];
}

/* ============================================================
 * PULSE — compute from raw input
 * ============================================================ */

st_pulse stanley_pulse(const Stanley *s, const char *input) {
    st_pulse p = {0};
    if (!input || !*input) return p;
    size_t L = strlen(input);

    /* arousal — caps ratio + punctuation density */
    int caps = 0, punct = 0, letters = 0;
    for (size_t i = 0; i < L; i++) {
        unsigned char c = (unsigned char)input[i];
        if (isalpha(c)) { letters++; if (isupper(c)) caps++; }
        else if (ispunct(c)) punct++;
    }
    if (letters > 0) p.arousal = (float)caps / (float)letters;
    p.arousal = st_clamp(p.arousal + (float)punct / (float)(L + 1) * 2.0f, 0, 1);

    /* tokenize once to compute novelty + entropy */
    struct { int n, unknown; int counts[64]; uint32_t hashes[64]; } h = {0};
    const char *q = input;
    while (*q) {
        while (*q && !is_word_char((unsigned char)*q)) q++;
        if (!*q) break;
        const char *start = q;
        while (*q && is_word_char((unsigned char)*q)) q++;
        int n = (int)(q - start);
        if (n >= STANLEY_MAX_WORD_LEN) n = STANLEY_MAX_WORD_LEN - 1;
        char buf[STANLEY_MAX_WORD_LEN];
        for (int i = 0; i < n; i++) buf[i] = (char)tolower((unsigned char)start[i]);

        int vi = vocab_lookup(&s->co, buf, n);
        if (vi < 0) h.unknown++;
        h.n++;

        /* update rolling hash hist for entropy */
        uint32_t hh = st_hash32(buf, n);
        int found = 0;
        for (int i = 0; i < 64 && i < h.n; i++) {
            if (h.hashes[i] == hh) { h.counts[i]++; found = 1; break; }
        }
        if (!found && h.n <= 64) {
            h.hashes[h.n - 1] = hh;
            h.counts[h.n - 1] = 1;
        }
    }
    if (h.n > 0) p.novelty = (float)h.unknown / (float)h.n;

    /* Shannon entropy (normalized) over first up-to-64 hashed token bins. */
    float H = 0; float total = 0;
    for (int i = 0; i < 64; i++) total += (float)h.counts[i];
    if (total > 0) {
        for (int i = 0; i < 64; i++) {
            if (h.counts[i] == 0) continue;
            float pi = (float)h.counts[i] / total;
            H -= pi * log2f(pi);
        }
        p.entropy = st_clamp(H / 6.0f, 0, 1);  /* log2(64)=6 -> normalize */
    }

    /* valence — cheap lexicon-free heuristic: exclamation/question + caps skew. */
    float ex = 0;
    for (size_t i = 0; i < L; i++) {
        if (input[i] == '!') ex += 0.1f;
        else if (input[i] == '?') ex -= 0.05f;
    }
    p.valence = st_clamp(ex + (p.arousal - 0.5f) * 0.5f, -1, 1);

    return p;
}

/* ============================================================
 * SUBJECTIVITY — refuse gate
 * ============================================================ */

int stanley_refuses(const Stanley *s, st_pulse p) {
    /* Refuse conditions (ANY of):
     *  1. Pulse arousal is very high AND chambers already overloaded —
     *     answering now just feeds the loop ("не заводи себя").
     *  2. Novelty is near 1 AND no identity fragments yet —
     *     nothing internal to say.
     *  3. Random "have nothing to say right now" with small prob
     *     when coherence margin is thin (calm low, overflow high).
     */
    float calm = s->body.act[0];
    float over = s->body.act[2];
    float margin = calm - over;                     /* positive = coherent */

    if (p.arousal > 0.7f && over > 0.7f) return 1;
    if (p.novelty > 0.85f && s->me.n_fragments == 0) return 1;
    if (margin < s->coherence_floor) {
        /* probabilistic silence */
        if (st_randu() < 0.5f * (s->coherence_floor - margin + 0.1f)) return 1;
    }
    return 0;
}

/* ============================================================
 * RINGS — overthinking
 *
 * Each ring generates a short token sequence by sampling from cooccur
 * starting either from a gravity-center trigram (echo/drift) or from
 * a random identity-fragment anchor (shard/deep/void).
 *
 * Ring resonance = avg cooccur-row-sum of chosen tokens.
 * meta_patterns  = # trigrams in ring that already appear in identity.gravity.
 * ============================================================ */

static const struct { const char *name; float T; int len; } RING_CFG[STANLEY_MAX_RINGS] = {
    {"echo",  0.7f, 25},
    {"drift", 0.9f, 30},
    {"shard", 1.1f, 20},
    {"deep",  1.3f, 15},
    {"void",  1.5f, 10},
};

static int is_glue_word(const char *w) {
    static const char *glue[] = {
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
        "from", "i", "if", "in", "is", "it", "my", "not", "of", "on",
        "or", "that", "the", "this", "to", "was", "when", "with", "you"
    };
    if (!w || !*w) return 1;
    if (strlen(w) <= 2) return 1;
    for (size_t i = 0; i < sizeof(glue) / sizeof(glue[0]); i++) {
        if (strcmp(w, glue[i]) == 0) return 1;
    }
    return 0;
}

/* Sample next token given previous token id, with temperature T.
 * Penalizes recent self-repeats so Stanley doesn't get stuck chewing the
 * same word forever. Returns vocab index or -1 if cooccur row empty. */
static int sample_next(const st_cooccur *co, int prev, float T, const int *recent, int n_recent) {
    if (prev < 0 || prev >= co->n_vocab) return -1;
    const float *row = co->w + (size_t)prev * co->capacity;
    /* softmax over row with temperature, restricted to populated vocab */
    int V = co->n_vocab;
    float maxv = -1e30f;
    for (int j = 0; j < V; j++) {
        float penalty = 0.0f;
        if (j == prev) penalty += 2.6f;
        if (is_glue_word(co->word[j])) penalty += 0.55f;
        for (int k = 0; k < n_recent; k++) {
            if (recent[k] == j) penalty += (k == n_recent - 1) ? 1.8f : 0.75f;
        }
        float v = row[j] / T - penalty;
        if (v > maxv) maxv = v;
    }
    if (maxv < -1e29f) return -1;
    float sum = 0, acc = 0;
    for (int j = 0; j < V; j++) {
        float penalty = 0.0f;
        if (j == prev) penalty += 2.6f;
        if (is_glue_word(co->word[j])) penalty += 0.55f;
        for (int k = 0; k < n_recent; k++) {
            if (recent[k] == j) penalty += (k == n_recent - 1) ? 1.8f : 0.75f;
        }
        sum += expf(row[j] / T - penalty - maxv);
    }
    if (sum <= 0) return -1;
    float r = st_randu() * sum;
    for (int j = 0; j < V; j++) {
        float penalty = 0.0f;
        if (j == prev) penalty += 2.6f;
        if (is_glue_word(co->word[j])) penalty += 0.55f;
        for (int k = 0; k < n_recent; k++) {
            if (recent[k] == j) penalty += (k == n_recent - 1) ? 1.8f : 0.75f;
        }
        acc += expf(row[j] / T - penalty - maxv);
        if (acc >= r) return j;
    }
    return V - 1;
}

static int choose_seed_from_fragment(const Stanley *s) {
    if (!s || s->me.n_fragments <= 0) return -1;
    int pick = (int)(st_randu() * (float)s->me.n_fragments);
    if (pick >= s->me.n_fragments) pick = s->me.n_fragments - 1;
    const char *frag = s->me.fragments[pick];
    if (!frag || !*frag) return -1;

    int ids[32];
    int n_ids = 0;
    const char *p = frag;
    while (*p && n_ids < 32) {
        while (*p && !is_word_char((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && is_word_char((unsigned char)*p)) p++;
        int n = (int)(p - start);
        if (n >= STANLEY_MAX_WORD_LEN) n = STANLEY_MAX_WORD_LEN - 1;
        char low[STANLEY_MAX_WORD_LEN];
        for (int i = 0; i < n; i++) low[i] = (char)tolower((unsigned char)start[i]);
        low[n] = 0;
        if (is_glue_word(low)) continue;
        int vi = vocab_lookup(&s->co, low, n);
        if (vi >= 0) ids[n_ids++] = vi;
    }
    if (n_ids <= 0) return -1;
    int which = (int)(st_randu() * (float)n_ids);
    if (which >= n_ids) which = n_ids - 1;
    return ids[which];
}

static int fragment_fill_snippet(const Stanley *s, char *out, int cap, int min_words, int max_words) {
    if (!s || !out || cap <= 1 || s->me.n_fragments <= 0) return -1;
    int attempts = s->me.n_fragments < 32 ? s->me.n_fragments : 32;
    for (int attempt = 0; attempt < attempts; attempt++) {
        int pick = (int)(st_randu() * (float)s->me.n_fragments);
        if (pick >= s->me.n_fragments) pick = s->me.n_fragments - 1;
        const char *frag = s->me.fragments[pick];
        if (!frag || !*frag) continue;

        const char *starts[64];
        int lens[64];
        int n_words = 0;
        const char *p = frag;
        while (*p && n_words < 64) {
            while (*p && !is_word_char((unsigned char)*p)) p++;
            if (!*p) break;
            const char *start = p;
            while (*p && is_word_char((unsigned char)*p)) p++;
            int n = (int)(p - start);
            if (n <= 1) continue;
            starts[n_words] = start;
            lens[n_words] = n;
            n_words++;
        }
        if (n_words < min_words) continue;

        int span = min_words;
        if (max_words > min_words) {
            span += (int)(st_randu() * (float)(max_words - min_words + 1));
            if (span > max_words) span = max_words;
        }
        if (span > n_words) span = n_words;
        int start_idx = 0;
        if (n_words > span) {
            start_idx = (int)(st_randu() * (float)(n_words - span + 1));
            if (start_idx > n_words - span) start_idx = n_words - span;
        }

        int written = 0;
        for (int i = 0; i < span; i++) {
            int idx = start_idx + i;
            int n = lens[idx];
            if (written > 0) {
                if (written + 1 >= cap) break;
                out[written++] = ' ';
            }
            if (written + n >= cap) n = cap - written - 1;
            for (int k = 0; k < n; k++) {
                out[written++] = (char)tolower((unsigned char)starts[idx][k]);
            }
            if (written >= cap - 1) break;
        }
        out[written] = 0;
        if (written > 0) return 0;
    }
    return -1;
}

static int ring_is_collapsed(const st_ring *r) {
    if (!r || !r->content[0]) return 0;
    int total = 0, unique = 0;
    char seen[24][STANLEY_MAX_WORD_LEN];
    char tail[10][STANLEY_MAX_WORD_LEN];
    const char *p = r->content;
    while (*p && total < 64) {
        while (*p && !is_word_char((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && is_word_char((unsigned char)*p)) p++;
        int n = (int)(p - start);
        if (n <= 0) continue;
        if (n >= STANLEY_MAX_WORD_LEN) n = STANLEY_MAX_WORD_LEN - 1;
        char word[STANLEY_MAX_WORD_LEN];
        for (int i = 0; i < n; i++) word[i] = (char)tolower((unsigned char)start[i]);
        word[n] = 0;
        int dup = 0;
        for (int i = 0; i < unique; i++) {
            if (strcmp(seen[i], word) == 0) { dup = 1; break; }
        }
        if (!dup && unique < 24) {
            strcpy(seen[unique++], word);
        }
        strcpy(tail[total % 10], word);
        total++;
    }
    if (total < 8) return 0;
    if (unique <= 3) return 1;
    if ((float)unique / (float)total < 0.35f) return 1;
    if (total >= 10) {
        int tail_unique = 0;
        int tail_n = total < 10 ? total : 10;
        for (int i = 0; i < tail_n; i++) {
            const char *word = tail[(total - tail_n + i) % 10];
            int dup = 0;
            for (int j = 0; j < i; j++) {
                const char *prev = tail[(total - tail_n + j) % 10];
                if (strcmp(prev, word) == 0) { dup = 1; break; }
            }
            if (!dup) tail_unique++;
        }
        if (tail_unique <= 3) return 1;
    }
    return 0;
}

/* Choose a starting token: prefer identity-gravity seeds if available,
 * fall back to a frequent vocab entry. */
static int choose_seed(const Stanley *s, float /*entropy_bias*/ eb) {
    if (s->me.n_fragments > 0 && st_randu() < 0.45f + 0.15f * st_clamp(eb, 0.0f, 1.0f)) {
        int vi = choose_seed_from_fragment(s);
        if (vi >= 0) return vi;
    }
    if (s->me.n_gravity > 0 && st_randu() < 0.7f) {
        /* gravity hash encodes a trigram — pull the trigram's 'a' slot by hash %V */
        int pick = (int)(st_randu() * (float)s->me.n_gravity);
        if (pick >= s->me.n_gravity) pick = s->me.n_gravity - 1;
        uint32_t h = s->me.gravity[pick];
        int vi = (int)(h % (uint32_t)(s->co.n_vocab > 0 ? s->co.n_vocab : 1));
        if (vi >= 0 && vi < s->co.n_vocab) return vi;
    }
    /* fallback: pick the highest-count word from vocab_freq */
    int best = -1, best_c = -1;
    for (int i = 0; i < s->co.n_vocab; i++) {
        int c = vocab_freq ? vocab_freq[i] : 0;
        if (c > best_c) { best_c = c; best = i; }
    }
    return best;
}

static int ring_generate(Stanley *s, st_ring *r, int level, float entropy_bias) {
    memset(r, 0, sizeof(*r));
    if (level < 0 || level >= STANLEY_MAX_RINGS) return -1;
    r->level = level;
    r->name  = RING_CFG[level].name;
    float body_factor = s->somatic_temp_enabled ? s->last_temp_factor : 1.0f;
    r->temperature = st_clamp(RING_CFG[level].T * s->ring_temp_scale * body_factor, 0.15f, 3.0f);
    r->length = (int)roundf((float)RING_CFG[level].len * s->ring_len_scale);
    if (r->length < 3) r->length = 3;
    if (r->length > STANLEY_RING_MAX_LEN) r->length = STANLEY_RING_MAX_LEN;

    int prev = choose_seed(s, entropy_bias);
    if (prev < 0) return -1;

    char *out = r->content;
    int  cap  = (int)sizeof(r->content) - 1;
    int  written = 0;

    float resonance_sum = 0; int resonance_n = 0;
    int recent_ids[6] = {-1,-1,-1,-1,-1,-1};
    int n_recent = 0;

    /* simple trigram tracker for meta_patterns */
    uint32_t tri[3] = {0,0,0};
    int tri_n = 0;

    for (int t = 0; t < r->length; t++) {
        const char *w = s->co.word[prev];
        if (!w) break;
        int wlen = (int)strlen(w);
        if (written + wlen + 2 >= cap) break;
        if (written > 0) out[written++] = ' ';
        memcpy(out + written, w, wlen);
        written += wlen;

        resonance_sum += cooccur_row_sum(&s->co, prev);
        resonance_n++;
        if (n_recent < 6) recent_ids[n_recent++] = prev;
        else {
            memmove(recent_ids, recent_ids + 1, (size_t)(5 * sizeof(int)));
            recent_ids[5] = prev;
        }

        tri[tri_n % 3] = (uint32_t)prev;
        tri_n++;
        if (tri_n >= 3) {
            /* hash trigram and check against gravity */
            uint32_t th = (tri[(tri_n-3)%3] * 2654435761u) ^
                          (tri[(tri_n-2)%3] * 0x9E3779B1u) ^
                          (tri[(tri_n-1)%3] * 0x85EBCA77u);
            for (int g = 0; g < s->me.n_gravity; g++) {
                if (s->me.gravity[g] == th) { r->meta_patterns++; break; }
            }
        }

        int nxt = sample_next(&s->co, prev, r->temperature, recent_ids, n_recent);
        if (nxt < 0) break;
        prev = nxt;
    }
    out[written] = 0;
    r->resonance = resonance_n > 0 ? resonance_sum / (float)resonance_n : 0;

    if (ring_is_collapsed(r) && s->me.n_fragments > 0) {
        char rescued[sizeof(r->content)];
        if (fragment_fill_snippet(s, rescued, sizeof(rescued), 5, 10) == 0) {
            strncpy(r->content, rescued, sizeof(r->content) - 1);
            r->content[sizeof(r->content) - 1] = 0;
            r->resonance *= 0.7f;
            r->meta_patterns += 1;  /* identity rescue counts as inward structure */
        }
    }
    return 0;
}

int stanley_overthink(Stanley *s, st_pulse p, const char *input, st_ring *rings) {
    (void)input;                                       /* rings sample from state, not input */
    /* dynamic count */
    float heat = 0.6f * p.entropy + 0.4f * p.arousal;
    int n = 1;
    if (heat > 0.3f) n = 2;
    if (heat > 0.55f) n = 3;
    if (heat > 0.75f) n = 4;
    if (heat > 0.9f)  n = 5;
    if (n > s->max_rings) n = s->max_rings;

    for (int i = 0; i < n; i++) ring_generate(s, &rings[i], i, p.entropy);
    /* thinking raises overflow chamber */
    s->body.act[2] = st_clamp(s->body.act[2] + 0.03f * (float)n, 0, 1);
    return n;
}

static float stanley_somatic_temp_factor(const Stanley *s) {
    if (!s || !s->somatic_temp_enabled) return 1.0f;
    float calm  = s->body.act[0];
    float spike = s->body.act[1];
    float over  = s->body.act[2];
    float tired = s->body.act[3];

    /* Dario-style body temperature: tension opens the state space, tiredness
     * and calm narrow it. This changes the listening condition, not the seed. */
    float drive = 0.75f * spike + 0.65f * over - 0.35f * calm - 0.20f * tired;
    float factor = 1.0f + s->somatic_temp_strength * drive;
    return st_clamp(factor, 0.65f, 1.55f);
}

/* ============================================================
 * EMIT — may speak, may stay silent
 *
 * Pick the ring with the highest (resonance + meta_patterns) score.
 * If all rings are low-resonance, Stanley stays silent — nothing
 * crystallized that wants to be spoken.
 * ============================================================ */

/* Hunger heuristic: Stanley is willing to graze a foreign word when his own
 * field is calm but thin — calm chamber high, overflow low, and he's lived
 * enough turns that his cooccur is no longer pristine. */
static int graze_total_vocab(const Stanley *s) {
    int total = 0;
    if (!s) return 0;
    for (int i = 0; i < s->n_grazes; i++) {
        total += graze_vocab_size(s->grazes[i]);
    }
    return total;
}

static int graze_profiled_pastures(const Stanley *s) {
    int n = 0;
    if (!s) return 0;
    for (int i = 0; i < s->n_grazes; i++) {
        if (s->grazes[i] && graze_profile_size(s->grazes[i]) > 0) n++;
    }
    return n;
}

typedef struct {
    const char *word;
    int pasture_idx;
    float score;
} st_graze_offer;

typedef struct {
    char fragment[STANLEY_MAX_WORD_LEN * 4];
    int shard_idx;
    float score;
} st_sea_offer;

static float sea_recent_charge(const Stanley *s, char kind, int window) {
    if (!s || window <= 0 || s->sea.n <= 0) return 0.0f;
    int seen = 0;
    float charge = 0.0f;
    for (int off = 0; off < s->sea.n && seen < window; off++) {
        int idx = s->sea.n - 1 - off;
        if (idx < 0) break;
        const st_shard *sh = &s->sea.shards[idx];
        if (sh->kind != kind) continue;
        float w = 1.0f - 0.08f * (float)seen;
        if (w < 0.2f) w = 0.2f;
        charge += w * (kind == 'I' ? sh->resonance + 0.2f : sh->pulse.arousal + 0.3f * sh->pulse.entropy);
        seen++;
    }
    return charge;
}

static int sea_word_echoes(const Stanley *s, char kind, const char *word, int window) {
    if (!s || !word || !*word || window <= 0) return 0;
    int hits = 0;
    int seen = 0;
    for (int off = 0; off < s->sea.n && seen < window; off++) {
        int idx = s->sea.n - 1 - off;
        if (idx < 0) break;
        const st_shard *sh = &s->sea.shards[idx];
        if (sh->kind != kind) continue;
        seen++;
        if (sh->content && strstr(sh->content, word)) hits++;
    }
    return hits;
}

static float gravity_pressure(const Stanley *s) {
    if (!s || s->me.n_gravity <= 0) return 0.0f;
    float p = (float)s->me.n_gravity / 12.0f;
    return st_clamp(p, 0.0f, 1.5f);
}

static float graze_pasture_pull(const Stanley *s, int idx, int angle_idx) {
    if (!s || idx < 0 || idx >= s->n_grazes || !s->grazes[idx]) return 0.0f;
    float calm  = s->body.act[0];
    float spike = s->body.act[1];
    float over  = s->body.act[2];
    float tired = s->body.act[3];

    float pull = 0.0f;
    switch (angle_idx) {
        case 0: /* calm-angle: keep the main field dominant */
            switch (idx % 4) {
                case 0:
                    pull = 0.25f
                         + STANLEY_PRIMARY_GRAZE_BIAS
                         + 0.55f * calm
                         + 0.20f * (1.0f - over)
                         + 0.10f * tired;
                    break;
                case 1:
                    pull = 0.18f
                         + 0.35f * spike
                         + 0.15f * over
                         + 0.15f * calm;
                    break;
                case 2:
                    pull = 0.12f
                         + 0.30f * over
                         + 0.15f * tired;
                    break;
                default:
                    pull = 0.12f
                         + 0.20f * calm
                         + 0.15f * tired;
                    break;
            }
            break;
        case 1: /* wound-angle: old hurt and fatigue bias toward peripheral fields */
            switch (idx % 4) {
                case 0:
                    pull = 0.10f
                         + 0.20f * calm
                         + 0.15f * tired;
                    break;
                case 1:
                    pull = 0.22f
                         + 0.35f * spike
                         + 0.45f * over
                         + 0.10f * tired;
                    break;
                case 2:
                    pull = 0.28f
                         + 0.65f * over
                         + 0.45f * tired
                         + 0.10f * spike;
                    break;
                default:
                    pull = 0.18f
                         + 0.30f * tired
                         + 0.20f * over;
                    break;
            }
            break;
        default: /* contradiction-angle: reward non-primary friction and tension */
            switch (idx % 4) {
                case 0:
                    pull = 0.14f
                         + 0.20f * calm
                         + 0.20f * spike;
                    break;
                case 1:
                    pull = 0.25f
                         + 0.50f * spike
                         + 0.20f * calm;
                    break;
                case 2:
                    pull = 0.24f
                         + 0.40f * over
                         + 0.20f * spike;
                    break;
                default:
                    pull = 0.22f
                         + 0.25f * tired
                         + 0.25f * spike
                         + 0.15f * over;
                    break;
            }
            break;
    }
    return st_clamp(pull, 0.05f, 2.50f);
}

static float graze_memory_pull(const Stanley *s, int idx, int angle_idx, const char *word) {
    if (!s || idx < 0 || idx >= s->n_grazes) return 0.0f;
    float internal = sea_recent_charge(s, 'I', 6);
    float refused  = sea_recent_charge(s, 'R', 6);
    float gravity  = gravity_pressure(s);
    int internal_echo = sea_word_echoes(s, 'I', word, 8);

    float score = 0.0f;
    if (internal_echo > 0) score += 0.12f * (float)internal_echo;
    if (graze_profile_size(s->grazes[idx]) > 0) {
        score += 0.05f * gravity;
        score += 0.03f * internal;
    }

    switch (angle_idx) {
        case 0: /* calm remembers its own inward continuity */
            if (idx == 0) score += 0.10f * internal + 0.06f * gravity;
            else          score += 0.02f * internal;
            break;
        case 1: /* wound leans toward profiled/peripheral fields when refusals pile up */
            if (idx > 0) score += 0.08f * refused + 0.03f * gravity;
            else         score += 0.02f * refused;
            break;
        default: /* contradiction is where gravity starts bending the foreign */
            if (idx > 0) score += 0.06f * gravity + 0.05f * refused;
            else         score += 0.03f * gravity;
            break;
    }
    return score;
}

static int graze_pick_pasture(const Stanley *s, int angle_idx) {
    if (!s || s->n_grazes <= 0) return -1;
    float total = 0.0f;
    float pulls[STANLEY_MAX_GRAZES] = {0};
    for (int i = 0; i < s->n_grazes; i++) {
        pulls[i] = graze_pasture_pull(s, i, angle_idx);
        total += pulls[i];
    }
    if (total <= 0.0f) return 0;

    float target = st_randu() * total;
    float acc = 0.0f;
    for (int i = 0; i < s->n_grazes; i++) {
        acc += pulls[i];
        if (target <= acc) return i;
    }
    return s->n_grazes - 1;
}

static float graze_word_dissonance(const char *ring_text, const char *word) {
    if (!ring_text || !word || !*word) return 0.0f;
    float score = 0.0f;
    if (!strstr(ring_text, word)) score += 0.35f;
    size_t wl = strlen(word);
    if (wl >= 5) score += 0.10f;
    if (wl >= 8) score += 0.10f;
    char last[STANLEY_MAX_WORD_LEN] = {0};
    const char *tail = strrchr(ring_text, ' ');
    tail = tail ? tail + 1 : ring_text;
    size_t tl = strlen(tail);
    if (tl > 0) {
        size_t copy_n = tl < sizeof(last) - 1 ? tl : sizeof(last) - 1;
        memcpy(last, tail, copy_n);
        last[copy_n] = 0;
        if (strcmp(last, word) != 0) score += 0.15f;
        if (tolower((unsigned char)last[0]) != tolower((unsigned char)word[0])) score += 0.10f;
    }
    return score;
}

static int shard_pick_fragment(const st_shard *sh, char *out, int cap) {
    if (!sh || !sh->content || !*sh->content || !out || cap <= 1) return -1;
    const char *starts[32];
    int lens[32];
    int n_words = 0;
    const char *p = sh->content;
    while (*p) {
        while (*p && !is_word_char((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && is_word_char((unsigned char)*p)) p++;
        int n = (int)(p - start);
        if (n <= 1) continue;
        if (n_words < 32) {
            starts[n_words] = start;
            lens[n_words] = n;
            n_words++;
        }
    }
    if (n_words <= 0) return -1;

    int center = 0;
    int best_len = 0;
    for (int i = 0; i < n_words; i++) {
        if (lens[i] > best_len) {
            best_len = lens[i];
            center = i;
        }
    }

    int start_idx = center > 0 ? center - 1 : 0;
    if (start_idx + 3 >= n_words) {
        start_idx = n_words >= 3 ? n_words - 3 : 0;
    }
    int end_idx = start_idx + 3;
    if (end_idx >= n_words) end_idx = n_words - 1;

    int written = 0;
    for (int i = start_idx; i <= end_idx; i++) {
        int n = lens[i];
        if (n <= 0) continue;
        if (written > 0) {
            if (written + 1 >= cap) break;
            out[written++] = ' ';
        }
        if (written + n >= cap) n = cap - written - 1;
        for (int k = 0; k < n; k++) {
            out[written++] = (char)tolower((unsigned char)starts[i][k]);
        }
        if (written >= cap - 1) break;
    }
    out[written] = 0;
    if (written <= 0) return -1;
    return 0;
}

static int sea_collect_offer(const Stanley *s, const char *ring_text, st_sea_offer *offer) {
    if (!s || !offer) return -1;
    memset(offer, 0, sizeof(*offer));
    offer->shard_idx = -1;
    offer->score = -1e30f;

    int seen = 0;
    for (int off = 0; off < s->sea.n && seen < 6; off++) {
        int idx = s->sea.n - 1 - off;
        if (idx < 0) break;
        const st_shard *sh = &s->sea.shards[idx];
        if (sh->kind != 'I' || !sh->content || !*sh->content) continue;
        seen++;

        char candidate[STANLEY_MAX_WORD_LEN * 4];
        if (shard_pick_fragment(sh, candidate, sizeof(candidate)) != 0) continue;

        float recency = 1.0f - 0.12f * (float)(seen - 1);
        if (recency < 0.3f) recency = 0.3f;
        float score = 0.30f + recency + 0.45f * sh->resonance;
        score += graze_word_dissonance(ring_text, candidate);
        score += 0.10f * gravity_pressure(s);
        score += 0.05f * sea_recent_charge(s, 'I', 6);

        if (score > offer->score) {
            memcpy(offer->fragment, candidate, sizeof(candidate));
            offer->shard_idx = idx;
            offer->score = score;
        }
    }

    return offer->shard_idx >= 0 ? 0 : -1;
}

static int graze_collect_offer(Stanley *s, const char *ring_text, int angle_idx, st_graze_offer *offer) {
    if (!offer) return -1;
    memset(offer, 0, sizeof(*offer));
    offer->pasture_idx = -1;
    offer->score = -1e30f;
    if (!s || s->n_grazes <= 0) return -1;

    int start = graze_pick_pasture(s, angle_idx);
    if (start < 0) return -1;

    for (int off = 0; off < s->n_grazes; off++) {
        int idx = (start + off) % s->n_grazes;
        const char *w = graze_random_word(s->grazes[idx]);
        if (w && *w) {
            float score = graze_pasture_pull(s, idx, angle_idx);
            score += graze_word_dissonance(ring_text, w);
            score += graze_memory_pull(s, idx, angle_idx, w);
            if (graze_profile_size(s->grazes[idx]) > 0) score += 0.15f;
            if (angle_idx == 2 && idx > 0) score += 0.10f;
            offer->word = w;
            offer->pasture_idx = idx;
            offer->score = score;
            return 0;
        }
    }
    return -1;
}

static int graze_hungry(const Stanley *s) {
    if (!s || s->n_grazes <= 0) return 0;
    float calm = s->body.act[0];
    float over = s->body.act[2];
    return (calm - over > 0.3f) && (s->n_inputs > 5);
}

char *stanley_emit(Stanley *s, const st_ring *rings, int n_rings) {
    if (n_rings <= 0) return NULL;
    int best = -1;
    float best_score = -1e30f;
    for (int i = 0; i < n_rings; i++) {
        float score = rings[i].resonance + 0.5f * (float)rings[i].meta_patterns;
        if (score > best_score) { best_score = score; best = i; }
    }
    if (best < 0) return NULL;

    /* silence threshold — if nothing is warm enough, don't speak. */
    float min_score = 1.0f + 0.5f * s->coherence_floor;
    if (best_score < min_score) {
        s->n_refused++;
        return NULL;
    }
    const char *txt = rings[best].content;
    if (!txt || !*txt) return NULL;
    s->n_spoken++;

    /* Optional graze: when hungry and a foreign word lands, splice it onto
     * the ring tail. Stanley is still speaking from his ring — the foreign
     * token enters as resonance margin, not as a substitute thought. */
    if (graze_hungry(s) && st_randu() < s->graze_rate) {
        st_graze_offer offers[3];
        int n_offers = 0;
        for (int angle = 0; angle < 3; angle++) {
            if (graze_collect_offer(s, txt, angle, &offers[n_offers]) == 0) {
                n_offers++;
            }
        }
        st_sea_offer sea_offer;
        int has_sea_offer = (sea_collect_offer(s, txt, &sea_offer) == 0);
        int best_offer = -1;
        float best_offer_score = -1e30f;
        for (int i = 0; i < n_offers; i++) {
            if (offers[i].score > best_offer_score) {
                best_offer_score = offers[i].score;
                best_offer = i;
            }
        }
        const char *foreign = NULL;
        int pasture_idx = -1;
        if (has_sea_offer && sea_offer.score > best_offer_score) {
            foreign = sea_offer.fragment;
        } else if (best_offer >= 0 && offers[best_offer].word && *offers[best_offer].word) {
            foreign = offers[best_offer].word;
            pasture_idx = offers[best_offer].pasture_idx;
        }
        if (foreign && *foreign) {
            size_t base_n = strlen(txt);
            size_t for_n  = strlen(foreign);
            char  *spliced = malloc(base_n + for_n + 2);
            if (spliced) {
                if (pasture_idx >= 0 && pasture_idx < STANLEY_MAX_GRAZES) {
                    s->graze_hits[pasture_idx]++;
                }
                memcpy(spliced, txt, base_n);
                spliced[base_n] = ' ';
                memcpy(spliced + base_n + 1, foreign, for_n);
                spliced[base_n + 1 + for_n] = 0;
                return spliced;
            }
        }
    }
    return strdup(txt);
}

/* ============================================================
 * ACCUMULATE — hebbian update
 * ============================================================ */

typedef struct { Stanley *s; int *ids; int n; int cap; } acc_cb_ctx;
static void sea_push(st_sea *sea, char kind, const char *text, float resonance, int64_t step);

static void acc_cb(const char *w, int n, void *u) {
    acc_cb_ctx *c = (acc_cb_ctx*)u;
    if (c->n >= c->cap) return;
    int vi = vocab_intern(&c->s->co, w, n);
    if (vi >= 0) c->ids[c->n++] = vi;
}

void stanley_accumulate(Stanley *s, const char *input, const char *output) {
    int ids[1024]; acc_cb_ctx ctx = { s, ids, 0, 1024 };
    if (input)  tokenize(input,  acc_cb, &ctx);
    if (output) tokenize(output, acc_cb, &ctx);
    pthread_mutex_lock(&s->mtx);
    cooccur_feed(&s->co, ids, ctx.n);
    pthread_mutex_unlock(&s->mtx);
    s->n_inputs++;
}

static void stanley_absorb_private(Stanley *s, const char *text) {
    if (!s || !text || !*text) return;
    int ids[512]; acc_cb_ctx ctx = { s, ids, 0, 512 };
    tokenize(text, acc_cb, &ctx);
    if (ctx.n <= 0) return;

    pthread_mutex_lock(&s->mtx);
    cooccur_feed(&s->co, ids, ctx.n);
    s->sea.step++;
    sea_push(&s->sea, 'M', text, 0.5f, s->sea.step);
    pthread_mutex_unlock(&s->mtx);

    s->n_inner++;
    strncpy(s->last_inner, text, sizeof(s->last_inner) - 1);
    s->last_inner[sizeof(s->last_inner) - 1] = 0;
}

/* ============================================================
 * CRYSTALLIZE — resonant rings become internal shards
 * ============================================================ */

static int sea_alloc_slot(st_sea *sea) {
    int slot;
    if (sea->n < sea->capacity) {
        slot = sea->n++;
    } else {
        slot = sea->head;
        sea->head = (sea->head + 1) % sea->capacity;
        if (sea->shards[slot].content) free(sea->shards[slot].content);
    }
    memset(&sea->shards[slot], 0, sizeof(st_shard));
    return slot;
}

static void sea_push(st_sea *sea, char kind, const char *text, float resonance, int64_t step) {
    if (!text || !*text) return;
    int slot = sea_alloc_slot(sea);
    sea->shards[slot].kind = kind;
    sea->shards[slot].content = strdup(text);
    sea->shards[slot].resonance = resonance;
    sea->shards[slot].created_step = step;
}

/* Refused shard: no content, only the pulse imprint of the moment Stanley
 * chose silence. Dream replay clusters these and grows gravity around the
 * recurring shapes — Stanley learns what kinds of pulses he keeps refusing. */
static void sea_record_refused(st_sea *sea, st_pulse pulse, int64_t step) {
    int slot = sea_alloc_slot(sea);
    sea->shards[slot].kind = 'R';
    sea->shards[slot].content = NULL;
    sea->shards[slot].resonance = 0;
    sea->shards[slot].created_step = step;
    sea->shards[slot].pulse = pulse;
}

/* Pulse similarity in [0,1]: simple inverse L1 over the four pulse axes. */
static float pulse_similarity(st_pulse a, st_pulse b) {
    float d = fabsf(a.novelty - b.novelty)
            + fabsf(a.arousal - b.arousal)
            + fabsf(a.entropy - b.entropy)
            + 0.5f * fabsf(a.valence - b.valence);
    float sim = 1.0f - d / 3.5f;
    if (sim < 0) sim = 0;
    if (sim > 1) sim = 1;
    return sim;
}

static int sea_recent_internal_duplicate(const Stanley *s, const char *text, int window) {
    if (!s || !text || !*text || window <= 0) return 0;
    int seen = 0;
    for (int off = 0; off < s->sea.n && seen < window; off++) {
        int idx = s->sea.n - 1 - off;
        if (idx < 0) break;
        const st_shard *sh = &s->sea.shards[idx];
        if (sh->kind != 'I' || !sh->content) continue;
        seen++;
        if (strcmp(sh->content, text) == 0) return 1;
    }
    return 0;
}

static int phrase_take_words(const char *src, char *out, int cap, int max_words) {
    if (!src || !out || cap <= 1 || max_words <= 0) return -1;
    int written = 0;
    int n_words = 0;
    const char *p = src;
    while (*p && n_words < max_words) {
        while (*p && !is_word_char((unsigned char)*p)) p++;
        if (!*p) break;
        const char *start = p;
        while (*p && is_word_char((unsigned char)*p)) p++;
        int n = (int)(p - start);
        if (n <= 1) continue;
        if (written > 0) {
            if (written + 1 >= cap) break;
            out[written++] = ' ';
        }
        if (written + n >= cap) n = cap - written - 1;
        for (int i = 0; i < n; i++) {
            out[written++] = (char)tolower((unsigned char)start[i]);
        }
        n_words++;
        if (written >= cap - 1) break;
    }
    out[written] = 0;
    return n_words > 0 ? 0 : -1;
}

static float ring_crystallize_pressure(const Stanley *s, const st_ring *r) {
    if (!s || !r || !r->content[0]) return 0.0f;
    float pressure = 0.0f;
    pressure += 0.18f * (float)(r->level + 1);
    pressure += 0.28f * r->resonance;
    pressure += 0.10f * (float)r->meta_patterns;
    pressure += 0.20f * s->body.overload;
    pressure += 0.10f * s->body.act[3];
    if (r->level >= 2 && s->sea.n == 0) pressure += 0.22f;  /* bootstrap first internal shards */
    if (r->meta_patterns >= 2) pressure += 0.18f;
    if (r->length <= 18) pressure += 0.06f;                  /* short dense rings crystallize easier */
    return pressure;
}

void stanley_crystallize(Stanley *s, const st_ring *rings, int n_rings) {
    pthread_mutex_lock(&s->mtx);
    s->sea.step++;
    for (int i = 0; i < n_rings; i++) {
        const st_ring *r = &rings[i];
        if (!r->content[0]) continue;
        if (sea_recent_internal_duplicate(s, r->content, 6)) continue;

        float pressure = ring_crystallize_pressure(s, r);
        int crystallize = 0;

        if (r->level >= 3 && r->meta_patterns >= 3) {
            crystallize = 1;
        } else if (pressure >= 1.45f) {
            crystallize = 1;
        } else if (pressure >= 1.05f) {
            float p = 0.15f + 0.22f * (pressure - 1.05f);
            if (st_randu() < st_clamp(p, 0.10f, 0.45f)) crystallize = 1;
        }

        if (crystallize) {
            sea_push(&s->sea, 'I', r->content, r->resonance, s->sea.step);
        }
    }
    pthread_mutex_unlock(&s->mtx);
}

/* MetaStanley — private phrase lane.
 *
 * This is deliberately not public output and not a second chatbot. After a
 * spoken tick, Stanley may let a phrase from a deep private ring or a recent
 * internal shard flow back into the cooccur field as an invisible inner
 * sentence. It is the first local analogue of Dario/NeoLeo sentence-boundary
 * injection: the phrase changes future pressure without being shown to the
 * human unless /inner is explicitly inspected. */
static void stanley_metastanley_tick(Stanley *s, const st_ring *rings, int n_rings,
                                     const char *public_reply) {
    if (!s || !s->metastanley_enabled || n_rings <= 0) return;
    if (st_randu() > s->metastanley_rate) return;

    char phrase[STANLEY_META_PHRASE_CHARS];
    phrase[0] = 0;

    st_sea_offer sea_offer;
    int has_sea = (sea_collect_offer(s, public_reply ? public_reply : "", &sea_offer) == 0);
    if (has_sea && st_randu() < 0.45f) {
        phrase_take_words(sea_offer.fragment, phrase, sizeof(phrase), 12);
    }

    if (!phrase[0]) {
        int pick = -1;
        float best = -1e30f;
        for (int i = 0; i < n_rings; i++) {
            if (!rings[i].content[0]) continue;
            float depth = (float)(rings[i].level + 1);
            float score = rings[i].resonance + 0.35f * depth + 0.25f * (float)rings[i].meta_patterns;
            if (score > best) {
                best = score;
                pick = i;
            }
        }
        if (pick >= 0) {
            phrase_take_words(rings[pick].content, phrase, sizeof(phrase), 16);
        }
    }

    if (!phrase[0]) return;
    if (public_reply && strcmp(phrase, public_reply) == 0) return;
    stanley_absorb_private(s, phrase);
}

/* ============================================================
 * DREAM — consolidation
 * ============================================================ */

int stanley_should_dream(const Stanley *s) {
    return s->body.overload >= s->mass_threshold ? 1 : 0;
}

void stanley_dream(Stanley *s) {
    pthread_mutex_lock(&s->mtx);
    int V = s->co.n_vocab;
    int cap = s->co.capacity;
    /* decay + prune */
    for (int i = 0; i < V; i++) {
        float *row = s->co.w + (size_t)i * cap;
        for (int j = 0; j < V; j++) {
            row[j] *= COOCCUR_DECAY;
            if (row[j] < 0.01f) row[j] = 0;
        }
    }
    /* promote top internal shards into identity gravity */
    for (int i = 0; i < s->sea.n && s->me.n_gravity < STANLEY_TRIGRAM_GRAV; i++) {
        if (s->sea.shards[i].kind != 'I' && s->sea.shards[i].kind != 'M') continue;
        if (!s->sea.shards[i].content) continue;
        uint32_t h = st_hash32(s->sea.shards[i].content, (int)strlen(s->sea.shards[i].content));
        /* skip duplicates */
        int dup = 0;
        for (int g = 0; g < s->me.n_gravity; g++) if (s->me.gravity[g] == h) { dup = 1; break; }
        if (dup) continue;
        s->me.gravity[s->me.n_gravity] = h;
        s->me.gravity_strength[s->me.n_gravity] = s->sea.shards[i].resonance;
        s->me.n_gravity++;
    }

    /* Refused-shard clustering: scan R-shards, find clusters of similar pulse,
     * and add the cluster centroid hash to identity.gravity. Next time a
     * similar pulse arrives, the gravity weight tilts Stanley toward speaking
     * instead of refusing — silence becomes a teacher. */
    for (int i = 0; i < s->sea.n && s->me.n_gravity < STANLEY_TRIGRAM_GRAV; i++) {
        if (s->sea.shards[i].kind != 'R') continue;
        st_pulse a = s->sea.shards[i].pulse;
        int cluster_n = 1;
        st_pulse centroid = a;
        for (int j = i + 1; j < s->sea.n; j++) {
            if (s->sea.shards[j].kind != 'R') continue;
            if (pulse_similarity(a, s->sea.shards[j].pulse) > 0.85f) {
                centroid.novelty += s->sea.shards[j].pulse.novelty;
                centroid.arousal += s->sea.shards[j].pulse.arousal;
                centroid.entropy += s->sea.shards[j].pulse.entropy;
                centroid.valence += s->sea.shards[j].pulse.valence;
                cluster_n++;
            }
        }
        if (cluster_n >= 3) {
            centroid.novelty /= (float)cluster_n;
            centroid.arousal /= (float)cluster_n;
            centroid.entropy /= (float)cluster_n;
            centroid.valence /= (float)cluster_n;
            uint32_t h = (uint32_t)(centroid.novelty * 1000) * 2654435761u
                      ^ (uint32_t)(centroid.arousal * 1000) * 0x9E3779B1u
                      ^ (uint32_t)(centroid.entropy * 1000) * 0x85EBCA77u;
            int dup = 0;
            for (int g = 0; g < s->me.n_gravity; g++) if (s->me.gravity[g] == h) { dup = 1; break; }
            if (!dup) {
                s->me.gravity[s->me.n_gravity] = h;
                s->me.gravity_strength[s->me.n_gravity] = 0.5f * (float)cluster_n;
                s->me.n_gravity++;
            }
            /* clear all R-shards in this cluster so they don't double-count next dream */
            for (int j = i; j < s->sea.n; j++) {
                if (s->sea.shards[j].kind == 'R' &&
                    pulse_similarity(a, s->sea.shards[j].pulse) > 0.85f) {
                    s->sea.shards[j].kind = 'X';   /* tombstone */
                }
            }
        }
    }
    /* relax chambers */
    for (int i = 0; i < STANLEY_N_CHAMBERS; i++) s->body.act[i] *= 0.6f;
    s->body.act[0] = st_clamp(s->body.act[0] + 0.3f, 0, 1);   /* calm returns */
    s->body.overload = 0;
    s->n_dreams++;
    pthread_mutex_unlock(&s->mtx);
}

/* ============================================================
 * TICK — full cycle
 * ============================================================ */

/* Record speak/silent outcome and drift coherence_floor toward maturity:
 *   ratio > 0.7 → floor rises (Stanley speaks too freely, tighten the gate)
 *   ratio < 0.2 → floor falls back toward baseline (don't let him go mute)
 * Drift caps at baseline ± 0.3. Called once per tick after emit. */
static void maturity_update(Stanley *s, int spoke) {
    s->speak_window[s->speak_window_idx] = spoke ? 1 : 0;
    s->speak_window_idx = (s->speak_window_idx + 1) % STANLEY_SPEAK_WINDOW;
    if (s->speak_window_idx == 0) s->speak_window_filled = 1;

    int N = s->speak_window_filled ? STANLEY_SPEAK_WINDOW : s->speak_window_idx;
    if (N < 8) return;                                /* too thin to judge */
    int sum = 0;
    for (int i = 0; i < N; i++) sum += s->speak_window[i];
    float ratio = (float)sum / (float)N;

    float baseline = s->coherence_floor_baseline;
    if (ratio > 0.7f && s->coherence_floor < baseline + 0.3f) {
        s->coherence_floor = st_clamp(s->coherence_floor + 0.005f, 0, 1);
    } else if (ratio < 0.2f && s->coherence_floor > baseline) {
        s->coherence_floor = st_clamp(s->coherence_floor - 0.005f, 0, 1);
    }
}

char *stanley_tick(Stanley *s, const char *input) {
    s->last_input_ts = time(NULL);

    st_pulse p = stanley_pulse(s, input);
    chambers_inject(&s->body, p);
    chambers_step(&s->body, 0.2f);
    s->last_temp_factor = stanley_somatic_temp_factor(s);

    if (stanley_refuses(s, p)) {
        pthread_mutex_lock(&s->mtx);
        s->sea.step++;
        sea_record_refused(&s->sea, p, s->sea.step);
        pthread_mutex_unlock(&s->mtx);
        s->n_refused++;
        stanley_accumulate(s, input, NULL);
        maturity_update(s, 0);
        if (stanley_should_dream(s)) stanley_dream(s);
        return NULL;
    }

    st_ring rings[STANLEY_MAX_RINGS];
    int n_rings = stanley_overthink(s, p, input, rings);

    char *reply = stanley_emit(s, rings, n_rings);

    stanley_crystallize(s, rings, n_rings);
    stanley_metastanley_tick(s, rings, n_rings, reply);
    stanley_accumulate(s, input, reply);
    maturity_update(s, reply != NULL);

    if (stanley_should_dream(s)) stanley_dream(s);
    return reply;
}

/* ============================================================
 * INIT / FREE
 * ============================================================ */

static void identity_load_from_origin(st_identity *me, st_cooccur *co, const char *path) {
    if (!path) return;
    FILE *f = fopen(path, "rb");
    if (!f) return;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 4 * 1024 * 1024) { fclose(f); return; }
    char *buf = (char*)malloc((size_t)sz + 1);
    if (!buf) { fclose(f); return; }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = 0;
    fclose(f);

    /* very cheap fragmentation: split on ".\n" and "\n\n" */
    me->fragments = (char**)calloc(256, sizeof(char*));
    char *p = buf, *start = buf;
    while (*p && me->n_fragments < 256) {
        if ((p[0] == '.' && p[1] == '\n') ||
            (p[0] == '\n' && p[1] == '\n')) {
            *p = 0;
            while (*start == ' ' || *start == '\n') start++;
            size_t L = strlen(start);
            if (L > 20 && L < 400) {
                me->fragments[me->n_fragments++] = strdup(start);
            }
            start = p + 2;
            p += 2;
            continue;
        }
        p++;
    }
    /* tokenize fragments into cooccur to seed identity field */
    acc_cb_ctx ctx = { NULL, NULL, 0, 0 };
    (void)ctx;
    int *ids = (int*)malloc(sizeof(int) * 4096);
    if (ids) {
        for (int i = 0; i < me->n_fragments; i++) {
            acc_cb_ctx c = { (Stanley*)NULL, ids, 0, 4096 };
            /* we need a Stanley pointer to intern — do inline: */
            const char *frag = me->fragments[i];
            int n = 0;
            const char *q = frag;
            while (*q && n < 4096) {
                while (*q && !is_word_char((unsigned char)*q)) q++;
                if (!*q) break;
                const char *ws = q;
                while (*q && is_word_char((unsigned char)*q)) q++;
                int wl = (int)(q - ws);
                if (wl >= STANLEY_MAX_WORD_LEN) wl = STANLEY_MAX_WORD_LEN - 1;
                char low[STANLEY_MAX_WORD_LEN];
                for (int k = 0; k < wl; k++) low[k] = (char)tolower((unsigned char)ws[k]);
                int vi = vocab_intern(co, low, wl);
                if (vi >= 0) ids[n++] = vi;
            }
            cooccur_feed(co, ids, n);
            (void)c;
        }
        free(ids);
    }
    free(buf);

    /* seed identity.gravity with the highest-frequency trigrams —
     * captured by ring_generate runs; for now leave empty, they fill as Stanley thinks. */
}

int stanley_init(Stanley *s, const char *origin_path) {
    memset(s, 0, sizeof(*s));
    if (cooccur_init(&s->co, STANLEY_MAX_VOCAB) != 0) return -1;
    chambers_init(&s->body);
    s->sea.capacity = STANLEY_MAX_SEA;
    s->sea.shards = (st_shard*)calloc(STANLEY_MAX_SEA, sizeof(st_shard));
    if (!s->sea.shards) { cooccur_free(&s->co); return -1; }
    s->coherence_floor          = 0.15f;
    s->coherence_floor_baseline = 0.15f;
    s->mass_threshold           = 0.85f;
    s->max_rings                = STANLEY_MAX_RINGS;
    s->ring_temp_scale          = 1.0f;
    s->ring_len_scale           = 1.0f;
    s->graze_rate               = 0.25f;
    s->somatic_temp_enabled     = 0;
    s->somatic_temp_strength    = 0.35f;
    s->last_temp_factor         = 1.0f;
    s->metastanley_enabled      = 0;
    s->metastanley_rate         = 0.35f;
    s->last_input_ts            = time(NULL);
    pthread_mutex_init(&s->mtx, NULL);

    /* seed rng from time to vary across runs */
    st_rng_state ^= (uint32_t)time(NULL);

    identity_load_from_origin(&s->me, &s->co, origin_path);
    return 0;
}

void stanley_free(Stanley *s) {
    stanley_shimmer_stop(s);
    stanley_graze_detach(s);
    pthread_mutex_destroy(&s->mtx);
    for (int i = 0; i < s->sea.n; i++) free(s->sea.shards[i].content);
    free(s->sea.shards);
    for (int i = 0; i < s->me.n_fragments; i++) free(s->me.fragments[i]);
    free(s->me.fragments);
    cooccur_free(&s->co);
    if (vocab_freq) { free(vocab_freq); vocab_freq = NULL; }
}

/* ============================================================
 * REPL
 * ============================================================ */

void stanley_repl(Stanley *s) {
    char line[2048];
    printf("stanley %s — weightless organism.\n", STANLEY_VERSION);
    printf("  /quit to exit, /stats for state, /dream to force consolidation.\n");
    printf("  (silence is a valid reply — stanley may not speak.)\n\n");
    fflush(stdout);
    while (1) {
        printf("you> ");
        fflush(stdout);
        if (!fgets(line, sizeof(line), stdin)) break;
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L == 0) continue;
        if (!strcmp(line, "/quit")) break;
        if (!strcmp(line, "/stats")) {
            int graze_v = graze_total_vocab(s);
            int speak_n = s->speak_window_filled ? STANLEY_SPEAK_WINDOW : s->speak_window_idx;
            int speak_sum = 0;
            for (int i = 0; i < speak_n; i++) speak_sum += s->speak_window[i];
            float ratio = speak_n > 0 ? (float)speak_sum / (float)speak_n : 0;
            printf("  vocab=%d  inputs=%lld  spoken=%lld  refused=%lld  dreams=%lld  shimmers=%lld\n",
                   s->co.n_vocab,
                   (long long)s->n_inputs, (long long)s->n_spoken,
                   (long long)s->n_refused, (long long)s->n_dreams,
                   (long long)s->n_shimmers);
            printf("  chambers: calm=%.2f spike=%.2f over=%.2f tired=%.2f overload=%.2f\n",
                   s->body.act[0], s->body.act[1], s->body.act[2], s->body.act[3], s->body.overload);
            printf("  identity: fragments=%d gravity=%d  sea=%d  pastures=%d  graze_vocab=%d\n",
                   s->me.n_fragments, s->me.n_gravity, s->sea.n, s->n_grazes, graze_v);
            printf("  grazing: profiled=%d\n", graze_profiled_pastures(s));
            printf("  memory: internal_charge=%.2f refused_charge=%.2f gravity_pressure=%.2f\n",
                   sea_recent_charge(s, 'I', 6),
                   sea_recent_charge(s, 'R', 6),
                   gravity_pressure(s));
            printf("  maturity: speak_ratio=%.2f  coherence_floor=%.3f (baseline %.3f)\n",
                   ratio, s->coherence_floor, s->coherence_floor_baseline);
            printf("  listening: max_rings=%d temp_scale=%.3f len_scale=%.3f graze_rate=%.3f\n",
                   s->max_rings, s->ring_temp_scale, s->ring_len_scale, s->graze_rate);
            printf("  somatic_temp: %s strength=%.3f factor=%.3f\n",
                   s->somatic_temp_enabled ? "on" : "off",
                   s->somatic_temp_strength,
                   s->last_temp_factor);
            printf("  inner: metastanley=%s rate=%.3f inner_ticks=%lld\n",
                   s->metastanley_enabled ? "on" : "off",
                   s->metastanley_rate,
                   (long long)s->n_inner);
            continue;
        }
        if (!strcmp(line, "/pastures")) {
            printf("  pastures=%d\n", s->n_grazes);
            for (int i = 0; i < s->n_grazes; i++) {
                printf("  %c pasture[%d]: %s (vocab=%d profile=%d pull=%.2f hits=%lld)\n",
                       i == 0 ? '*' : '-',
                       i,
                       s->graze_labels[i] ? s->graze_labels[i] : "(unknown)",
                       graze_vocab_size(s->grazes[i]),
                       graze_profile_size(s->grazes[i]),
                       graze_pasture_pull(s, i, 0),
                       (long long)s->graze_hits[i]);
                if (s->graze_profile_labels[i]) {
                    printf("    profile: %s\n", s->graze_profile_labels[i]);
                }
            }
            continue;
        }
        if (!strcmp(line, "/dream")) { stanley_dream(s); printf("  [dream]\n"); continue; }
        if (!strcmp(line, "/shimmer")) { stanley_shimmer_now(s); printf("  [shimmer]\n"); continue; }
        if (!strcmp(line, "/inner")) {
            printf("  inner_ticks=%lld\n", (long long)s->n_inner);
            printf("  last_inner=%s\n", s->last_inner[0] ? s->last_inner : "(none)");
            continue;
        }
        char *reply = stanley_tick(s, line);
        if (reply) { printf("stanley> %s\n", reply); free(reply); }
        else       { printf("stanley> ...\n"); }
    }
}

/* ============================================================
 * GRAZE — attach/detach optional GGUF vocab pasture
 * ============================================================ */

int stanley_graze_attach(Stanley *s, const char *gguf_path) {
    if (!s || !gguf_path) return -1;
    st_graze *g;
    char *label = NULL;
    pthread_mutex_lock(&s->mtx);
    if (s->n_grazes >= STANLEY_MAX_GRAZES) {
        pthread_mutex_unlock(&s->mtx);
        return -1;
    }
    g = graze_open(gguf_path);
    if (g) {
        label = st_strdup_local(gguf_path);
        s->grazes[s->n_grazes] = g;
        s->graze_labels[s->n_grazes] = label;
        s->graze_hits[s->n_grazes] = 0;
        s->n_grazes++;
    }
    int ok = (g != NULL);
    pthread_mutex_unlock(&s->mtx);
    return ok ? 0 : -1;
}

int stanley_graze_profile_attach(Stanley *s, const char *text_path) {
    if (!s || !text_path) return -1;
    pthread_mutex_lock(&s->mtx);
    if (s->n_grazes <= 0 || !s->grazes[s->n_grazes - 1]) {
        pthread_mutex_unlock(&s->mtx);
        return -1;
    }
    int idx = s->n_grazes - 1;
    int rc = graze_profile_load(s->grazes[idx], text_path);
    if (rc == 0) {
        char *label = st_strdup_local(text_path);
        if (s->graze_profile_labels[idx]) free(s->graze_profile_labels[idx]);
        s->graze_profile_labels[idx] = label;
    }
    pthread_mutex_unlock(&s->mtx);
    return rc;
}

void stanley_graze_detach(Stanley *s) {
    if (!s) return;
    pthread_mutex_lock(&s->mtx);
    for (int i = 0; i < s->n_grazes; i++) {
        if (s->grazes[i]) {
            graze_close(s->grazes[i]);
            s->grazes[i] = NULL;
        }
        if (s->graze_labels[i]) {
            free(s->graze_labels[i]);
            s->graze_labels[i] = NULL;
        }
        if (s->graze_profile_labels[i]) {
            free(s->graze_profile_labels[i]);
            s->graze_profile_labels[i] = NULL;
        }
        s->graze_hits[i] = 0;
    }
    s->n_grazes = 0;
    pthread_mutex_unlock(&s->mtx);
}

/* ============================================================
 * SHIMMER — Stanley dreams in silence
 *
 * After STANLEY_SHIMMER_IDLE_S seconds without input, if the chambers
 * are calm enough, fire one internal pass: synthetic pulse from current
 * body state, generate a deep ring, maybe crystallize. No speech, no
 * accumulation from outside — pure self-talk.
 * ============================================================ */

static void shimmer_pass(Stanley *s) {
    pthread_mutex_lock(&s->mtx);
    if (s->co.n_vocab < 8) {        /* not enough field to dream from */
        pthread_mutex_unlock(&s->mtx);
        return;
    }

    /* synthesize an internal pulse — derived from body, not input. */
    st_pulse p;
    p.novelty = 0;
    p.arousal = 0.1f * s->body.act[1];
    p.entropy = 0.4f + 0.3f * (1.0f - s->body.act[0]);
    p.valence = 0;

    /* run a single deep ring — level 3 ("deep") for maximum reflection. */
    st_ring r;
    ring_generate(s, &r, 3, p.entropy);
    s->n_shimmers++;

    /* shimmer raises a tiny tired tick + chips overflow */
    s->body.act[2] = st_clamp(s->body.act[2] + 0.02f, 0, 1);
    s->body.act[3] = st_clamp(s->body.act[3] + 0.01f, 0, 1);

    /* crystallize if it resonated */
    if (r.resonance > 1.5f && r.meta_patterns >= 2) {
        s->sea.step++;
        sea_push(&s->sea, 'I', r.content, r.resonance, s->sea.step);
    }
    pthread_mutex_unlock(&s->mtx);
}

static void *shimmer_loop(void *arg) {
    Stanley *s = (Stanley *)arg;
    while (s->shimmer_running) {
        sleep(STANLEY_SHIMMER_TICK_S);
        if (!s->shimmer_running) break;
        time_t now = time(NULL);
        time_t last = s->last_input_ts;
        if (now - last < STANLEY_SHIMMER_IDLE_S) continue;

        float calm = s->body.act[0];
        float over = s->body.act[2];
        if (calm < 0.5f || over > 0.4f) continue;       /* not the right body */

        shimmer_pass(s);
    }
    return NULL;
}

void stanley_shimmer_start(Stanley *s) {
    if (!s || s->shimmer_running) return;
    s->shimmer_running = 1;
    if (pthread_create(&s->shimmer_thr, NULL, shimmer_loop, s) != 0) {
        s->shimmer_running = 0;
    }
}

void stanley_shimmer_stop(Stanley *s) {
    if (!s || !s->shimmer_running) return;
    s->shimmer_running = 0;
    pthread_join(s->shimmer_thr, NULL);
}

void stanley_shimmer_now(Stanley *s) {
    if (!s) return;
    shimmer_pass(s);
}

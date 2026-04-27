/*
 * graze.c — minimal GGUF metadata-only vocab harvester.
 *
 * Reads tokenizer.ggml.tokens from a GGUF v3 file via mmap.
 * Tensor regions are never touched, so the OS keeps them swapped out.
 *
 * Parser is a stripped-down port of doe.c index_load() — only the bits
 * needed to walk header KV pairs and pull one string array.
 */
#include "graze.h"

#include <ctype.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define GGUF_MAGIC 0x46554747u /* "GGUF" little-endian */

struct st_graze {
    uint8_t *mmap_base;
    size_t   mmap_size;
    char   **tokens;
    int      n_tokens;
    char   **profile_words;
    int     *profile_counts;
    int      n_profile;
    int      profile_total;
};

#define GRAZE_MAX_PROFILE_WORDS 4096

/* GGUF type sizes for non-string scalars (used to skip values we ignore). */
static size_t gguf_scalar_size(uint32_t vtype) {
    switch (vtype) {
        case 0: case 1: case 7:           return 1; /* uint8, int8, bool */
        case 2: case 3:                   return 2; /* uint16, int16 */
        case 4: case 5: case 6:           return 4; /* uint32, int32, float32 */
        case 10: case 11: case 12:        return 8; /* uint64, int64, float64 */
        default:                          return 0; /* string (8) and array (9) handled separately */
    }
}

st_graze *graze_open(const char *path) {
    if (!path) return NULL;

    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0 || st.st_size < 24) {
        close(fd);
        return NULL;
    }

    uint8_t *base = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return NULL;

    st_graze *g = calloc(1, sizeof(*g));
    if (!g) { munmap(base, st.st_size); return NULL; }
    g->mmap_base = base;
    g->mmap_size = (size_t)st.st_size;

    uint8_t *p = base, *pend = base + st.st_size;

#define NEED(n) do { if (p + (n) > pend) goto bail; } while (0)

    NEED(4);
    if (*(uint32_t *)p != GGUF_MAGIC) goto bail;
    p += 4;
    NEED(4); p += 4; /* version (assume v2/v3 layout) */
    NEED(8); p += 8; /* n_tensors — skipped */
    NEED(8); uint64_t n_kv = *(uint64_t *)p; p += 8;

    for (uint64_t i = 0; i < n_kv; i++) {
        NEED(8); uint64_t klen = *(uint64_t *)p; p += 8;
        if (klen > 255 || p + klen > pend) goto bail;
        char key[256];
        memcpy(key, p, klen);
        key[klen] = 0;
        p += klen;

        NEED(4); uint32_t vtype = *(uint32_t *)p; p += 4;

        if (vtype == 8) { /* string */
            NEED(8); uint64_t vlen = *(uint64_t *)p; p += 8;
            if (p + vlen > pend) goto bail;
            p += vlen;
        } else if (vtype == 9) { /* array */
            NEED(4); uint32_t atype = *(uint32_t *)p; p += 4;
            NEED(8); uint64_t alen = *(uint64_t *)p; p += 8;

            if (atype == 8) { /* string array */
                int is_vocab = strstr(key, "tokenizer.ggml.tokens") != NULL;
                if (is_vocab && alen > 0 && alen < 200000) {
                    g->tokens = calloc((size_t)alen, sizeof(char *));
                    if (!g->tokens) goto bail;
                    g->n_tokens = (int)alen;
                }
                for (uint64_t ai = 0; ai < alen; ai++) {
                    NEED(8); uint64_t slen = *(uint64_t *)p; p += 8;
                    if (slen > 1000000 || p + slen > pend) goto bail;
                    if (is_vocab && g->tokens && (int)ai < g->n_tokens) {
                        g->tokens[ai] = malloc(slen + 1);
                        if (g->tokens[ai]) {
                            memcpy(g->tokens[ai], p, slen);
                            g->tokens[ai][slen] = 0;
                        }
                    }
                    p += slen;
                }
            } else {
                size_t esz = gguf_scalar_size(atype);
                if (esz == 0) goto bail;
                if (p + alen * esz > pend) goto bail;
                p += alen * esz;
            }
        } else {
            size_t esz = gguf_scalar_size(vtype);
            if (esz == 0) goto bail; /* unknown vtype */
            NEED(esz); p += esz;
        }
    }

    if (g->n_tokens == 0) goto bail;
    return g;

bail:
    graze_close(g);
    return NULL;

#undef NEED
}

void graze_close(st_graze *g) {
    if (!g) return;
    if (g->profile_words) {
        for (int i = 0; i < g->n_profile; i++) free(g->profile_words[i]);
        free(g->profile_words);
    }
    free(g->profile_counts);
    if (g->tokens) {
        for (int i = 0; i < g->n_tokens; i++) free(g->tokens[i]);
        free(g->tokens);
    }
    if (g->mmap_base) munmap(g->mmap_base, g->mmap_size);
    free(g);
}

int graze_vocab_size(const st_graze *g) {
    return g ? g->n_tokens : 0;
}

const char *graze_token(const st_graze *g, int idx) {
    if (!g || idx < 0 || idx >= g->n_tokens) return NULL;
    return g->tokens[idx];
}

/* Strip SentencePiece leading ▁ (U+2581 = E2 96 81) if present. */
static const char *strip_sp_marker(const char *s) {
    if (!s) return NULL;
    if ((unsigned char)s[0] == 0xE2 &&
        (unsigned char)s[1] == 0x96 &&
        (unsigned char)s[2] == 0x81) {
        return s + 3;
    }
    return s;
}

/* Heuristic: token is a "real word" if it contains at least one ASCII letter
 * AND does not start with a control marker like '<' or '['. */
static int looks_like_word(const char *raw) {
    if (!raw || !*raw) return 0;
    if (raw[0] == '<' || raw[0] == '[') return 0;
    const char *s = strip_sp_marker(raw);
    if (!*s || s[0] == '<' || s[0] == '[') return 0;
    int has_alpha = 0;
    for (const char *c = s; *c; c++) {
        unsigned char u = (unsigned char)*c;
        if ((u >= 'a' && u <= 'z') || (u >= 'A' && u <= 'Z')) {
            has_alpha = 1;
            break;
        }
    }
    return has_alpha;
}

static void graze_profile_clear(st_graze *g) {
    if (!g) return;
    if (g->profile_words) {
        for (int i = 0; i < g->n_profile; i++) free(g->profile_words[i]);
        free(g->profile_words);
    }
    free(g->profile_counts);
    g->profile_words = NULL;
    g->profile_counts = NULL;
    g->n_profile = 0;
    g->profile_total = 0;
}

static int profile_insert_word(st_graze *g, const char *word) {
    if (!g || !word || !*word) return -1;
    for (int i = 0; i < g->n_profile; i++) {
        if (strcmp(g->profile_words[i], word) == 0) {
            g->profile_counts[i]++;
            g->profile_total++;
            return 0;
        }
    }
    if (g->n_profile >= GRAZE_MAX_PROFILE_WORDS) return 0;
    char **new_words = realloc(g->profile_words, (size_t)(g->n_profile + 1) * sizeof(char *));
    if (!new_words) return -1;
    int  *new_counts = realloc(g->profile_counts, (size_t)(g->n_profile + 1) * sizeof(int));
    if (!new_counts) {
        g->profile_words = new_words;
        return -1;
    }
    g->profile_words = new_words;
    g->profile_counts = new_counts;
    g->profile_words[g->n_profile] = strdup(word);
    if (!g->profile_words[g->n_profile]) return -1;
    g->profile_counts[g->n_profile] = 1;
    g->n_profile++;
    g->profile_total++;
    return 0;
}

int graze_profile_load(st_graze *g, const char *text_path) {
    if (!g || !text_path) return -1;
    FILE *f = fopen(text_path, "rb");
    if (!f) return -1;

    graze_profile_clear(g);

    char buf[128];
    int n = 0;
    int ch;
    while ((ch = fgetc(f)) != EOF) {
        unsigned char c = (unsigned char)ch;
        if (isalpha(c) || (c == '\'' && n > 0)) {
            if (n < (int)sizeof(buf) - 1) {
                buf[n++] = (char)tolower(c);
            }
        } else if (n > 1) {
            buf[n] = 0;
            if (profile_insert_word(g, buf) != 0) {
                fclose(f);
                graze_profile_clear(g);
                return -1;
            }
            n = 0;
        } else {
            n = 0;
        }
    }
    if (n > 1) {
        buf[n] = 0;
        if (profile_insert_word(g, buf) != 0) {
            fclose(f);
            graze_profile_clear(g);
            return -1;
        }
    }

    fclose(f);
    return g->n_profile > 0 ? 0 : -1;
}

int graze_profile_size(const st_graze *g) {
    return g ? g->n_profile : 0;
}

static const char *graze_random_profile_word(const st_graze *g) {
    if (!g || g->n_profile <= 0 || g->profile_total <= 0) return NULL;
    int target = rand() % g->profile_total;
    int acc = 0;
    for (int i = 0; i < g->n_profile; i++) {
        acc += g->profile_counts[i];
        if (target < acc) return g->profile_words[i];
    }
    return g->profile_words[g->n_profile - 1];
}

const char *graze_random_word(const st_graze *g) {
    if (!g || g->n_tokens == 0 || !g->tokens) return NULL;
    if (g->n_profile > 0 && (rand() % 100) < 75) {
        const char *pw = graze_random_profile_word(g);
        if (pw && *pw) return pw;
    }
    for (int attempt = 0; attempt < 32; attempt++) {
        int idx = rand() % g->n_tokens;
        const char *t = g->tokens[idx];
        if (looks_like_word(t)) return strip_sp_marker(t);
    }
    return NULL;
}

/*
 * test_graze.c — GGUF metadata vocab harvester.
 *
 * Exits 77 (SKIP) if no GGUF is available on the host.
 */
#include "../graze.h"
#include "../stanley.h"
#include "check.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *resolve_gguf(int argc, char **argv) {
    if (argc > 1) return argv[1];
    static const char *candidates[] = {
        "weights/nano89-base-q4.gguf",
        "/Users/ataeff/Downloads/nanollama/weights/nano-base-q4_0.gguf",
        "/Users/ataeff/Downloads/nanollama/weights/nano-f16.gguf",
        NULL,
    };
    for (int i = 0; candidates[i]; i++)
        if (access(candidates[i], R_OK) == 0) return candidates[i];
    return NULL;
}

int main(int argc, char **argv) {
    /* missing-file case is ALWAYS testable */
    st_graze *g_missing = graze_open("/nonexistent/path/does/not/exist.gguf");
    CHECK(g_missing == NULL, "graze_open returns NULL on missing file");
    CHECK(graze_vocab_size(NULL) == 0, "vocab_size on NULL is 0");
    CHECK(graze_token(NULL, 0) == NULL, "token() on NULL returns NULL");
    CHECK(graze_random_word(NULL) == NULL, "random_word() on NULL returns NULL");
    graze_close(NULL);                       /* must not crash */

    const char *path = resolve_gguf(argc, argv);
    if (!path) {
        printf("  [SKIP] no GGUF available; pass one as argv[1] to enable parsing checks\n");
        CHECK_REPORT("graze (partial)");
    }

    printf("  [info] using %s\n", path);
    st_graze *g = graze_open(path);
    CHECK(g != NULL, "graze_open succeeds on real GGUF");
    int vsz = graze_vocab_size(g);
    CHECK(vsz >= 1000, "vocab size >= 1000");

    /* control tokens at index 0–2 in any LLaMA-family GGUF */
    const char *t0 = graze_token(g, 0);
    const char *t1 = graze_token(g, 1);
    CHECK(t0 && t0[0] == '<', "token[0] is a control marker (starts with '<')");
    CHECK(t1 && t1[0] == '<', "token[1] is a control marker");

    /* random_word skips control tokens and SentencePiece markers */
    int got_words = 0, got_clean = 1;
    for (int i = 0; i < 8; i++) {
        const char *w = graze_random_word(g);
        if (w) {
            got_words++;
            if (w[0] == '<' || w[0] == '[') got_clean = 0;
        }
    }
    CHECK(got_words > 0, "random_word returns non-NULL at least once in 8 tries");
    CHECK(got_clean, "random_word never returns control-marked tokens");

    graze_close(g);

    /* organism-level attach semantics: repeated attach appends a new pasture */
    Stanley s;
    CHECK(stanley_init(&s, NULL) == 0, "stanley_init for multi-graze check");
    CHECK(stanley_graze_attach(&s, path) == 0, "first graze attach succeeds");
    CHECK(s.n_grazes == 1, "first attach creates one pasture");
    CHECK(stanley_graze_attach(&s, path) == 0, "second graze attach also succeeds");
    CHECK(s.n_grazes == 2, "second attach appends instead of replacing");
    stanley_free(&s);

    CHECK_REPORT("graze");
}

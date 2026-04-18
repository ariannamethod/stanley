/*
 * test_graze.c — verifies graze.c reads vocab from a real GGUF file.
 *
 * Pass the GGUF path as argv[1]. Default tries common local files.
 * Prints first 10 tokens + 5 random words, asserts vocab > 1000.
 */
#include "../graze.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char *fallback_paths[] = {
    "/Users/ataeff/Downloads/nanollama/weights/nano-base-q4_0.gguf",
    "/Users/ataeff/Downloads/nanollama/weights/nano-f16.gguf",
    NULL,
};

static const char *resolve_path(int argc, char **argv) {
    if (argc > 1) return argv[1];
    for (int i = 0; fallback_paths[i]; i++) {
        if (access(fallback_paths[i], R_OK) == 0) return fallback_paths[i];
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *path = resolve_path(argc, argv);
    if (!path) {
        fprintf(stderr, "test_graze: no GGUF available; pass one as argv[1]\n");
        return 77; /* SKIP exit code */
    }

    printf("[graze] opening %s\n", path);
    st_graze *g = graze_open(path);
    if (!g) {
        fprintf(stderr, "FAIL: graze_open returned NULL\n");
        return 1;
    }

    int vsz = graze_vocab_size(g);
    printf("[graze] vocab_size = %d\n", vsz);
    if (vsz < 1000) {
        fprintf(stderr, "FAIL: vocab too small (%d, expected > 1000)\n", vsz);
        graze_close(g);
        return 1;
    }

    printf("[graze] first 10 tokens:\n");
    for (int i = 0; i < 10 && i < vsz; i++) {
        const char *t = graze_token(g, i);
        printf("  [%2d] '%s'\n", i, t ? t : "(null)");
    }

    printf("[graze] 5 random words:\n");
    int got = 0;
    for (int i = 0; i < 5; i++) {
        const char *w = graze_random_word(g);
        if (w) { printf("  '%s'\n", w); got++; }
    }
    if (got == 0) {
        fprintf(stderr, "FAIL: graze_random_word returned NULL 5 times\n");
        graze_close(g);
        return 1;
    }

    graze_close(g);
    printf("PASS test_graze\n");
    return 0;
}

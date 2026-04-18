/*
 * main.c — thin CLI wrapper for stanley.
 *
 * usage:
 *   ./stanley                       REPL with origin.txt (if present)
 *   ./stanley --no-origin           REPL with no origin — Stanley starts silent
 *   ./stanley --origin PATH         REPL with alternate origin
 *   ./stanley --graze PATH.gguf     attach external GGUF as vocab pasture
 *   ./stanley --shimmer             enable idle dream thread
 *   ./stanley --help
 */

#include "stanley.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *p) {
    printf("stanley %s — weightless organism.\n", STANLEY_VERSION);
    printf("usage: %s [--origin PATH] [--no-origin] [--graze GGUF] [--shimmer] [--help]\n", p);
}

int main(int argc, char **argv) {
    const char *origin = "origin.txt";
    const char *graze_path = NULL;
    int shimmer = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        if (!strcmp(argv[i], "--no-origin")) origin = NULL;
        else if (!strcmp(argv[i], "--origin") && i + 1 < argc) origin = argv[++i];
        else if (!strcmp(argv[i], "--graze")  && i + 1 < argc) graze_path = argv[++i];
        else if (!strcmp(argv[i], "--shimmer")) shimmer = 1;
    }
    Stanley s;
    if (stanley_init(&s, origin) != 0) {
        fprintf(stderr, "stanley: init failed\n");
        return 1;
    }
    if (graze_path) {
        if (stanley_graze_attach(&s, graze_path) != 0) {
            fprintf(stderr, "stanley: graze attach failed for %s — continuing weightless\n", graze_path);
        } else {
            fprintf(stderr, "stanley: grazing on %s\n", graze_path);
        }
    }
    if (shimmer) {
        stanley_shimmer_start(&s);
        fprintf(stderr, "stanley: shimmer thread on (idle %ds)\n", STANLEY_SHIMMER_IDLE_S);
    }
    stanley_repl(&s);
    stanley_free(&s);
    return 0;
}

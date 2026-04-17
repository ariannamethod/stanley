/*
 * main.c — thin CLI wrapper for stanley 2.0.
 *
 * usage:
 *   ./stanley                # REPL with origin.txt (if present)
 *   ./stanley --no-origin    # REPL with no origin — Stanley starts silent
 *   ./stanley --origin PATH  # REPL with alternate origin
 *   ./stanley --help
 */

#include "stanley.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *p) {
    printf("stanley %s — weightless organism.\n", STANLEY_VERSION);
    printf("usage: %s [--origin PATH] [--no-origin] [--help]\n", p);
}

int main(int argc, char **argv) {
    const char *origin = "origin.txt";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        if (!strcmp(argv[i], "--no-origin")) origin = NULL;
        else if (!strcmp(argv[i], "--origin") && i + 1 < argc) origin = argv[++i];
    }
    Stanley s;
    if (stanley_init(&s, origin) != 0) {
        fprintf(stderr, "stanley: init failed\n");
        return 1;
    }
    stanley_repl(&s);
    stanley_free(&s);
    return 0;
}

/*
 * main.c — thin CLI wrapper for stanley.
 *
 * usage:
 *   ./stanley                       REPL with origin.txt (if present)
 *   ./stanley --no-origin           REPL with no origin — Stanley starts silent
 *   ./stanley --origin PATH         REPL with alternate origin
 *   ./stanley --graze PATH.gguf     attach external GGUF as vocab pasture
 *   ./stanley --graze-profile TXT   bias the most recent pasture with a text profile
 *   ./stanley --shimmer             enable idle dream thread
 *   ./stanley --help
 */

#include "stanley.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char *p) {
    printf("stanley %s — weightless organism.\n", STANLEY_VERSION);
    printf("usage: %s [--origin PATH] [--no-origin] [--graze GGUF]... [--graze-profile TXT]... [--shimmer] [--help]\n", p);
}

int main(int argc, char **argv) {
    const char *origin = "origin.txt";
    const char *graze_paths[STANLEY_MAX_GRAZES];
    const char *graze_profiles[STANLEY_MAX_GRAZES];
    int n_graze_paths = 0;
    int last_graze = -1;
    int shimmer = 0;
    for (int i = 0; i < STANLEY_MAX_GRAZES; i++) graze_profiles[i] = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        if (!strcmp(argv[i], "--no-origin")) origin = NULL;
        else if (!strcmp(argv[i], "--origin") && i + 1 < argc) origin = argv[++i];
        else if (!strcmp(argv[i], "--graze")  && i + 1 < argc) {
            const char *path = argv[++i];
            if (n_graze_paths < STANLEY_MAX_GRAZES) {
                graze_paths[n_graze_paths++] = path;
                last_graze = n_graze_paths - 1;
            } else {
                fprintf(stderr, "stanley: too many --graze pastures (max %d); ignoring %s\n",
                        STANLEY_MAX_GRAZES, path);
            }
        }
        else if (!strcmp(argv[i], "--graze-profile") && i + 1 < argc) {
            const char *path = argv[++i];
            if (last_graze >= 0) {
                graze_profiles[last_graze] = path;
            } else {
                fprintf(stderr, "stanley: --graze-profile %s ignored (attach a pasture first)\n", path);
            }
        }
        else if (!strcmp(argv[i], "--shimmer")) shimmer = 1;
    }
    Stanley s;
    if (stanley_init(&s, origin) != 0) {
        fprintf(stderr, "stanley: init failed\n");
        return 1;
    }
    for (int i = 0; i < n_graze_paths; i++) {
        if (stanley_graze_attach(&s, graze_paths[i]) != 0) {
            fprintf(stderr, "stanley: graze attach failed for %s — continuing weightless\n", graze_paths[i]);
        } else {
            fprintf(stderr, "stanley: grazing on %s\n", graze_paths[i]);
            if (graze_profiles[i]) {
                if (stanley_graze_profile_attach(&s, graze_profiles[i]) == 0) {
                    fprintf(stderr, "stanley: tuned last pasture with %s\n", graze_profiles[i]);
                } else {
                    fprintf(stderr, "stanley: graze profile failed for %s — keeping untuned pasture\n", graze_profiles[i]);
                }
            }
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

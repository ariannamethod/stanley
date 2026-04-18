/*
 * test_maturity.c — adaptive coherence_floor drift via speak/silence ratio.
 */
#include "../stanley.h"
#include "check.h"

#include <stdlib.h>

int main(void) {
    /* ----- ratio > 0.7 raises floor ----- */
    Stanley s1;
    stanley_init(&s1, NULL);
    stanley_accumulate(&s1, "pressure makes motion echoes flow currents", NULL);
    float floor0 = s1.coherence_floor;
    for (int i = 0; i < 64; i++) s1.speak_window[i] = 1;
    s1.speak_window_idx = 0;
    s1.speak_window_filled = 1;
    char *r = stanley_tick(&s1, "ping");
    if (r) free(r);
    CHECK(s1.coherence_floor > floor0, "speak_ratio=1.0 raises coherence_floor");

    /* ----- ratio < 0.2 lowers floor (after we've raised it) ----- */
    s1.coherence_floor = s1.coherence_floor_baseline + 0.1f;
    float floor1 = s1.coherence_floor;
    for (int i = 0; i < 64; i++) s1.speak_window[i] = 0;
    char *r2 = stanley_tick(&s1, "ping again");
    if (r2) free(r2);
    CHECK(s1.coherence_floor < floor1, "speak_ratio=0 lowers coherence_floor");

    /* ----- floor never falls below baseline ----- */
    Stanley s2;
    stanley_init(&s2, NULL);
    stanley_accumulate(&s2, "pressure makes motion echoes flow currents", NULL);
    for (int i = 0; i < 64; i++) s2.speak_window[i] = 0;
    s2.speak_window_filled = 1;
    s2.coherence_floor = s2.coherence_floor_baseline;        /* already at floor */
    for (int i = 0; i < 50; i++) {
        char *rr = stanley_tick(&s2, "ping");
        if (rr) free(rr);
    }
    CHECK(s2.coherence_floor >= s2.coherence_floor_baseline - 0.01f,
          "floor refuses to drift below baseline (after 50 silent ticks)");

    /* ----- floor caps at baseline + 0.3 ----- */
    Stanley s3;
    stanley_init(&s3, NULL);
    stanley_accumulate(&s3, "pressure makes motion echoes flow currents", NULL);
    for (int i = 0; i < 64; i++) s3.speak_window[i] = 1;
    s3.speak_window_filled = 1;
    for (int i = 0; i < 200; i++) {
        char *rr = stanley_tick(&s3, "ping");
        if (rr) free(rr);
    }
    CHECK(s3.coherence_floor <= s3.coherence_floor_baseline + 0.31f,
          "floor caps at baseline + 0.3 (after 200 talkative ticks)");

    stanley_free(&s1);
    stanley_free(&s2);
    stanley_free(&s3);
    CHECK_REPORT("maturity");
}

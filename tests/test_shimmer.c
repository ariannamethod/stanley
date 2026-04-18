/*
 * test_shimmer.c — Stanley dreams in silence.
 */
#include "../stanley.h"
#include "check.h"

#include <stdlib.h>
#include <unistd.h>

int main(void) {
    /* ----- shimmer_now bumps n_shimmers when vocab is non-trivial ----- */
    Stanley s;
    stanley_init(&s, NULL);
    stanley_accumulate(&s,
        "pressure made motion echoes flow currents tide swell rise fall hum drift", NULL);
    int before = (int)s.n_shimmers;
    stanley_shimmer_now(&s);
    CHECK((int)s.n_shimmers == before + 1, "shimmer_now increments n_shimmers");

    /* ----- shimmer is a no-op on an empty Stanley (vocab too thin) ----- */
    Stanley empty;
    stanley_init(&empty, NULL);
    stanley_shimmer_now(&empty);
    CHECK(empty.n_shimmers == 0, "shimmer is a no-op when vocab < 8");
    stanley_free(&empty);

    /* ----- shimmer raises the tired chamber a tiny tick ----- */
    float tired_before = s.body.act[3];
    stanley_shimmer_now(&s);
    CHECK(s.body.act[3] > tired_before, "shimmer raises tired chamber");

    /* ----- thread lifecycle: start raises flag, stop clears it ----- */
    Stanley t;
    stanley_init(&t, NULL);
    CHECK(t.shimmer_running == 0, "shimmer flag default is 0");
    stanley_shimmer_start(&t);
    CHECK(t.shimmer_running == 1, "shimmer_start raises running flag");
    stanley_shimmer_stop(&t);
    CHECK(t.shimmer_running == 0, "shimmer_stop clears running flag");
    /* idempotent stop must not crash */
    stanley_shimmer_stop(&t);
    CHECK(1, "shimmer_stop is idempotent");
    stanley_free(&t);

    stanley_free(&s);
    CHECK_REPORT("shimmer");
}

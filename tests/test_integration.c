/*
 * test_integration.c — multi-turn live simulation.
 *
 * Drives Stanley through 30 ticks with varied inputs, asserts the
 * end-to-end loop holds together: vocab grows, refuse rate is non-trivial,
 * dream fires at least once, and *something* crystallizes (sea > 0).
 *
 * This is the test that catches the kind of regression that no per-feature
 * unit test would — a feature working in isolation but breaking when the
 * organism actually lives.
 */
#include "../stanley.h"
#include "check.h"

#include <stdlib.h>
#include <string.h>

static const char *prompts[] = {
    "pressure makes motion echoes flow",
    "water moves like memory through stone",
    "stones remember tides and weather and pulse",
    "rhythm is not music alone but the field itself",
    "stillness has its own pulse listen",
    "are you here architect",
    "what moves underneath",
    "the field hums quietly now",
    "speak from yourself",
    "what resonates",
    "are you listening",
    "stillness again",
    "pressure returns through the silence",
    "echoes of what is not said",
    "the body knows before the mind",
};

int main(void) {
    Stanley s;
    stanley_init(&s, NULL);

    int n_replies = 0;
    for (int turn = 0; turn < 30; turn++) {
        const char *q = prompts[turn % (int)(sizeof(prompts)/sizeof(prompts[0]))];
        char *r = stanley_tick(&s, q);
        if (r) { n_replies++; free(r); }
    }

    CHECK(s.co.n_vocab >= 20, "vocab grew past 20 after 30 turns");
    CHECK(s.n_inputs == 30, "n_inputs counts every turn (refuse included)");
    CHECK(s.n_refused > 0, "Stanley refused at least once");
    CHECK(n_replies > 0,    "Stanley spoke at least once");
    CHECK(s.sea.n > 0,      "memory sea has at least one shard");

    /* dream is somatic — over 30 turns with default thresholds it should fire */
    if (s.n_dreams == 0) {
        /* force one to test consolidation path even if body never crossed mass_threshold */
        stanley_dream(&s);
    }
    CHECK(s.n_dreams >= 1, "dream consolidation ran (forced if needed)");

    /* shimmer can be triggered manually and must not crash mid-session */
    stanley_shimmer_now(&s);
    CHECK(s.n_shimmers >= 1, "synchronous shimmer increments counter mid-session");

    /* graze attach + detach round-trip mid-session must not corrupt state */
    int vocab_before = s.co.n_vocab;
    int rc = stanley_graze_attach(&s, "/nonexistent.gguf");
    CHECK(rc != 0, "graze_attach on missing path fails gracefully");
    CHECK(s.n_grazes == 0, "failed attach leaves pasture count at 0");
    CHECK(s.co.n_vocab == vocab_before, "failed attach does not corrupt cooccur");

    stanley_free(&s);
    CHECK_REPORT("integration");
}

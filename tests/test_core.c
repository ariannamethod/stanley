/*
 * test_core.c — v2.0 base behaviours that 2.1 must preserve.
 * Covers: pulse, cooccur, chambers, refuse gate, dream consolidation.
 */
#include "../stanley.h"
#include "check.h"

#include <stdlib.h>
#include <string.h>

int main(void) {
    Stanley s;
    int rc = stanley_init(&s, NULL);
    CHECK(rc == 0, "init no-origin succeeds");
    CHECK(s.co.n_vocab == 0, "vocab fresh = 0");
    CHECK(s.body.act[0] > 0.5f, "body starts calm");

    /* ----- pulse axes ----- */
    st_pulse caps = stanley_pulse(&s, "HELLO WORLD!!!");
    CHECK(caps.arousal > 0.4f, "caps + punctuation raise arousal");
    st_pulse low = stanley_pulse(&s, "hello world");
    CHECK(low.arousal < 0.3f, "lowercase plain text low arousal");

    /* novelty rises on unseen tokens */
    st_pulse novel = stanley_pulse(&s, "qzxk plk wqrz mnbv");
    CHECK(novel.novelty > 0.9f, "all-unseen tokens push novelty near 1");

    /* ----- cooccur grows monotonically ----- */
    stanley_accumulate(&s, "pressure made motion echo flow", NULL);
    int v1 = s.co.n_vocab;
    CHECK(v1 >= 5, "vocab grew with first accumulate");
    stanley_accumulate(&s, "pressure and motion again", NULL);
    CHECK(s.co.n_vocab == v1 + 2, "vocab adds only new tokens (and, again)");

    /* ----- one tick should not crash ----- */
    char *r = stanley_tick(&s, "pressure is what pressure does");
    if (r) free(r);
    CHECK(1, "tick runs without crash on familiar tokens");

    /* ----- dream relaxes overload ----- */
    s.body.act[2] = 1.0f;
    s.body.overload = 1.0f;
    stanley_dream(&s);
    CHECK(s.body.act[2] < 1.0f, "dream relaxes overflow chamber");
    CHECK(s.body.overload == 0.0f, "dream clears overload");

    /* ----- strong deep ring crystallizes into an internal shard ----- */
    st_ring deep = {0};
    deep.level = 3;
    deep.length = 15;
    deep.resonance = 2.4f;
    deep.meta_patterns = 1;
    strcpy(deep.content, "pressure becomes form in the dark");
    s.body.overload = 0.9f;
    s.body.act[3] = 0.4f;
    int sea_before = s.sea.n;
    stanley_crystallize(&s, &deep, 1);
    CHECK(s.sea.n == sea_before + 1, "strong ring crystallizes into an internal shard");
    CHECK(s.sea.shards[s.sea.n - 1].kind == 'I', "crystallized shard kind is I");

    /* same ring immediately after should not duplicate in the recent window */
    stanley_crystallize(&s, &deep, 1);
    CHECK(s.sea.n == sea_before + 1, "recent duplicate internal shard is suppressed");

    /* ----- refuse gate fires under chamber overload + hot pulse ----- */
    s.body.act[2] = 0.9f;
    st_pulse hot = { .novelty = 0.5f, .arousal = 0.9f, .entropy = 0.8f, .valence = 0.0f };
    CHECK(stanley_refuses(&s, hot) == 1, "refuses under overload + hot pulse");

    /* ----- novelty gate when no identity exists ----- */
    s.body.act[2] = 0.1f;
    Stanley empty;
    stanley_init(&empty, NULL);
    st_pulse alien = { .novelty = 0.95f, .arousal = 0.2f, .entropy = 0.3f, .valence = 0 };
    CHECK(stanley_refuses(&empty, alien) == 1, "refuses alien input with no identity fragments");
    stanley_free(&empty);

    stanley_free(&s);
    CHECK_REPORT("core");
}

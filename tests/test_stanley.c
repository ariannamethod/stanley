/*
 * test_stanley.c — minimal smoke for stanley 2.0.
 *
 *   cc -I.. ../stanley.c test_stanley.c -o test_stanley -lm -lpthread
 *   ./test_stanley
 */

#include "../stanley.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int pass = 0, fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { pass++; printf("  [PASS] %s\n", name); } \
    else      { fail++; printf("  [FAIL] %s\n", name); } \
} while (0)

int main(void) {
    Stanley s;
    int rc = stanley_init(&s, NULL);
    CHECK(rc == 0, "stanley_init(no origin) succeeds");
    CHECK(s.co.n_vocab == 0, "fresh vocab is empty");
    CHECK(s.body.act[0] > 0.5f, "body starts calm");

    /* pulse has expected shape */
    st_pulse p = stanley_pulse(&s, "HELLO WORLD");
    CHECK(p.arousal > 0.4f, "caps input raises arousal");

    p = stanley_pulse(&s, "hello world");
    CHECK(p.arousal < 0.3f, "lowercase input has low arousal");

    /* accumulate grows vocab */
    stanley_accumulate(&s, "pressure came first pressure made motion", NULL);
    CHECK(s.co.n_vocab >= 5, "vocab grew after accumulate");
    int pressure_before = s.co.n_vocab;
    stanley_accumulate(&s, "pressure and motion again", NULL);
    CHECK(s.co.n_vocab == pressure_before + 2, "vocab grows only by new words (and, again)");

    /* one tick with known tokens should not crash */
    char *r = stanley_tick(&s, "pressure is what pressure does");
    /* silence is valid — only assert no crash */
    if (r) free(r);
    CHECK(1, "tick runs without crash");

    /* dream relaxes overload */
    s.body.act[2] = 1.0f;
    s.body.overload = 1.0f;
    stanley_dream(&s);
    CHECK(s.body.act[2] < 1.0f, "dream relaxes overflow chamber");
    CHECK(s.body.overload == 0.0f, "dream clears overload");

    /* refuse fires when chambers overloaded + high arousal */
    s.body.act[2] = 0.9f;
    st_pulse hot = { .novelty = 0.5f, .arousal = 0.9f, .entropy = 0.8f, .valence = 0.0f };
    CHECK(stanley_refuses(&s, hot) == 1, "refuses under chamber overload + hot pulse");

    /* pulse novelty is high on wholly unseen input */
    p = stanley_pulse(&s, "sdlkfjsldkfjslkdjf qwerty zxcvbn");
    CHECK(p.novelty > 0.9f, "novel words trigger high novelty");

    stanley_free(&s);

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}

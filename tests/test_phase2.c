/*
 * test_phase2.c — smoke for Phase 2 features:
 *   - vocab_graze attach
 *   - refused shard recording
 *   - adaptive coherence_floor maturity drift
 *   - shimmer pass produces a shimmer event
 */
#include "../stanley.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int pass = 0, fail = 0;
#define CHECK(cond, name) do { \
    if (cond) { pass++; printf("  [PASS] %s\n", name); } \
    else      { fail++; printf("  [FAIL] %s\n", name); } \
} while (0)

static const char *gguf_candidates[] = {
    "/Users/ataeff/Downloads/nanollama/weights/nano-base-q4_0.gguf",
    "/Users/ataeff/Downloads/nanollama/weights/nano-f16.gguf",
    NULL,
};

int main(void) {
    Stanley s;
    if (stanley_init(&s, NULL) != 0) {
        fprintf(stderr, "init failed\n");
        return 1;
    }

    /* ---- graze attach ---- */
    const char *gguf = NULL;
    for (int i = 0; gguf_candidates[i]; i++) {
        if (access(gguf_candidates[i], R_OK) == 0) { gguf = gguf_candidates[i]; break; }
    }
    if (gguf) {
        int rc = stanley_graze_attach(&s, gguf);
        CHECK(rc == 0, "graze_attach succeeds on real GGUF");
        CHECK(s.graze != NULL, "graze pointer set after attach");
        stanley_graze_detach(&s);
        CHECK(s.graze == NULL, "graze_detach clears pointer");
    } else {
        printf("  [SKIP] no GGUF available — graze tests skipped\n");
    }

    /* ---- refused shard recording ---- */
    /* Force an overloaded body so refuse fires deterministically. */
    s.body.act[2] = 0.95f;
    s.body.act[1] = 0.9f;
    int sea_before = s.sea.n;
    char *r = stanley_tick(&s, "EVERYTHING IS LOUD AND HOT!!!");
    if (r) free(r);
    CHECK(s.sea.n == sea_before + 1, "refuse writes one shard");
    int found_R = 0;
    for (int i = 0; i < s.sea.n; i++) {
        if (s.sea.shards[i].kind == 'R') { found_R = 1; break; }
    }
    CHECK(found_R, "shard kind 'R' present after refuse");
    CHECK(s.n_refused >= 1, "n_refused incremented");

    /* ---- adaptive maturity ---- */
    /* Force speak_window full of 1's, simulate over-talkative Stanley. */
    Stanley m;
    stanley_init(&m, NULL);
    float floor0 = m.coherence_floor;
    for (int i = 0; i < 64; i++) m.speak_window[i] = 1;
    m.speak_window_idx = 0;
    m.speak_window_filled = 1;
    /* one tick to trigger maturity_update. need vocab so it runs cleanly. */
    stanley_accumulate(&m, "test test test test test test test", NULL);
    char *rr = stanley_tick(&m, "ping");
    if (rr) free(rr);
    CHECK(m.coherence_floor > floor0, "high speak_ratio raises coherence_floor");

    /* opposite direction: window of 0's with floor above baseline */
    m.coherence_floor = m.coherence_floor_baseline + 0.1f;
    float floor1 = m.coherence_floor;
    for (int i = 0; i < 64; i++) m.speak_window[i] = 0;
    char *rrr = stanley_tick(&m, "ping again");
    if (rrr) free(rrr);
    CHECK(m.coherence_floor < floor1, "low speak_ratio lowers coherence_floor");
    stanley_free(&m);

    /* ---- shimmer pass ---- */
    Stanley sh;
    stanley_init(&sh, NULL);
    stanley_accumulate(&sh,
        "pressure made motion echoes flows currents tide swell rise fall hum drift", NULL);
    int shim_before = (int)sh.n_shimmers;
    stanley_shimmer_now(&sh);
    CHECK((int)sh.n_shimmers == shim_before + 1, "shimmer_now bumps n_shimmers");
    stanley_free(&sh);

    /* ---- shimmer thread lifecycle (no crash) ---- */
    Stanley st;
    stanley_init(&st, NULL);
    stanley_shimmer_start(&st);
    CHECK(st.shimmer_running == 1, "shimmer_start raises running flag");
    stanley_shimmer_stop(&st);
    CHECK(st.shimmer_running == 0, "shimmer_stop clears running flag");
    stanley_free(&st);

    stanley_free(&s);

    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail == 0 ? 0 : 1;
}

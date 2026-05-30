/*
 * test_refused.c — silence-as-teacher pipeline.
 *   refuse → 'R' shard with pulse fingerprint
 *   dream  → cluster R-shards by pulse similarity → promote centroid into gravity + scar
 */
#include "../stanley.h"
#include "check.h"

#include <stdlib.h>

int main(void) {
    /* ----- a single refuse writes one 'R' shard ----- */
    Stanley s;
    stanley_init(&s, NULL);
    s.body.act[2] = 0.95f;
    s.body.act[1] = 0.9f;
    int sea_before = s.sea.n;
    char *r = stanley_tick(&s, "EVERYTHING IS LOUD AND HOT!!!");
    if (r) free(r);
    CHECK(s.sea.n == sea_before + 1, "refuse writes exactly one shard");
    int found_R = 0;
    for (int i = 0; i < s.sea.n; i++)
        if (s.sea.shards[i].kind == 'R') { found_R = 1; break; }
    CHECK(found_R, "the new shard has kind == 'R'");
    CHECK(s.n_refused >= 1, "n_refused incremented");

    /* the refused shard must carry a pulse fingerprint, not random bytes */
    int has_pulse = 0;
    for (int i = 0; i < s.sea.n; i++) {
        if (s.sea.shards[i].kind != 'R') continue;
        st_pulse p = s.sea.shards[i].pulse;
        if (p.arousal > 0 || p.novelty > 0 || p.entropy > 0) has_pulse = 1;
    }
    CHECK(has_pulse, "R-shard carries non-zero pulse fingerprint");
    stanley_free(&s);

    /* ----- 3 similar refusals + dream → gravity climbs ----- */
    Stanley c;
    stanley_init(&c, NULL);
    stanley_accumulate(&c, "pressure makes motion echoes flow currents tide swell", NULL);

    /* manufacture 4 nearly-identical R-shards directly */
    st_pulse p = { .novelty = 0.4f, .arousal = 0.8f, .entropy = 0.6f, .valence = 0.0f };
    for (int i = 0; i < 4; i++) {
        c.sea.step++;
        int slot;
        if (c.sea.n < c.sea.capacity) { slot = c.sea.n++; }
        else                          { slot = c.sea.head; c.sea.head++; }
        c.sea.shards[slot].kind = 'R';
        c.sea.shards[slot].content = NULL;
        c.sea.shards[slot].resonance = 0;
        c.sea.shards[slot].created_step = c.sea.step;
        c.sea.shards[slot].pulse = p;
        c.sea.shards[slot].pulse.arousal += 0.01f * (float)i;   /* tiny jitter, still cluster */
    }
    int gravity_before = c.me.n_gravity;
    stanley_dream(&c);
    CHECK(c.me.n_gravity > gravity_before, "R-shard cluster of 4 promotes into gravity");
    CHECK(c.n_scars == 1, "R-shard cluster leaves one scar");

    int found_S = 0;
    for (int i = 0; i < c.sea.n; i++) {
        if (c.sea.shards[i].kind == 'S') {
            found_S = 1;
            CHECK(c.sea.shards[i].resonance > 0.0f, "scar carries repulsive strength");
            CHECK(c.sea.shards[i].pulse.arousal > 0.0f, "scar carries pulse centroid");
            break;
        }
    }
    CHECK(found_S, "scar shard kind is S");

    /* the cluster members must be tombstoned ('X') after promotion */
    int still_R = 0;
    for (int i = 0; i < c.sea.n; i++) if (c.sea.shards[i].kind == 'R') still_R++;
    CHECK(still_R == 0, "cluster members tombstoned after promotion (no 'R' left)");

    /* ----- similar future pulse hits the scar lane and refuses earlier ----- */
    c.body.act[0] = 0.80f;
    c.body.act[1] = 0.60f;
    c.body.act[2] = 0.50f;
    c.body.act[3] = 0.10f;
    CHECK(stanley_refuses(&c, p) == 1, "similar tense pulse is gated by scar pressure");

    /* ----- 2 similar refusals + dream → no promotion (cluster too small) ----- */
    Stanley d;
    stanley_init(&d, NULL);
    stanley_accumulate(&d, "pressure makes motion echoes flow currents tide swell", NULL);
    for (int i = 0; i < 2; i++) {
        d.sea.step++;
        int slot = d.sea.n++;
        d.sea.shards[slot].kind = 'R';
        d.sea.shards[slot].content = NULL;
        d.sea.shards[slot].pulse = p;
    }
    int g0 = d.me.n_gravity;
    stanley_dream(&d);
    CHECK(d.me.n_gravity == g0, "cluster of 2 does NOT promote (threshold is 3)");
    CHECK(d.n_scars == 0, "cluster of 2 does NOT leave a scar");

    stanley_free(&c);
    stanley_free(&d);
    CHECK_REPORT("refused");
}

/*
 * graze.h — vocab substrate for Stanley.
 *
 * Stanley is weightless. graze attaches an external GGUF as an OPTIONAL
 * lexical pasture: when chambers signal hunger, Stanley may sample a foreign
 * word from the substrate. No inference. No tensor reads. mmap reaches the
 * vocab metadata only — the rest of the file stays cold on disk.
 *
 * Substrate is opt-in: stanley_graze_attach() may fail (file missing, bad
 * GGUF) and the organism keeps running in pure cooccur mode.
 */
#ifndef STANLEY_GRAZE_H
#define STANLEY_GRAZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_graze st_graze;

/* Open a GGUF and harvest its tokenizer.ggml.tokens vocabulary.
 * Returns NULL on any failure (missing file, malformed GGUF, no vocab).
 * Stanley calls this opportunistically; failure is not fatal. */
st_graze *graze_open(const char *gguf_path);

/* Free vocab table and unmap the GGUF. NULL-safe. */
void graze_close(st_graze *g);

/* How many tokens the substrate exposes (0 if g == NULL). */
int graze_vocab_size(const st_graze *g);

/* Raw token by index. Returns NULL if out of range. May contain BPE markers
 * like leading U+2581. Caller must NOT free. */
const char *graze_token(const st_graze *g, int idx);

/* Pick a token that looks like a real word: contains [A-Za-z], skips
 * control tokens (<s>, [INST], etc.), strips SentencePiece leading ▁
 * marker if present. Returns NULL on g == NULL or no candidate after 32
 * tries. Caller must NOT free. */
const char *graze_random_word(const st_graze *g);

/* Optional lexical tuning profile from a plain text file.
 * The profile does not touch tensors; it only harvests weighted words from
 * the text so Stanley can preferentially graze them from this pasture.
 * Returns 0 on success, nonzero on failure. Replaces any previous profile. */
int graze_profile_load(st_graze *g, const char *text_path);

/* Number of unique words loaded into the optional lexical profile. */
int graze_profile_size(const st_graze *g);

#ifdef __cplusplus
}
#endif

#endif /* STANLEY_GRAZE_H */

/* Character-level tokenizer.
 *
 * One token per byte value present in the corpus. This is the least interesting
 * tokenizer that works, chosen deliberately: BPE is a substantial piece of work
 * in its own right and would sit between this project and the thing it is about,
 * which is the forward and backward passes. A character vocabulary also keeps
 * the embedding and output matrices small enough that the parameter count stays
 * dominated by the transformer blocks rather than by the vocabulary.
 *
 * The cost is real and worth stating: sequences are several times longer than
 * they would be under BPE, so a given context length covers far less text, and
 * attention is quadratic in that length.
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>

#define TOKENIZER_MAX_VOCAB 256

typedef struct {
    /* Dense id for each byte, or -1 when that byte never appears. Indexed by
     * the byte value itself, so encoding is a single array lookup. */
    int to_token[TOKENIZER_MAX_VOCAB];
    unsigned char to_byte[TOKENIZER_MAX_VOCAB];
    int vocab_size;
} tokenizer;

/* Builds the vocabulary from a corpus. Tokens are assigned in ascending byte
 * order rather than by frequency, so the mapping is reproducible from the text
 * alone and a checkpoint does not need the vocabulary stored alongside it. */
void tokenizer_fit(tokenizer *tok, const unsigned char *text, size_t length);

/* Returns the number of tokens written. Bytes absent from the vocabulary are
 * skipped rather than mapped to a placeholder: with a vocabulary built from the
 * same corpus there are none, and inventing an unknown token would put a symbol
 * in the training data that never appears in the text. */
size_t tokenizer_encode(const tokenizer *tok, const unsigned char *text,
                        size_t length, int *out);

void tokenizer_decode(const tokenizer *tok, const int *tokens, size_t count,
                      unsigned char *out);

#endif /* TOKENIZER_H */

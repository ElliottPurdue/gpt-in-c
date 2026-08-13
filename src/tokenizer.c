#include "tokenizer.h"

#include <string.h>

void tokenizer_fit(tokenizer *tok, const unsigned char *text, size_t length)
{
    int seen[TOKENIZER_MAX_VOCAB];
    memset(seen, 0, sizeof(seen));

    for (size_t i = 0; i < length; ++i) {
        seen[text[i]] = 1;
    }

    for (int i = 0; i < TOKENIZER_MAX_VOCAB; ++i) {
        tok->to_token[i] = -1;
    }

    tok->vocab_size = 0;
    for (int byte = 0; byte < TOKENIZER_MAX_VOCAB; ++byte) {
        if (seen[byte]) {
            tok->to_token[byte] = tok->vocab_size;
            tok->to_byte[tok->vocab_size] = (unsigned char)byte;
            tok->vocab_size++;
        }
    }
}

size_t tokenizer_encode(const tokenizer *tok, const unsigned char *text,
                        size_t length, int *out)
{
    size_t written = 0;
    for (size_t i = 0; i < length; ++i) {
        int token = tok->to_token[text[i]];
        if (token >= 0) {
            out[written++] = token;
        }
    }
    return written;
}

void tokenizer_decode(const tokenizer *tok, const int *tokens, size_t count,
                      unsigned char *out)
{
    for (size_t i = 0; i < count; ++i) {
        int t = tokens[i];
        out[i] = (t >= 0 && t < tok->vocab_size) ? tok->to_byte[t] : (unsigned char)'?';
    }
}

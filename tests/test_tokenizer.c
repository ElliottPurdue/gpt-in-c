/* Character tokenizer. */

#include "test.h"
#include "../src/tokenizer.h"

#include <string.h>

static void test_roundtrip_is_lossless(void)
{
    const char *text = "Hello, world!\nThe quick brown fox: 12345.";
    size_t n = strlen(text);

    tokenizer tok;
    tokenizer_fit(&tok, (const unsigned char *)text, n);

    int tokens[128];
    size_t count = tokenizer_encode(&tok, (const unsigned char *)text, n, tokens);
    CHECK(count == n);

    unsigned char back[128];
    tokenizer_decode(&tok, tokens, count, back);
    CHECK(memcmp(back, text, n) == 0);
}

static void test_vocabulary_holds_distinct_bytes_only(void)
{
    const char *text = "aaabbbccc";

    tokenizer tok;
    tokenizer_fit(&tok, (const unsigned char *)text, strlen(text));

    CHECK(tok.vocab_size == 3);
    /* Ascending byte order, not order of first appearance or frequency, so the
     * mapping is reproducible from the text alone. */
    CHECK(tok.to_byte[0] == 'a');
    CHECK(tok.to_byte[1] == 'b');
    CHECK(tok.to_byte[2] == 'c');
}

static void test_token_ids_are_dense(void)
{
    /* Bytes far apart in value must still produce contiguous ids: the embedding
     * table is indexed by token, so a sparse mapping would size it by the
     * largest byte present rather than by the number of distinct ones. */
    const char *text = "\n Az~";

    tokenizer tok;
    tokenizer_fit(&tok, (const unsigned char *)text, strlen(text));

    CHECK(tok.vocab_size == 5);
    for (int i = 0; i < tok.vocab_size; ++i) {
        CHECK(tok.to_token[tok.to_byte[i]] == i);
    }
}

static void test_unseen_bytes_are_skipped(void)
{
    tokenizer tok;
    tokenizer_fit(&tok, (const unsigned char *)"abc", 3);

    int tokens[8];
    size_t count = tokenizer_encode(&tok, (const unsigned char *)"axbxc", 5, tokens);

    CHECK(count == 3);
    CHECK(tokens[0] == 0 && tokens[1] == 1 && tokens[2] == 2);
}

static void test_handles_every_byte_value(void)
{
    unsigned char all[256];
    for (int i = 0; i < 256; ++i) {
        all[i] = (unsigned char)i;
    }

    tokenizer tok;
    tokenizer_fit(&tok, all, 256);
    CHECK(tok.vocab_size == 256);

    int tokens[256];
    size_t count = tokenizer_encode(&tok, all, 256, tokens);
    CHECK(count == 256);

    unsigned char back[256];
    tokenizer_decode(&tok, tokens, count, back);
    CHECK(memcmp(back, all, 256) == 0);
}

void register_tokenizer_tests(void)
{
    RUN(test_roundtrip_is_lossless);
    RUN(test_vocabulary_holds_distinct_bytes_only);
    RUN(test_token_ids_are_dense);
    RUN(test_unseen_bytes_are_skipped);
    RUN(test_handles_every_byte_value);
}

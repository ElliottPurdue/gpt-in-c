/* Trains the model on a text file.
 *
 * Lives outside src/ for the same reason bench/ does in the attitude-estimation
 * project: the library has no file I/O and no stdio, so the same objects compile
 * for a freestanding target. Everything that touches a file is here.
 *
 * The validation split is the point of the whole program. A language model can
 * drive its training loss arbitrarily low by memorising, and a character model
 * on a small corpus will do exactly that, so training loss alone says nothing
 * about whether it learned the language. The split is contiguous rather than
 * random: sampling held-out windows at random from the same text leaves them
 * overlapping their training neighbours by up to block_size - 1 characters,
 * which leaks the answer and produces a validation curve that tracks the
 * training curve no matter how badly the model overfits.
 */

#include "src/model.h"
#include "src/optim.h"
#include "src/tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double seconds_now(void)
{
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

/* Checkpoint format, little-endian:
 *
 *     magic "GPTC", uint32 version
 *     5 x int32 config (vocab, block, layer, head, embd)
 *     uint32 vocab byte table length, then that many bytes
 *     float32 parameters, in the library's own flat layout
 *
 * The parameters are one contiguous block by construction, so writing them is a
 * single fwrite and no serialiser has to know the model's structure. The config
 * is stored ahead of them and checked on load: a checkpoint from a different
 * shape would otherwise be read as garbage of the right length.
 *
 * The vocabulary travels with the weights. Token ids mean nothing without the
 * table that produced them, and a checkpoint loaded against a different corpus
 * would generate confident nonsense rather than fail.
 *
 * File I/O lives here rather than in src/ so the library keeps no stdio
 * dependency; on a microcontroller the same bytes would be linked in as an
 * array instead.
 */
#define CHECKPOINT_MAGIC "GPTC"
#define CHECKPOINT_VERSION 1u

static int save_checkpoint(const char *path, const gpt_model *model,
                           const tokenizer *tok)
{
    FILE *f = fopen(path, "wb");
    if (!f) {
        return 0;
    }

    int header[5] = { model->config.vocab_size, model->config.block_size,
                      model->config.n_layer, model->config.n_head,
                      model->config.n_embd };
    unsigned version = CHECKPOINT_VERSION;
    unsigned vocab = (unsigned)tok->vocab_size;

    int ok = fwrite(CHECKPOINT_MAGIC, 1, 4, f) == 4
          && fwrite(&version, sizeof(version), 1, f) == 1
          && fwrite(header, sizeof(int), 5, f) == 5
          && fwrite(&vocab, sizeof(vocab), 1, f) == 1
          && fwrite(tok->to_byte, 1, vocab, f) == vocab
          && fwrite(model->params.storage, sizeof(float), model->params.count, f)
             == model->params.count;

    fclose(f);
    return ok;
}

static int load_checkpoint(const char *path, gpt_model *model, tokenizer *tok)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return 0;
    }

    char magic[4];
    unsigned version = 0, vocab = 0;
    int header[5] = { 0, 0, 0, 0, 0 };

    int ok = fread(magic, 1, 4, f) == 4
          && memcmp(magic, CHECKPOINT_MAGIC, 4) == 0
          && fread(&version, sizeof(version), 1, f) == 1
          && version == CHECKPOINT_VERSION
          && fread(header, sizeof(int), 5, f) == 5;

    if (ok) {
        gpt_config c = model->config;
        ok = header[0] == c.vocab_size && header[1] == c.block_size
          && header[2] == c.n_layer && header[3] == c.n_head
          && header[4] == c.n_embd;
        if (!ok) {
            fprintf(stderr,
                    "checkpoint is for a %d-layer, width-%d model; "
                    "this one is %d-layer, width-%d\n",
                    header[2], header[4], c.n_layer, c.n_embd);
        }
    }

    if (ok) {
        ok = fread(&vocab, sizeof(vocab), 1, f) == 1
          && vocab <= TOKENIZER_MAX_VOCAB
          && fread(tok->to_byte, 1, vocab, f) == vocab;
    }
    if (ok) {
        /* Rebuild the reverse map rather than storing it: it is derivable, and
         * a stored copy is one more thing that can disagree with itself. */
        tok->vocab_size = (int)vocab;
        for (int i = 0; i < TOKENIZER_MAX_VOCAB; ++i) {
            tok->to_token[i] = -1;
        }
        for (int i = 0; i < tok->vocab_size; ++i) {
            tok->to_token[tok->to_byte[i]] = i;
        }
        ok = fread(model->params.storage, sizeof(float), model->params.count, f)
             == model->params.count;
    }

    fclose(f);
    return ok;
}

static unsigned char *read_file(const char *path, size_t *length)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return NULL;
    }

    unsigned char *buffer = (unsigned char *)malloc((size_t)size);
    if (!buffer || fread(buffer, 1, (size_t)size, f) != (size_t)size) {
        free(buffer);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *length = (size_t)size;
    return buffer;
}

/* Draws a batch of (input, target) windows. The target is the input shifted one
 * position, so every position in the sequence supplies a prediction rather than
 * only the last -- which is what makes a single forward pass worth B*T training
 * signals instead of B. */
static void sample_batch(const int *tokens, size_t count, int B, int T,
                         int *idx, int *targets, unsigned *rng)
{
    for (int b = 0; b < B; ++b) {
        *rng = *rng * 1103515245u + 12345u;
        size_t start = (size_t)((*rng >> 8) % (count - (size_t)T - 1));

        for (int t = 0; t < T; ++t) {
            idx[b * T + t] = tokens[start + (size_t)t];
            targets[b * T + t] = tokens[start + (size_t)t + 1];
        }
    }
}

/* Mean loss over a fixed set of batches, drawn from a fixed seed so the number
 * is comparable across evaluations rather than moving with the sample. */
static float evaluate(gpt_model *model, const int *tokens, size_t count,
                      int B, int T, int batches, int *idx, int *targets)
{
    unsigned rng = 12345u;
    float total = 0.0f;

    for (int i = 0; i < batches; ++i) {
        sample_batch(tokens, count, B, T, idx, targets, &rng);
        total += gpt_forward(model, idx, targets, T);
    }
    return total / (float)batches;
}

static void print_sample(gpt_model *model, const tokenizer *tok,
                         const int *tokens, int T, unsigned *rng)
{
    int prompt[1] = { tokens[0] };
    int *generated = (int *)malloc((size_t)(T + 200) * sizeof(int));
    unsigned char *text = (unsigned char *)malloc((size_t)(T + 200) + 1);

    int n = gpt_generate(model, prompt, 1, generated, 200, 0.8f, rng);
    tokenizer_decode(tok, generated, (size_t)n, text);
    text[n] = '\0';

    printf("  ------------------------------------------------------------\n");
    for (int i = 0; i < n; ++i) {
        putchar(text[i] == '\n' ? ' ' : text[i]);
    }
    printf("\n  ------------------------------------------------------------\n");

    free(generated);
    free(text);
}

int main(int argc, char **argv)
{
    const char *path = (argc > 1) ? argv[1] : "data/input.txt";
    int steps = (argc > 2) ? atoi(argv[2]) : 500;
    const char *checkpoint = (argc > 3) ? argv[3] : NULL;
    const char *resume = (argc > 4) ? argv[4] : NULL;

    size_t file_length = 0;
    unsigned char *text = read_file(path, &file_length);
    if (!text) {
        fprintf(stderr, "cannot read %s\n", path);
        fprintf(stderr, "  run: make data/input.txt\n");
        return 1;
    }

    tokenizer tok;
    tokenizer_fit(&tok, text, file_length);

    int *tokens = (int *)malloc(file_length * sizeof(int));
    size_t token_count = tokenizer_encode(&tok, text, file_length, tokens);
    free(text);

    /* Contiguous split, last 10% held out. */
    size_t train_count = token_count * 9 / 10;
    size_t val_count = token_count - train_count;
    const int *val_tokens = tokens + train_count;

    gpt_config config;
    config.vocab_size = tok.vocab_size;
    config.block_size = 64;
    config.n_layer = 3;
    config.n_head = 4;
    config.n_embd = 96;

    const int B = 16, T = config.block_size;

    gpt_model model;
    if (!gpt_init(&model, config, B)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    gpt_randomize(&model, 1337u);

    adamw opt;
    if (!adamw_init(&opt, model.params.count)) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    opt.lr = 1e-3f;

    int *idx = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));

    printf("\ngpt-in-c\n");
    printf("  corpus      %s, %zu bytes, %d distinct characters\n",
           path, file_length, tok.vocab_size);
    printf("  split       %zu train / %zu validation tokens\n",
           train_count, val_count);
    printf("  model       %d layers, %d heads, width %d, context %d\n",
           config.n_layer, config.n_head, config.n_embd, config.block_size);
    printf("  parameters  %zu\n", model.params.count);
    printf("  batch       %d x %d = %d tokens per step\n", B, T, B * T);
    printf("  steps       %d\n\n", steps);

    /* Before training, the model should be at the entropy of a uniform
     * distribution over the vocabulary: -log(1/V). Printing it makes the first
     * loss meaningful rather than an unanchored number, and a starting loss far
     * from it means the initialisation is wrong. */
    printf("  expected initial loss  %.4f  (ln %d)\n\n",
           (double)logf((float)tok.vocab_size), tok.vocab_size);

    if (resume) {
        if (!load_checkpoint(resume, &model, &tok)) {
            fprintf(stderr, "could not load checkpoint %s\n", resume);
            return 1;
        }
        printf("  resumed from %s\n", resume);
    }

    unsigned rng = 42u;
    double start_time = seconds_now();
    double train_seconds = 0.0;
    long tokens_seen = 0;

    for (int step = 1; step <= steps; ++step) {
        double step_start = seconds_now();

        sample_batch(tokens, train_count, B, T, idx, targets, &rng);

        float loss = gpt_forward(&model, idx, targets, T);
        gpt_zero_grad(&model);
        gpt_backward(&model, idx, targets, T);

        float norm = clip_grad_norm(model.grads.storage, model.grads.count, 1.0f);
        adamw_step(&opt, model.params.storage, model.grads.storage);

        train_seconds += seconds_now() - step_start;
        tokens_seen += B * T;

        if (step == 1 || step % 50 == 0 || step == steps) {
            float val = evaluate(&model, val_tokens, val_count, B, T, 10,
                                 idx, targets);
            printf("  step %5d   train %.4f   val %.4f   |grad| %.3f   "
                   "%.0f tok/s\n",
                   step, (double)loss, (double)val, (double)norm,
                   (double)tokens_seen / train_seconds);
        }
    }

    printf("\n  %d steps in %.1f s, %.0f tokens/s\n",
           steps, seconds_now() - start_time,
           (double)tokens_seen / train_seconds);

    if (checkpoint) {
        if (save_checkpoint(checkpoint, &model, &tok)) {
            printf("\n  wrote %s (%zu parameters)\n",
                   checkpoint, model.params.count);
        } else {
            fprintf(stderr, "\n  could not write %s\n", checkpoint);
        }
    }

    printf("\n  sample at temperature 0.8:\n");
    print_sample(&model, &tok, tokens, T, &rng);

    free(idx); free(targets); free(tokens);
    adamw_free(&opt);
    gpt_free(&model);
    return 0;
}

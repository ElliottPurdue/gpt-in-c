/* The assembled GPT: embeddings, a stack of pre-norm blocks, and a language
 * modelling head.
 *
 * MEMORY. Parameters live in one allocation and activations in another, with
 * the named fields pointing into them. Three reasons, in order of how much they
 * matter:
 *
 *   1. Zeroing gradients is one memset over a contiguous block rather than a
 *      walk over a dozen buffers. Since every backward function accumulates,
 *      failing to zero one of them produces a slow drift rather than a crash,
 *      and the only reliable defence is making it impossible to miss one.
 *   2. An optimiser sees a flat float array. AdamW becomes a single loop over
 *      `count` elements with no knowledge of the model's structure.
 *   3. Checkpointing is an fwrite of one block.
 *
 * The same struct describes parameters and their gradients, because they have
 * identical shapes. `gpt_params grads` reads better than a parallel set of
 * `d`-prefixed fields, and it makes the symmetry between the two passes visible.
 *
 * ACTIVATIONS are kept for the whole forward pass rather than recomputed. This
 * is the standard time/memory trade and it is the right one here: the backward
 * pass needs almost every intermediate, and recomputing them would roughly
 * double the cost of a step to save memory this model does not need.
 */

#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

typedef struct {
    int vocab_size;
    int block_size;     /* maximum context length */
    int n_layer;
    int n_head;
    int n_embd;
} gpt_config;

/* Parameters, or equally their gradients. Per-layer fields are laid out with
 * layer index outermost, so layer l's LayerNorm weight starts at
 * ln1_w + l * n_embd. */
typedef struct {
    float *wte;         /* (vocab_size, n_embd)  token embedding */
    float *wpe;         /* (block_size, n_embd)  position embedding */

    float *ln1_w, *ln1_b;       /* (L, C) */
    float *qkv_w;               /* (L, 3C, C) */
    float *qkv_b;               /* (L, 3C) */
    float *proj_w, *proj_b;     /* (L, C, C) and (L, C) */
    float *ln2_w, *ln2_b;       /* (L, C) */
    float *fc_w, *fc_b;         /* (L, 4C, C) and (L, 4C) */
    float *fcproj_w, *fcproj_b; /* (L, C, 4C) and (L, C) */

    float *lnf_w, *lnf_b;       /* (C) */
    float *lm_head;             /* (vocab_size, n_embd), no bias */

    float *storage;             /* the single allocation behind all of it */
    size_t count;               /* total floats, for the optimiser */
} gpt_params;

/* Saved intermediates. Named for what produced them, in forward order. */
typedef struct {
    float *embedded;        /* (B, T, C)      wte[idx] + wpe[pos] */

    float *ln1;             /* (L, B, T, C) */
    float *ln1_mean, *ln1_rstd;     /* (L, B, T) */
    float *qkv;             /* (L, B, T, 3C) */
    float *att;             /* (L, B, n_head, T, T) */
    float *attn_out;        /* (L, B, T, C)  heads merged, before the projection */
    float *attn_proj;       /* (L, B, T, C) */
    float *residual1;       /* (L, B, T, C) */

    float *ln2;             /* (L, B, T, C) */
    float *ln2_mean, *ln2_rstd;
    float *fc;              /* (L, B, T, 4C) */
    float *gelu;            /* (L, B, T, 4C) */
    float *mlp_out;         /* (L, B, T, C) */
    float *residual2;       /* (L, B, T, C)  the block's output */

    float *lnf;             /* (B, T, C) */
    float *lnf_mean, *lnf_rstd;
    float *logits;          /* (B, T, vocab_size) */
    float *probs;           /* (B, T, vocab_size) */

    float *storage;
    size_t count;
    int batch;              /* the B these were sized for */
} gpt_activations;

/* Scratch for the backward pass: one buffer per activation shape that needs a
 * gradient, plus the two attention temporaries. Allocated once and reused, and
 * zeroed at the start of every backward pass. */
typedef struct {
    float *storage;
    size_t count;
    int batch;
} gpt_grad_workspace;

typedef struct {
    gpt_config config;
    gpt_params params;
    gpt_params grads;
    gpt_activations acts;
    gpt_grad_workspace workspace;
} gpt_model;

/* Returns 0 on allocation failure, leaving nothing allocated. */
int gpt_init(gpt_model *model, gpt_config config, int batch);
void gpt_free(gpt_model *model);

size_t gpt_parameter_count(gpt_config config);

/* Fills parameters with a deterministic pseudo-random init. Kept in the library
 * rather than left to the caller so that a from-scratch run is reproducible
 * without a source of randomness. */
void gpt_randomize(gpt_model *model, unsigned seed);

/* Forward pass. targets may be NULL, in which case no loss is computed and the
 * return value is 0; logits are still written. idx and targets are (B, T). */
float gpt_forward(gpt_model *model, const int *idx, const int *targets, int T);

/* Forward over fewer rows than the model was allocated for. The activation
 * buffers are sized by the maximum batch, and every stride is computed from the
 * batch actually passed, so a smaller one simply uses less of them. Generation
 * needs this: it runs one sequence at a time. */
float gpt_forward_batch(gpt_model *model, const int *idx, const int *targets,
                        int B, int T);

/* Backward pass. Must follow a gpt_forward with the same inputs, whose saved
 * activations it reads. Gradients accumulate into model->grads, so call
 * gpt_zero_grad first unless deliberately accumulating across microbatches. */
void gpt_backward(gpt_model *model, const int *idx, const int *targets, int T);

void gpt_zero_grad(gpt_model *model);

/* Autoregressive sampling. Writes prompt_len + max_new tokens into out and
 * returns the count.
 *
 * Context is truncated to the last block_size tokens, because the position
 * embedding has no entry beyond that -- the model cannot represent a longer
 * sequence, and indexing past the table would read whatever follows it.
 *
 * temperature divides the logits before the softmax: below 1 it sharpens the
 * distribution toward the argmax, above 1 flattens it. At exactly 0 it would
 * divide by zero, so that case takes the argmax directly. */
int gpt_generate(gpt_model *model, const int *prompt, int prompt_len,
                 int *out, int max_new, float temperature, unsigned *rng);

#endif /* MODEL_H */

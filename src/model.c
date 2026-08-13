#include "model.h"
#include "ops.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Walks the parameter layout once. Called twice: with base == NULL to total the
 * sizes, and again with a real base to assign the pointers. Keeping both uses in
 * one function is what guarantees the count and the layout cannot disagree --
 * two separate versions would drift the first time a parameter was added. */
static size_t map_params(gpt_params *p, gpt_config c, float *base)
{
    size_t offset = 0;
    int C = c.n_embd, L = c.n_layer;

#define FIELD(name, n) do {                                                   \
        if (p) { p->name = base ? base + offset : NULL; }                     \
        offset += (size_t)(n);                                                \
    } while (0)

    FIELD(wte, (size_t)c.vocab_size * C);
    FIELD(wpe, (size_t)c.block_size * C);

    FIELD(ln1_w, (size_t)L * C);
    FIELD(ln1_b, (size_t)L * C);
    FIELD(qkv_w, (size_t)L * 3 * C * C);
    FIELD(qkv_b, (size_t)L * 3 * C);
    FIELD(proj_w, (size_t)L * C * C);
    FIELD(proj_b, (size_t)L * C);
    FIELD(ln2_w, (size_t)L * C);
    FIELD(ln2_b, (size_t)L * C);
    FIELD(fc_w, (size_t)L * 4 * C * C);
    FIELD(fc_b, (size_t)L * 4 * C);
    FIELD(fcproj_w, (size_t)L * C * 4 * C);
    FIELD(fcproj_b, (size_t)L * C);

    FIELD(lnf_w, (size_t)C);
    FIELD(lnf_b, (size_t)C);
    FIELD(lm_head, (size_t)c.vocab_size * C);

#undef FIELD
    return offset;
}

static size_t map_acts(gpt_activations *a, gpt_config c, int B, float *base)
{
    size_t offset = 0;
    int C = c.n_embd, L = c.n_layer, T = c.block_size, H = c.n_head;
    size_t BTC = (size_t)B * T * C;
    size_t BT = (size_t)B * T;

#define FIELD(name, n) do {                                                   \
        if (a) { a->name = base ? base + offset : NULL; }                     \
        offset += (size_t)(n);                                                \
    } while (0)

    FIELD(embedded, BTC);

    FIELD(ln1, (size_t)L * BTC);
    FIELD(ln1_mean, (size_t)L * BT);
    FIELD(ln1_rstd, (size_t)L * BT);
    FIELD(qkv, (size_t)L * B * T * 3 * C);
    FIELD(att, (size_t)L * B * H * T * T);
    FIELD(attn_out, (size_t)L * BTC);
    FIELD(attn_proj, (size_t)L * BTC);
    FIELD(residual1, (size_t)L * BTC);

    FIELD(ln2, (size_t)L * BTC);
    FIELD(ln2_mean, (size_t)L * BT);
    FIELD(ln2_rstd, (size_t)L * BT);
    FIELD(fc, (size_t)L * B * T * 4 * C);
    FIELD(gelu, (size_t)L * B * T * 4 * C);
    FIELD(mlp_out, (size_t)L * BTC);
    FIELD(residual2, (size_t)L * BTC);

    FIELD(lnf, BTC);
    FIELD(lnf_mean, BT);
    FIELD(lnf_rstd, BT);
    FIELD(logits, (size_t)B * T * c.vocab_size);
    FIELD(probs, (size_t)B * T * c.vocab_size);

#undef FIELD
    return offset;
}

size_t gpt_parameter_count(gpt_config config)
{
    return map_params(NULL, config, NULL);
}

/* The backward pass needs a gradient buffer for most activation shapes. Rather
 * than mirroring the activation struct, it takes slices off one workspace as it
 * goes -- the gradients flow layer by layer and only a few are live at once. */
static size_t workspace_size(gpt_config c, int B)
{
    size_t C = (size_t)c.n_embd, T = (size_t)c.block_size;
    size_t BTC = (size_t)B * T * C;
    size_t largest_logits = (size_t)B * T * c.vocab_size;

    /* Ten (B,T,C) buffers for the residual stream and block internals, one
     * (B,T,3C) for qkv, two (B,T,4C) for the MLP, one attention map, and the
     * logits. Sized generously and asserted by use rather than tuned. */
    return 10 * BTC + (size_t)B * T * 3 * C + 2 * (size_t)B * T * 4 * C
         + (size_t)B * c.n_head * T * T + largest_logits;
}

int gpt_init(gpt_model *model, gpt_config config, int batch)
{
    memset(model, 0, sizeof(*model));
    model->config = config;

    size_t n_params = map_params(NULL, config, NULL);
    size_t n_acts = map_acts(NULL, config, batch, NULL);
    size_t n_work = workspace_size(config, batch);

    model->params.storage = (float *)calloc(n_params, sizeof(float));
    model->grads.storage = (float *)calloc(n_params, sizeof(float));
    model->acts.storage = (float *)calloc(n_acts, sizeof(float));
    model->workspace.storage = (float *)calloc(n_work, sizeof(float));

    if (!model->params.storage || !model->grads.storage ||
        !model->acts.storage || !model->workspace.storage) {
        gpt_free(model);
        return 0;
    }

    map_params(&model->params, config, model->params.storage);
    map_params(&model->grads, config, model->grads.storage);
    map_acts(&model->acts, config, batch, model->acts.storage);

    model->params.count = n_params;
    model->grads.count = n_params;
    model->acts.count = n_acts;
    model->acts.batch = batch;
    model->workspace.count = n_work;
    model->workspace.batch = batch;
    return 1;
}

void gpt_free(gpt_model *model)
{
    free(model->params.storage);
    free(model->grads.storage);
    free(model->acts.storage);
    free(model->workspace.storage);
    memset(model, 0, sizeof(*model));
}

void gpt_zero_grad(gpt_model *model)
{
    memset(model->grads.storage, 0, model->grads.count * sizeof(float));
}

/* xorshift, so initialisation is reproducible without depending on the host's
 * rand() or on a seeded global. */
static unsigned next_random(unsigned *state)
{
    unsigned x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float normal(unsigned *state)
{
    /* Box-Muller. Both uniforms are pushed strictly inside (0, 1): u1 at exactly
     * zero makes log() infinite, and a value at or above 1 makes it positive,
     * which puts a negative argument into sqrt and yields NaN. */
    float u1 = ((float)(next_random(state) >> 8) + 0.5f) / 16777216.0f;
    float u2 = ((float)(next_random(state) >> 8) + 0.5f) / 16777216.0f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

void gpt_randomize(gpt_model *model, unsigned seed)
{
    unsigned state = seed ? seed : 1u;
    gpt_config c = model->config;
    gpt_params *p = &model->params;

    memset(p->storage, 0, p->count * sizeof(float));

    /* 0.02 is the GPT-2 initialisation. The scale matters more than it looks:
     * too large and the first softmax saturates, too small and the residual
     * stream carries no signal for the early layers to work with. */
    const float sigma = 0.02f;
    size_t weights[] = {
        (size_t)c.vocab_size * c.n_embd,        /* wte */
        (size_t)c.block_size * c.n_embd,        /* wpe */
    };
    float *targets[] = { p->wte, p->wpe };
    for (int i = 0; i < 2; ++i) {
        for (size_t j = 0; j < weights[i]; ++j) {
            targets[i][j] = sigma * normal(&state);
        }
    }

    int C = c.n_embd, L = c.n_layer;
    for (int l = 0; l < L; ++l) {
        /* LayerNorm starts as the identity: unit gain, zero shift. */
        for (int i = 0; i < C; ++i) {
            p->ln1_w[l * C + i] = 1.0f;
            p->ln2_w[l * C + i] = 1.0f;
        }
        for (size_t i = 0; i < (size_t)3 * C * C; ++i) {
            p->qkv_w[(size_t)l * 3 * C * C + i] = sigma * normal(&state);
        }
        for (size_t i = 0; i < (size_t)C * C; ++i) {
            p->proj_w[(size_t)l * C * C + i] = sigma * normal(&state);
        }
        for (size_t i = 0; i < (size_t)4 * C * C; ++i) {
            p->fc_w[(size_t)l * 4 * C * C + i] = sigma * normal(&state);
            p->fcproj_w[(size_t)l * 4 * C * C + i] = sigma * normal(&state);
        }
    }
    for (int i = 0; i < C; ++i) {
        p->lnf_w[i] = 1.0f;
    }
    for (size_t i = 0; i < (size_t)c.vocab_size * C; ++i) {
        p->lm_head[i] = sigma * normal(&state);
    }
}

/* ------------------------------------------------------------------------
 * Forward
 * ------------------------------------------------------------------------ */

float gpt_forward(gpt_model *model, const int *idx, const int *targets, int T)
{
    return gpt_forward_batch(model, idx, targets, model->acts.batch, T);
}

float gpt_forward_batch(gpt_model *model, const int *idx, const int *targets,
                        int B, int T)
{
    gpt_config c = model->config;
    gpt_params *p = &model->params;
    gpt_activations *a = &model->acts;

    int C = c.n_embd, L = c.n_layer, H = c.n_head, V = c.vocab_size;
    size_t BTC = (size_t)B * T * C;
    size_t BT = (size_t)B * T;

    /* Token plus position. The position embedding is indexed by offset within
     * the sequence, not by token identity, which is the only thing giving the
     * model any notion of order -- attention itself is permutation invariant. */
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *token = p->wte + (size_t)idx[b * T + t] * C;
            const float *position = p->wpe + (size_t)t * C;
            float *out = a->embedded + ((size_t)b * T + t) * C;
            for (int i = 0; i < C; ++i) {
                out[i] = token[i] + position[i];
            }
        }
    }

    const float *residual = a->embedded;

    for (int l = 0; l < L; ++l) {
        float *ln1 = a->ln1 + (size_t)l * BTC;
        float *ln1_mean = a->ln1_mean + (size_t)l * BT;
        float *ln1_rstd = a->ln1_rstd + (size_t)l * BT;
        float *qkv = a->qkv + (size_t)l * B * T * 3 * C;
        float *att = a->att + (size_t)l * B * H * T * T;
        float *attn_out = a->attn_out + (size_t)l * BTC;
        float *attn_proj = a->attn_proj + (size_t)l * BTC;
        float *residual1 = a->residual1 + (size_t)l * BTC;
        float *ln2 = a->ln2 + (size_t)l * BTC;
        float *ln2_mean = a->ln2_mean + (size_t)l * BT;
        float *ln2_rstd = a->ln2_rstd + (size_t)l * BT;
        float *fc = a->fc + (size_t)l * B * T * 4 * C;
        float *gelu = a->gelu + (size_t)l * B * T * 4 * C;
        float *mlp_out = a->mlp_out + (size_t)l * BTC;
        float *residual2 = a->residual2 + (size_t)l * BTC;

        /* Pre-norm: normalise going into the sublayer, and add the raw residual
         * afterwards. Post-norm normalises the sum instead, which puts a
         * LayerNorm on the residual path and makes deep stacks need a warmup to
         * train at all. */
        layernorm_forward(ln1, ln1_mean, ln1_rstd, residual,
                          p->ln1_w + (size_t)l * C, p->ln1_b + (size_t)l * C,
                          B * T, C, 1e-5f);

        attention_forward(attn_out, qkv, att, ln1,
                          p->qkv_w + (size_t)l * 3 * C * C,
                          p->qkv_b + (size_t)l * 3 * C, B, T, C, H);

        linear_forward(attn_proj, attn_out, p->proj_w + (size_t)l * C * C,
                       p->proj_b + (size_t)l * C, B * T, C, C);

        for (size_t i = 0; i < BTC; ++i) {
            residual1[i] = residual[i] + attn_proj[i];
        }

        layernorm_forward(ln2, ln2_mean, ln2_rstd, residual1,
                          p->ln2_w + (size_t)l * C, p->ln2_b + (size_t)l * C,
                          B * T, C, 1e-5f);

        linear_forward(fc, ln2, p->fc_w + (size_t)l * 4 * C * C,
                       p->fc_b + (size_t)l * 4 * C, B * T, C, 4 * C);
        gelu_forward(gelu, fc, (int)((size_t)B * T * 4 * C));
        linear_forward(mlp_out, gelu, p->fcproj_w + (size_t)l * C * 4 * C,
                       p->fcproj_b + (size_t)l * C, B * T, 4 * C, C);

        for (size_t i = 0; i < BTC; ++i) {
            residual2[i] = residual1[i] + mlp_out[i];
        }

        residual = residual2;
    }

    layernorm_forward(a->lnf, a->lnf_mean, a->lnf_rstd, residual,
                      p->lnf_w, p->lnf_b, B * T, C, 1e-5f);

    linear_forward(a->logits, a->lnf, p->lm_head, NULL, B * T, C, V);

    if (!targets) {
        return 0.0f;
    }
    return crossentropy_forward(a->probs, a->logits, targets, B * T, V);
}

/* ------------------------------------------------------------------------
 * Backward
 * ------------------------------------------------------------------------ */

void gpt_backward(gpt_model *model, const int *idx, const int *targets, int T)
{
    gpt_config c = model->config;
    gpt_params *p = &model->params;
    gpt_params *g = &model->grads;
    gpt_activations *a = &model->acts;

    int B = a->batch, C = c.n_embd, L = c.n_layer, H = c.n_head, V = c.vocab_size;
    size_t BTC = (size_t)B * T * C;
    size_t BT4C = (size_t)B * T * 4 * C;

    /* Slice the workspace. The gradients flow one layer at a time, so these are
     * reused across layers rather than being kept per layer. */
    float *cursor = model->workspace.storage;
    float *dresidual = cursor;   cursor += BTC;   /* into the block */
    float *dresidual1 = cursor;  cursor += BTC;
    float *dresidual2 = cursor;  cursor += BTC;
    float *dln1 = cursor;        cursor += BTC;
    float *dln2 = cursor;        cursor += BTC;
    float *dattn_out = cursor;   cursor += BTC;
    float *dattn_proj = cursor;  cursor += BTC;
    float *dmlp_out = cursor;    cursor += BTC;
    float *dlnf = cursor;        cursor += BTC;
    float *dembedded = cursor;   cursor += BTC;
    float *dqkv = cursor;        cursor += (size_t)B * T * 3 * C;
    float *dfc = cursor;         cursor += BT4C;
    float *dgelu = cursor;       cursor += BT4C;
    float *datt = cursor;        cursor += (size_t)B * H * T * T;
    float *dlogits = cursor;

    memset(model->workspace.storage, 0,
           model->workspace.count * sizeof(float));

    crossentropy_backward(dlogits, a->probs, targets, B * T, V);

    /* lm_head has no bias, hence the NULL. */
    linear_backward(dlnf, g->lm_head, NULL, dlogits, a->lnf, p->lm_head,
                    B * T, C, V);

    const float *last = (L > 0) ? a->residual2 + (size_t)(L - 1) * BTC
                                : a->embedded;
    layernorm_backward(dresidual, g->lnf_w, g->lnf_b, dlnf, last,
                       a->lnf_mean, a->lnf_rstd, p->lnf_w, B * T, C);

    for (int l = L - 1; l >= 0; --l) {
        const float *ln1 = a->ln1 + (size_t)l * BTC;
        const float *qkv = a->qkv + (size_t)l * B * T * 3 * C;
        const float *att = a->att + (size_t)l * B * H * T * T;
        const float *attn_out = a->attn_out + (size_t)l * BTC;
        const float *residual1 = a->residual1 + (size_t)l * BTC;
        const float *ln2 = a->ln2 + (size_t)l * BTC;
        const float *gelu = a->gelu + (size_t)l * B * T * 4 * C;
        const float *fc = a->fc + (size_t)l * B * T * 4 * C;
        const float *block_input = (l > 0) ? a->residual2 + (size_t)(l - 1) * BTC
                                           : a->embedded;

        /* dresidual holds the gradient arriving at this block's output. The
         * residual connection is an addition, so it forwards that gradient
         * unchanged down both paths -- which is exactly why residual networks
         * train: the identity path carries gradient to every depth without
         * being attenuated by any weight matrix. */
        memcpy(dresidual2, dresidual, BTC * sizeof(float));
        memcpy(dmlp_out, dresidual, BTC * sizeof(float));

        memset(dgelu, 0, BT4C * sizeof(float));
        linear_backward(dgelu, g->fcproj_w + (size_t)l * C * 4 * C,
                        g->fcproj_b + (size_t)l * C, dmlp_out, gelu,
                        p->fcproj_w + (size_t)l * C * 4 * C,
                        B * T, 4 * C, C);

        memset(dfc, 0, BT4C * sizeof(float));
        gelu_backward(dfc, dgelu, fc, (int)BT4C);

        memset(dln2, 0, BTC * sizeof(float));
        linear_backward(dln2, g->fc_w + (size_t)l * 4 * C * C,
                        g->fc_b + (size_t)l * 4 * C, dfc, ln2,
                        p->fc_w + (size_t)l * 4 * C * C, B * T, C, 4 * C);

        /* The residual's other branch: dresidual2 is the gradient that skipped
         * the MLP entirely, and the LayerNorm's contribution adds to it. */
        memcpy(dresidual1, dresidual2, BTC * sizeof(float));
        layernorm_backward(dresidual1, g->ln2_w + (size_t)l * C,
                           g->ln2_b + (size_t)l * C, dln2, residual1,
                           a->ln2_mean + (size_t)l * B * T,
                           a->ln2_rstd + (size_t)l * B * T,
                           p->ln2_w + (size_t)l * C, B * T, C);

        memcpy(dattn_proj, dresidual1, BTC * sizeof(float));

        memset(dattn_out, 0, BTC * sizeof(float));
        linear_backward(dattn_out, g->proj_w + (size_t)l * C * C,
                        g->proj_b + (size_t)l * C, dattn_proj, attn_out,
                        p->proj_w + (size_t)l * C * C, B * T, C, C);

        memset(dln1, 0, BTC * sizeof(float));
        memset(dqkv, 0, (size_t)B * T * 3 * C * sizeof(float));
        memset(datt, 0, (size_t)B * H * T * T * sizeof(float));
        attention_backward(dln1, g->qkv_w + (size_t)l * 3 * C * C,
                           g->qkv_b + (size_t)l * 3 * C, dqkv, datt,
                           dattn_out, ln1, p->qkv_w + (size_t)l * 3 * C * C,
                           qkv, att, B, T, C, H);

        /* And the attention branch's residual, feeding the next iteration. */
        memcpy(dresidual, dresidual1, BTC * sizeof(float));
        layernorm_backward(dresidual, g->ln1_w + (size_t)l * C,
                           g->ln1_b + (size_t)l * C, dln1, block_input,
                           a->ln1_mean + (size_t)l * B * T,
                           a->ln1_rstd + (size_t)l * B * T,
                           p->ln1_w + (size_t)l * C, B * T, C);
    }

    memcpy(dembedded, dresidual, BTC * sizeof(float));

    /* Embeddings are a gather forward, so their gradient is a scatter-add. Every
     * occurrence of a token accumulates into the same row, which is the clearest
     * case for why these functions add rather than assign. */
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            const float *d = dembedded + ((size_t)b * T + t) * C;
            float *dtoken = g->wte + (size_t)idx[b * T + t] * C;
            float *dposition = g->wpe + (size_t)t * C;
            for (int i = 0; i < C; ++i) {
                dtoken[i] += d[i];
                dposition[i] += d[i];
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * Sampling
 * ------------------------------------------------------------------------ */

int gpt_generate(gpt_model *model, const int *prompt, int prompt_len,
                 int *out, int max_new, float temperature, unsigned *rng)
{
    gpt_config c = model->config;
    int V = c.vocab_size, T_max = c.block_size;

    int count = prompt_len < T_max ? prompt_len : T_max;
    for (int i = 0; i < count; ++i) {
        out[i] = prompt[i];
    }

    for (int step = 0; step < max_new; ++step) {
        /* Only the trailing block_size tokens are visible. This is a real
         * limitation of the model rather than an implementation shortcut: the
         * position embedding table has exactly block_size rows. */
        int start = count > T_max ? count - T_max : 0;
        int window = count - start;

        gpt_forward_batch(model, out + start, NULL, 1, window);

        /* Only the last position predicts the next token; the earlier ones
         * predict tokens already present. */
        const float *logits = model->acts.logits + (size_t)(window - 1) * V;

        int next = 0;
        if (temperature <= 0.0f) {
            for (int v = 1; v < V; ++v) {
                if (logits[v] > logits[next]) {
                    next = v;
                }
            }
        } else {
            /* Softmax over the scaled logits, with the usual max subtraction,
             * then an inverse-CDF draw. */
            float maximum = logits[0];
            for (int v = 1; v < V; ++v) {
                if (logits[v] > maximum) {
                    maximum = logits[v];
                }
            }

            float sum = 0.0f;
            for (int v = 0; v < V; ++v) {
                sum += expf((logits[v] - maximum) / temperature);
            }

            *rng = *rng * 1103515245u + 12345u;
            float target = ((float)((*rng >> 16) & 0x7fff) / 32768.0f) * sum;

            float running = 0.0f;
            next = V - 1;   /* fallback if rounding leaves target just above sum */
            for (int v = 0; v < V; ++v) {
                running += expf((logits[v] - maximum) / temperature);
                if (running >= target) {
                    next = v;
                    break;
                }
            }
        }

        out[count++] = next;
    }
    return count;
}

/* The assembled model against PyTorch, and against the definition of a
 * derivative.
 *
 * Two independent checks, which is the point. The oracle comparison is precise
 * but shares an assumption with the thing it checks: both implement the same
 * architecture from the same description, so a misreading of the architecture
 * would be reproduced on both sides and agree perfectly. The finite-difference
 * check knows nothing about transformers -- it only knows that a gradient is
 * the limit of a difference quotient -- so it cannot share that error.
 */

#include "test.h"
#include "oracle.h"
#include "../src/model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ABS_TOL 2e-5f
#define REL_TOL 2e-4f

static oracle full;

/* The oracle's config, matching ref/reference.py's DEFAULT_CONFIG. */
static gpt_config oracle_config(void)
{
    gpt_config c;
    c.vocab_size = 65;
    c.block_size = 16;
    c.n_layer = 2;
    c.n_head = 4;
    c.n_embd = 32;
    return c;
}

/* Copies one PyTorch tensor into a destination, checking the element count
 * rather than trusting the layout. A silent size mismatch here would corrupt
 * every parameter after it in the flat storage. */
static void load_into(float *dst, const char *name, size_t expected)
{
    const tensor *t = oracle_require(&full, name);
    if ((size_t)t->size != expected) {
        printf("  FAILED %s: oracle has %d floats, model expects %zu\n",
               name, t->size, expected);
        checks_failed++;
        return;
    }
    memcpy(dst, t->data, expected * sizeof(float));
}

/* Walks every parameter, applying `visit` to the model pointer and the oracle's
 * name for it. Used for loading parameters and for comparing gradients, so the
 * two cannot cover different sets. */
static void for_each_parameter(gpt_model *model, const char *prefix,
                               void (*visit)(float *, const char *, size_t))
{
    gpt_config c = model->config;
    gpt_params *p = &model->params;
    int C = c.n_embd;
    char name[ORACLE_MAX_NAME];

    snprintf(name, sizeof(name), "%s.wte.weight", prefix);
    visit(p->wte, name, (size_t)c.vocab_size * C);
    snprintf(name, sizeof(name), "%s.wpe.weight", prefix);
    visit(p->wpe, name, (size_t)c.block_size * C);

    for (int l = 0; l < c.n_layer; ++l) {
        snprintf(name, sizeof(name), "%s.blocks.%d.ln_1.weight", prefix, l);
        visit(p->ln1_w + (size_t)l * C, name, (size_t)C);
        snprintf(name, sizeof(name), "%s.blocks.%d.ln_1.bias", prefix, l);
        visit(p->ln1_b + (size_t)l * C, name, (size_t)C);

        snprintf(name, sizeof(name), "%s.blocks.%d.attn.c_attn.weight", prefix, l);
        visit(p->qkv_w + (size_t)l * 3 * C * C, name, (size_t)3 * C * C);
        snprintf(name, sizeof(name), "%s.blocks.%d.attn.c_attn.bias", prefix, l);
        visit(p->qkv_b + (size_t)l * 3 * C, name, (size_t)3 * C);

        snprintf(name, sizeof(name), "%s.blocks.%d.attn.c_proj.weight", prefix, l);
        visit(p->proj_w + (size_t)l * C * C, name, (size_t)C * C);
        snprintf(name, sizeof(name), "%s.blocks.%d.attn.c_proj.bias", prefix, l);
        visit(p->proj_b + (size_t)l * C, name, (size_t)C);

        snprintf(name, sizeof(name), "%s.blocks.%d.ln_2.weight", prefix, l);
        visit(p->ln2_w + (size_t)l * C, name, (size_t)C);
        snprintf(name, sizeof(name), "%s.blocks.%d.ln_2.bias", prefix, l);
        visit(p->ln2_b + (size_t)l * C, name, (size_t)C);

        snprintf(name, sizeof(name), "%s.blocks.%d.mlp.c_fc.weight", prefix, l);
        visit(p->fc_w + (size_t)l * 4 * C * C, name, (size_t)4 * C * C);
        snprintf(name, sizeof(name), "%s.blocks.%d.mlp.c_fc.bias", prefix, l);
        visit(p->fc_b + (size_t)l * 4 * C, name, (size_t)4 * C);

        snprintf(name, sizeof(name), "%s.blocks.%d.mlp.c_proj.weight", prefix, l);
        visit(p->fcproj_w + (size_t)l * C * 4 * C, name, (size_t)C * 4 * C);
        snprintf(name, sizeof(name), "%s.blocks.%d.mlp.c_proj.bias", prefix, l);
        visit(p->fcproj_b + (size_t)l * C, name, (size_t)C);
    }

    snprintf(name, sizeof(name), "%s.ln_f.weight", prefix);
    visit(p->lnf_w, name, (size_t)C);
    snprintf(name, sizeof(name), "%s.ln_f.bias", prefix);
    visit(p->lnf_b, name, (size_t)C);
    snprintf(name, sizeof(name), "%s.lm_head.weight", prefix);
    visit(p->lm_head, name, (size_t)c.vocab_size * C);
}

static void compare_gradient(float *computed, const char *name, size_t count)
{
    const tensor *expected = oracle_require(&full, name);
    if ((size_t)expected->size != count) {
        printf("  FAILED %s: shape mismatch\n", name);
        checks_failed++;
        return;
    }

    int dims[1] = { (int)count };
    tensor got = tensor_wrap(computed, 1, dims);
    tensor want = tensor_wrap(expected->data, 1, dims);
    tensor_diff d = tensor_compare(&got, &want);

    if (!(d.max_abs <= ABS_TOL || d.max_rel <= REL_TOL)) {
        printf("  FAILED gradient %s: max_abs %.3g, max_rel %.3g at %d\n",
               name, (double)d.max_abs, (double)d.max_rel, d.index);
        checks_failed++;
    }
}

/* for_each_parameter hands the model's parameter pointer to its visitor, but
 * gradients live in a parallel struct. This offsets into it by the same
 * distance, which works because both are laid out by the same function. */
static gpt_model *active_model;

static void load_parameter(float *dst, const char *name, size_t count)
{
    load_into(dst, name, count);
}

static void compare_matching_gradient(float *param_ptr, const char *name,
                                      size_t count)
{
    size_t offset = (size_t)(param_ptr - active_model->params.storage);
    compare_gradient(active_model->grads.storage + offset, name, count);
}

static void read_indices(int *out, const char *name, int count)
{
    const tensor *t = oracle_require(&full, name);
    for (int i = 0; i < count && i < t->size; ++i) {
        out[i] = (int)t->data[i];
    }
}

static void test_forward_matches_pytorch(void)
{
    gpt_config config = oracle_config();
    const tensor *idx_tensor = oracle_require(&full, "input.idx");
    int B = idx_tensor->dims[0], T = idx_tensor->dims[1];

    gpt_model model;
    CHECK(gpt_init(&model, config, B));
    active_model = &model;

    for_each_parameter(&model, "param", load_parameter);

    int *idx = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    read_indices(idx, "input.idx", B * T);
    read_indices(targets, "input.targets", B * T);

    float loss = gpt_forward(&model, idx, targets, T);

    const tensor *expected_loss = oracle_require(&full, "loss");
    CHECK_NEAR(loss, expected_loss->data[0], 1e-4);

    const tensor *expected_logits = oracle_require(&full, "act.logits");
    int dims[1] = { expected_logits->size };
    tensor got = tensor_wrap(model.acts.logits, 1, dims);
    tensor want = tensor_wrap(expected_logits->data, 1, dims);
    CHECK_MATCHES(&got, &want, ABS_TOL, REL_TOL);

    free(idx); free(targets);
    gpt_free(&model);
}

static void test_backward_matches_pytorch(void)
{
    gpt_config config = oracle_config();
    const tensor *idx_tensor = oracle_require(&full, "input.idx");
    int B = idx_tensor->dims[0], T = idx_tensor->dims[1];

    gpt_model model;
    CHECK(gpt_init(&model, config, B));
    active_model = &model;

    for_each_parameter(&model, "param", load_parameter);

    int *idx = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    read_indices(idx, "input.idx", B * T);
    read_indices(targets, "input.targets", B * T);

    gpt_forward(&model, idx, targets, T);
    gpt_zero_grad(&model);
    gpt_backward(&model, idx, targets, T);

    /* Every parameter, not a sample of them. A gradient that is wrong in one
     * layer only is the realistic failure, and checking a few would miss it. */
    for_each_parameter(&model, "grad", compare_matching_gradient);

    free(idx); free(targets);
    gpt_free(&model);
}

static void test_gradients_match_finite_differences(void)
{
    /* Independent of PyTorch entirely: perturb a parameter, measure how the
     * loss actually moves, and compare against the analytic gradient.
     *
     * The tolerance is loose -- 5% -- and that is inherent rather than lazy. A
     * central difference has truncation error growing with eps^2, while float32
     * cancellation error grows as the loss difference shrinks relative to the
     * loss itself, so the two error sources squeeze the usable range of eps from
     * both sides. At eps = 1e-2 the difference in loss is around 1e-4 against a
     * loss of 4.3, which is only two or three digits above float32's resolution
     * there.
     *
     * This is therefore a coarse check, and it is aimed at coarse errors: a sign
     * flip, a missing factor, a gradient that belongs to a different parameter.
     * Anything subtler is the oracle's job, and the oracle is tight to 2e-5.
     */
    gpt_config config = oracle_config();
    const tensor *idx_tensor = oracle_require(&full, "input.idx");
    int B = idx_tensor->dims[0], T = idx_tensor->dims[1];

    gpt_model model;
    CHECK(gpt_init(&model, config, B));
    active_model = &model;
    for_each_parameter(&model, "param", load_parameter);

    int *idx = (int *)malloc((size_t)B * T * sizeof(int));
    int *targets = (int *)malloc((size_t)B * T * sizeof(int));
    read_indices(idx, "input.idx", B * T);
    read_indices(targets, "input.targets", B * T);

    gpt_forward(&model, idx, targets, T);
    gpt_zero_grad(&model);
    gpt_backward(&model, idx, targets, T);

    const float eps = 1e-2f;
    /* Spread across the whole parameter block using a prime stride, so the
     * sample touches embeddings, both layers and the head rather than
     * clustering in whichever one happens to come first. Most parameters in a
     * freshly initialised model have gradients too small to resolve against
     * float32 noise, so the stride has to be small enough that enough survive
     * the magnitude filter below. */
    size_t stride = 37;
    int checked = 0, agreed = 0;

    for (size_t i = 0; i < model.params.count && checked < 60; i += stride) {
        float analytic = model.grads.storage[i];

        /* The magnitude floor is derived, not guessed. A central difference
         * moves the loss by about 2*eps*grad, and float32 resolves a loss near
         * 4.3 to roughly 5e-7. Requiring the signal to be several hundred times
         * that resolution gives
         *
         *     2 * 1e-2 * grad > 2e-4     ->     grad > 1e-2
         *
         * Below that the quotient is mostly cancellation noise: a gradient of
         * 1e-3 shifts the loss by 2e-5, which carries barely two significant
         * digits, and comparing it at 5% fails on rounding alone. */
        if (fabsf(analytic) < 1e-2f) {
            continue;
        }

        float original = model.params.storage[i];

        model.params.storage[i] = original + eps;
        float up = gpt_forward(&model, idx, targets, T);

        model.params.storage[i] = original - eps;
        float down = gpt_forward(&model, idx, targets, T);

        model.params.storage[i] = original;

        float numeric = (up - down) / (2.0f * eps);
        float scale = fmaxf(fabsf(analytic), fabsf(numeric));
        float relative = fabsf(analytic - numeric) / scale;

        checked++;
        if (relative < 0.05f) {
            agreed++;
        } else {
            printf("  finite difference: index %zu analytic %.6f numeric %.6f "
                   "(%.1f%% apart)\n", i, (double)analytic, (double)numeric,
                   (double)(100.0f * relative));
        }
    }

    CHECK(checked >= 20);       /* the sample must actually have found gradients */
    CHECK(agreed == checked);

    free(idx); free(targets);
    gpt_free(&model);
}

static void test_parameter_count_matches_the_reference(void)
{
    /* 30,144 for this config, as ref/reference.py reports. A layout bug that
     * over- or under-allocates shows up here rather than as heap corruption. */
    CHECK(gpt_parameter_count(oracle_config()) == 30144);
}

static void test_zero_grad_clears_everything(void)
{
    gpt_config config = oracle_config();
    gpt_model model;
    CHECK(gpt_init(&model, config, 2));

    for (size_t i = 0; i < model.grads.count; ++i) {
        model.grads.storage[i] = 1.0f;
    }
    gpt_zero_grad(&model);

    int clean = 1;
    for (size_t i = 0; i < model.grads.count; ++i) {
        if (model.grads.storage[i] != 0.0f) {
            clean = 0;
        }
    }
    CHECK(clean);
    gpt_free(&model);
}

void register_model_tests(void)
{
    if (!oracle_load(&full, "data/oracle.bin")) {
        exit(2);
    }

    RUN(test_parameter_count_matches_the_reference);
    RUN(test_zero_grad_clears_everything);
    RUN(test_forward_matches_pytorch);
    RUN(test_backward_matches_pytorch);
    RUN(test_gradients_match_finite_differences);

    oracle_free(&full);
}

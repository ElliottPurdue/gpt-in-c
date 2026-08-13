/* AdamW against PyTorch's, over several steps. */

#include "test.h"
#include "oracle.h"
#include "../src/optim.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static oracle units;

static void test_adamw_matches_pytorch_over_several_steps(void)
{
    /* Four steps, not one. Adam's bias correction is largest at step 1 and
     * decays from there, so an implementation that omits it entirely is closest
     * to correct exactly where a one-step test would look. Following it across
     * steps is what pins it. */
    const tensor *initial = oracle_require(&units, "adamw.initial");

    tensor params = tensor_alloc(initial->ndim, initial->dims);
    tensor_copy(&params, initial);

    adamw opt;
    CHECK(adamw_init(&opt, (size_t)params.size));
    opt.lr = 1e-2f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.weight_decay = 0.1f;

    for (int step = 0; step < 4; ++step) {
        char name[64];
        snprintf(name, sizeof(name), "adamw.grad%d", step);
        const tensor *grad = oracle_require(&units, name);

        adamw_step(&opt, params.data, grad->data);

        snprintf(name, sizeof(name), "adamw.after%d", step);
        CHECK_MATCHES(&params, oracle_require(&units, name), 1e-6f, 1e-4f);
    }

    adamw_free(&opt);
    tensor_free(&params);
}

static void test_bias_correction_is_present(void)
{
    /* The first step of Adam moves a parameter by very close to the full
     * learning rate, whatever the gradient's magnitude: m_hat/sqrt(v_hat) is
     * g/|g| once both are bias-corrected. Without the correction the same step
     * is smaller by a factor of about sqrt(1-beta2)/(1-beta1) ~ 0.03, which is
     * the signature this checks. */
    float param = 1.0f, grad = 0.5f;

    adamw opt;
    CHECK(adamw_init(&opt, 1));
    opt.lr = 0.1f;
    opt.weight_decay = 0.0f;

    adamw_step(&opt, &param, &grad);

    CHECK_NEAR(1.0f - param, 0.1f, 1e-4);
    adamw_free(&opt);
}

static void test_weight_decay_is_decoupled(void)
{
    /* With a zero gradient nothing should happen except decay, and the amount
     * must not depend on any gradient history. An L2-style implementation would
     * push the decay through the 1/sqrt(v) scaling and produce a much larger
     * move here. */
    float param = 2.0f, grad = 0.0f;

    adamw opt;
    CHECK(adamw_init(&opt, 1));
    opt.lr = 0.1f;
    opt.weight_decay = 0.5f;

    adamw_step(&opt, &param, &grad);

    /* 2.0 * (1 - 0.1*0.5) = 1.9 exactly, and no gradient term. */
    CHECK_NEAR(param, 1.9f, 1e-5);
    adamw_free(&opt);
}

static void test_clipping_preserves_direction(void)
{
    float grads[4] = { 3.0f, -4.0f, 0.0f, 0.0f };   /* norm exactly 5 */

    float norm = clip_grad_norm(grads, 4, 1.0f);

    CHECK_NEAR(norm, 5.0f, 1e-5);
    CHECK_NEAR(grads[0], 0.6f, 1e-5);
    CHECK_NEAR(grads[1], -0.8f, 1e-5);
    /* The ratio between components is unchanged: only the length moved. */
    CHECK_NEAR(grads[0] / grads[1], 3.0f / -4.0f, 1e-5);
}

/* Bit-exact comparison. "Untouched" is a statement about storage, not about
 * numeric value, and on a toolchain with excess precision the two are not the
 * same claim -- this test originally failed on `grads[0] == 0.3f` while the
 * bytes were provably identical, because one side of the comparison had been
 * widened to 80 bits and the other had not. */
static int same_bits(float a, float b)
{
    return memcmp(&a, &b, sizeof(float)) == 0;
}

static void test_clipping_leaves_small_gradients_alone(void)
{
    float grads[2] = { 0.3f, 0.4f };    /* norm 0.5, under the limit */

    float norm = clip_grad_norm(grads, 2, 1.0f);

    CHECK_NEAR(norm, 0.5f, 1e-5);
    CHECK(same_bits(grads[0], 0.3f));
    CHECK(same_bits(grads[1], 0.4f));
}

void register_optim_tests(void)
{
    if (!oracle_load(&units, "data/units.bin")) {
        exit(2);
    }

    RUN(test_adamw_matches_pytorch_over_several_steps);
    RUN(test_bias_correction_is_present);
    RUN(test_weight_decay_is_decoupled);
    RUN(test_clipping_preserves_direction);
    RUN(test_clipping_leaves_small_gradients_alone);

    oracle_free(&units);
}

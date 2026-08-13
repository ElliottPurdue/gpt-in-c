/* Each operation against PyTorch autograd, forward and backward.
 *
 * Tolerances are absolute 1e-5 or relative 1e-4, either sufficing. Both
 * implementations are float32 and accumulate in a different order, so exact
 * agreement is not available and demanding it would only produce a test that
 * fails on a compiler flag change. These bounds are roughly two orders of
 * magnitude tighter than any real bug produces: a dropped LayerNorm mean term,
 * a missing softmax scale or a transposed matmul all disagree in the first or
 * second significant figure, not the fifth.
 */

#include "test.h"
#include "oracle.h"
#include "../src/ops.h"

#include <stdlib.h>
#include <string.h>

#define ABS_TOL 1e-5f
#define REL_TOL 1e-4f

static oracle units;

/* Allocates a tensor shaped like the oracle's, so a shape change in the dump
 * cannot silently pass by comparing differently shaped buffers. */
static tensor like(const tensor *reference)
{
    return tensor_alloc(reference->ndim, reference->dims);
}

static void test_linear_forward_and_backward(void)
{
    const tensor *x = oracle_require(&units, "linear.x");
    const tensor *weight = oracle_require(&units, "linear.weight");
    const tensor *bias = oracle_require(&units, "linear.bias");
    const tensor *expected = oracle_require(&units, "linear.out");
    const tensor *dout = oracle_require(&units, "linear.dout");

    int rows = x->dims[0], in_features = x->dims[1];
    int out_features = weight->dims[0];

    tensor out = like(expected);
    linear_forward(out.data, x->data, weight->data, bias->data,
                   rows, in_features, out_features);
    CHECK_MATCHES(&out, expected, ABS_TOL, REL_TOL);

    tensor dx = like(x), dweight = like(weight), dbias = like(bias);
    linear_backward(dx.data, dweight.data, dbias.data, dout->data,
                    x->data, weight->data, rows, in_features, out_features);

    CHECK_MATCHES(&dx, oracle_require(&units, "linear.dx"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dweight, oracle_require(&units, "linear.dweight"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dbias, oracle_require(&units, "linear.dbias"), ABS_TOL, REL_TOL);

    tensor_free(&out); tensor_free(&dx);
    tensor_free(&dweight); tensor_free(&dbias);
}

static void test_layernorm_forward_and_backward(void)
{
    const tensor *x = oracle_require(&units, "layernorm.x");
    const tensor *weight = oracle_require(&units, "layernorm.weight");
    const tensor *bias = oracle_require(&units, "layernorm.bias");
    const tensor *expected = oracle_require(&units, "layernorm.out");
    const tensor *dout = oracle_require(&units, "layernorm.dout");

    int cols = x->dims[x->ndim - 1];
    int rows = x->size / cols;

    tensor out = like(expected);
    float *mean = (float *)calloc((size_t)rows, sizeof(float));
    float *rstd = (float *)calloc((size_t)rows, sizeof(float));

    layernorm_forward(out.data, mean, rstd, x->data, weight->data, bias->data,
                      rows, cols, 1e-5f);
    CHECK_MATCHES(&out, expected, ABS_TOL, REL_TOL);

    tensor dx = like(x), dweight = like(weight), dbias = like(bias);
    layernorm_backward(dx.data, dweight.data, dbias.data, dout->data, x->data,
                       mean, rstd, weight->data, rows, cols);

    CHECK_MATCHES(&dx, oracle_require(&units, "layernorm.dx"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dweight, oracle_require(&units, "layernorm.dweight"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dbias, oracle_require(&units, "layernorm.dbias"), ABS_TOL, REL_TOL);

    free(mean); free(rstd);
    tensor_free(&out); tensor_free(&dx);
    tensor_free(&dweight); tensor_free(&dbias);
}

static void test_gelu_forward_and_backward(void)
{
    const tensor *x = oracle_require(&units, "gelu.x");
    const tensor *expected = oracle_require(&units, "gelu.out");
    const tensor *dout = oracle_require(&units, "gelu.dout");

    tensor out = like(expected);
    gelu_forward(out.data, x->data, x->size);
    CHECK_MATCHES(&out, expected, ABS_TOL, REL_TOL);

    tensor dx = like(x);
    gelu_backward(dx.data, dout->data, x->data, x->size);
    CHECK_MATCHES(&dx, oracle_require(&units, "gelu.dx"), ABS_TOL, REL_TOL);

    tensor_free(&out); tensor_free(&dx);
}

static void test_softmax_forward_and_backward(void)
{
    const tensor *x = oracle_require(&units, "softmax.x");
    const tensor *expected = oracle_require(&units, "softmax.out");
    const tensor *dout = oracle_require(&units, "softmax.dout");

    int cols = x->dims[x->ndim - 1];
    int rows = x->size / cols;

    tensor out = like(expected);
    softmax_forward(out.data, x->data, rows, cols);
    CHECK_MATCHES(&out, expected, ABS_TOL, REL_TOL);

    tensor dx = like(x);
    softmax_backward(dx.data, dout->data, out.data, rows, cols);
    CHECK_MATCHES(&dx, oracle_require(&units, "softmax.dx"), ABS_TOL, REL_TOL);

    tensor_free(&out); tensor_free(&dx);
}

static void test_crossentropy_forward_and_backward(void)
{
    const tensor *logits = oracle_require(&units, "crossentropy.logits");
    const tensor *targets = oracle_require(&units, "crossentropy.targets");
    const tensor *expected_loss = oracle_require(&units, "crossentropy.loss");

    int rows = logits->dims[0], classes = logits->dims[1];

    int *target_indices = (int *)calloc((size_t)rows, sizeof(int));
    for (int r = 0; r < rows; ++r) {
        target_indices[r] = (int)targets->data[r];
    }

    tensor probs = like(logits);
    float loss = crossentropy_forward(probs.data, logits->data, target_indices,
                                      rows, classes);
    CHECK_NEAR(loss, expected_loss->data[0], 1e-5);

    tensor dlogits = like(logits);
    crossentropy_backward(dlogits.data, probs.data, target_indices, rows, classes);
    CHECK_MATCHES(&dlogits, oracle_require(&units, "crossentropy.dlogits"),
                  ABS_TOL, REL_TOL);

    free(target_indices);
    tensor_free(&probs); tensor_free(&dlogits);
}

static void test_softmax_survives_logits_that_would_overflow(void)
{
    /* expf(89.0f) is infinity in float32. Without the max-subtraction the row
     * becomes NaN, and because attention logits grow with model width this is
     * reached by ordinary training rather than by adversarial input. */
    float logits[4] = { 300.0f, 299.0f, -300.0f, 305.0f };
    float out[4];

    softmax_forward(out, logits, 1, 4);

    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        CHECK(out[i] == out[i]);            /* not NaN */
        CHECK(out[i] >= 0.0f && out[i] <= 1.0f);
        sum += out[i];
    }
    CHECK_NEAR(sum, 1.0f, 1e-6);
    CHECK(out[3] > out[0]);                 /* the largest logit still wins */
}

static void test_gradients_accumulate_rather_than_overwrite(void)
{
    /* The whole library relies on this: a parameter used twice must collect
     * gradient from both uses. If any backward function assigned instead of
     * adding, a shared weight would silently receive only its last
     * contribution. */
    const tensor *x = oracle_require(&units, "gelu.x");
    const tensor *dout = oracle_require(&units, "gelu.dout");
    const tensor *expected = oracle_require(&units, "gelu.dx");

    tensor once = like(x), twice = like(x);
    gelu_backward(once.data, dout->data, x->data, x->size);
    gelu_backward(twice.data, dout->data, x->data, x->size);
    gelu_backward(twice.data, dout->data, x->data, x->size);

    for (int i = 0; i < x->size; ++i) {
        CHECK_NEAR(twice.data[i], 2.0f * once.data[i], 1e-5);
    }
    CHECK_MATCHES(&once, expected, ABS_TOL, REL_TOL);

    tensor_free(&once); tensor_free(&twice);
}

void register_op_tests(void)
{
    if (!oracle_load(&units, "data/units.bin")) {
        exit(2);
    }

    RUN(test_linear_forward_and_backward);
    RUN(test_layernorm_forward_and_backward);
    RUN(test_gelu_forward_and_backward);
    RUN(test_softmax_forward_and_backward);
    RUN(test_crossentropy_forward_and_backward);
    RUN(test_softmax_survives_logits_that_would_overflow);
    RUN(test_gradients_accumulate_rather_than_overwrite);

    oracle_free(&units);
}

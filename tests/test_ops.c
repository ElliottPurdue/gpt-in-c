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

static void test_attention_forward_and_backward(void)
{
    const tensor *x = oracle_require(&units, "attention.x");
    const tensor *qkv_w = oracle_require(&units, "attention.qkv_w");
    const tensor *qkv_b = oracle_require(&units, "attention.qkv_b");
    const tensor *expected = oracle_require(&units, "attention.out");
    const tensor *dout = oracle_require(&units, "attention.dout");
    const tensor *expected_att = oracle_require(&units, "attention.att");

    int B = x->dims[0], T = x->dims[1], C = x->dims[2];
    int n_head = expected_att->dims[1];

    tensor out = like(expected);
    tensor qkv = tensor_alloc3(B, T, 3 * C);
    tensor att = like(expected_att);

    /* Poisoned before the call, not left zeroed.
     *
     * These buffers are reused every step in training, so a forward pass that
     * relies on them arriving clean works exactly once. In particular the
     * masked upper triangle of att has to be written, not merely left alone --
     * a fresh calloc'd buffer hides that, and this caught a mutation that
     * removing the zeroing otherwise survived.
     */
    for (int i = 0; i < out.size; ++i) {
        out.data[i] = -7777.0f;
    }
    for (int i = 0; i < qkv.size; ++i) {
        qkv.data[i] = -7777.0f;
    }
    for (int i = 0; i < att.size; ++i) {
        att.data[i] = -7777.0f;
    }

    attention_forward(out.data, qkv.data, att.data, x->data,
                      qkv_w->data, qkv_b->data, B, T, C, n_head);

    CHECK_MATCHES(&out, expected, ABS_TOL, REL_TOL);

    /* The attention weights themselves are compared, not just the output. A
     * wrong mask or a wrong scale can still produce a plausible output while
     * the weights are visibly wrong, and this localises that. */
    CHECK_MATCHES(&att, expected_att, ABS_TOL, REL_TOL);

    tensor dx = like(x), dqkv_w = like(qkv_w), dqkv_b = like(qkv_b);
    tensor dqkv = like(&qkv), datt = like(&att);

    attention_backward(dx.data, dqkv_w.data, dqkv_b.data, dqkv.data, datt.data,
                       dout->data, x->data, qkv_w->data, qkv.data, att.data,
                       B, T, C, n_head);

    CHECK_MATCHES(&dx, oracle_require(&units, "attention.dx"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dqkv_w, oracle_require(&units, "attention.dqkv_w"), ABS_TOL, REL_TOL);
    CHECK_MATCHES(&dqkv_b, oracle_require(&units, "attention.dqkv_b"), ABS_TOL, REL_TOL);

    /* The attention-weight gradient is compared only on the causal region, and
     * the reason is worth stating because it looks like a fudge.
     *
     * PyTorch forms the full T x T product y = att @ v, so autograd computes
     * datt[i][j] = sum_d dy[i][d] * v[j][d] for every j -- including j > i,
     * where att itself is zero. Those entries are real numbers in the oracle
     * and exactly zero here, because this implementation never forms the upper
     * triangle at all.
     *
     * Neither is wrong: the masked entries are dead. Softmax backward
     * multiplies each by its own att value, which is zero, so nothing they
     * contain ever reaches a parameter. The proof is immediately above -- dx,
     * dqkv_w and dqkv_b all agree with autograd to 1e-5 despite this
     * difference. Comparing the causal region still catches a wrong softmax
     * Jacobian, a wrong scale, or a transposed v.
     */
    const tensor *expected_datt = oracle_require(&units, "attention.datt");
    tensor masked = like(expected_datt);
    for (int b_ = 0; b_ < B; ++b_) {
        for (int h_ = 0; h_ < n_head; ++h_) {
            for (int i_ = 0; i_ < T; ++i_) {
                for (int j_ = 0; j_ <= i_; ++j_) {
                    size_t k_ = (((size_t)b_ * n_head + h_) * T + i_) * T + j_;
                    masked.data[k_] = expected_datt->data[k_];
                }
            }
        }
    }
    CHECK_MATCHES(&datt, &masked, ABS_TOL, REL_TOL);

    /* And assert the upper triangle really is untouched, so the masking above
     * cannot hide this implementation writing something there. */
    for (int b_ = 0; b_ < B; ++b_) {
        for (int h_ = 0; h_ < n_head; ++h_) {
            for (int i_ = 0; i_ < T; ++i_) {
                for (int j_ = i_ + 1; j_ < T; ++j_) {
                    size_t k_ = (((size_t)b_ * n_head + h_) * T + i_) * T + j_;
                    CHECK(datt.data[k_] == 0.0f);
                }
            }
        }
    }
    tensor_free(&masked);

    tensor_free(&out); tensor_free(&qkv); tensor_free(&att);
    tensor_free(&dx); tensor_free(&dqkv_w); tensor_free(&dqkv_b);
    tensor_free(&dqkv); tensor_free(&datt);
}

static void test_attention_cannot_see_the_future(void)
{
    /* The causal mask, asserted directly rather than inferred from the output.
     *
     * A model whose mask leaks reads the token it is being asked to predict,
     * so its training loss collapses toward zero and its generated text is
     * garbage. That failure looks like success on every metric printed during
     * training, which is why this is checked structurally.
     */
    const int B = 1, T = 5, C = 4, n_head = 2;
    tensor x = tensor_alloc3(B, T, C);
    tensor qkv_w = tensor_alloc2(3 * C, C);
    tensor qkv_b = tensor_alloc(1, (int[]){ 3 * C });
    tensor out = tensor_alloc3(B, T, C);
    tensor qkv = tensor_alloc3(B, T, 3 * C);
    tensor att = tensor_alloc4(B, n_head, T, T);

    for (int i = 0; i < x.size; ++i) {
        x.data[i] = 0.1f * (float)(i % 7) - 0.3f;
    }
    for (int i = 0; i < qkv_w.size; ++i) {
        qkv_w.data[i] = 0.05f * (float)((i % 11) - 5);
    }
    for (int i = 0; i < att.size; ++i) {
        att.data[i] = 123.0f;       /* must be overwritten, including the mask */
    }

    attention_forward(out.data, qkv.data, att.data, x.data,
                      qkv_w.data, qkv_b.data, B, T, C, n_head);

    for (int h = 0; h < n_head; ++h) {
        for (int i = 0; i < T; ++i) {
            float row_sum = 0.0f;
            for (int j = 0; j < T; ++j) {
                float w = att.data[((size_t)h * T + i) * T + j];
                if (j > i) {
                    CHECK(w == 0.0f);       /* strictly future: no weight at all */
                } else {
                    CHECK(w > 0.0f);
                    row_sum += w;
                }
            }
            CHECK_NEAR(row_sum, 1.0f, 1e-5);
        }
    }

    /* Position 0 attends only to itself, so its output must be exactly its own
     * value vector -- a check that does not depend on any other position. */
    for (int h = 0; h < n_head; ++h) {
        CHECK_NEAR(att.data[(size_t)h * T * T], 1.0f, 1e-6);
    }

    tensor_free(&x); tensor_free(&qkv_w); tensor_free(&qkv_b);
    tensor_free(&out); tensor_free(&qkv); tensor_free(&att);
}

static void test_editing_a_later_token_cannot_change_an_earlier_output(void)
{
    /* The behavioural consequence of the mask, and the one that actually
     * matters: changing token t must leave outputs 0..t-1 bit-identical. The
     * structural test above would still pass if the value vectors were mixed
     * across positions somewhere after the softmax. */
    const int B = 1, T = 6, C = 4, n_head = 2;
    tensor x = tensor_alloc3(B, T, C);
    tensor qkv_w = tensor_alloc2(3 * C, C);
    tensor qkv_b = tensor_alloc(1, (int[]){ 3 * C });
    tensor out_a = tensor_alloc3(B, T, C), out_b = tensor_alloc3(B, T, C);
    tensor qkv = tensor_alloc3(B, T, 3 * C);
    tensor att = tensor_alloc4(B, n_head, T, T);

    for (int i = 0; i < x.size; ++i) {
        x.data[i] = 0.2f * (float)(i % 5) - 0.4f;
    }
    for (int i = 0; i < qkv_w.size; ++i) {
        qkv_w.data[i] = 0.05f * (float)((i % 11) - 5);
    }

    attention_forward(out_a.data, qkv.data, att.data, x.data,
                      qkv_w.data, qkv_b.data, B, T, C, n_head);

    const int edited = 4;
    for (int c = 0; c < C; ++c) {
        x.data[(size_t)edited * C + c] += 17.0f;    /* a large, obvious change */
    }
    attention_forward(out_b.data, qkv.data, att.data, x.data,
                      qkv_w.data, qkv_b.data, B, T, C, n_head);

    for (int i = 0; i < edited; ++i) {
        for (int c = 0; c < C; ++c) {
            size_t k = (size_t)i * C + c;
            CHECK(out_a.data[k] == out_b.data[k]);
        }
    }
    /* And the edited position itself must have moved, or the test proves
     * nothing about the mask -- only that the input was ignored. */
    CHECK(out_a.data[(size_t)edited * C] != out_b.data[(size_t)edited * C]);

    tensor_free(&x); tensor_free(&qkv_w); tensor_free(&qkv_b);
    tensor_free(&out_a); tensor_free(&out_b);
    tensor_free(&qkv); tensor_free(&att);
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
    RUN(test_attention_forward_and_backward);
    RUN(test_attention_cannot_see_the_future);
    RUN(test_editing_a_later_token_cannot_change_an_earlier_output);

    oracle_free(&units);
}

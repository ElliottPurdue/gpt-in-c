#include "ops.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Linear
 * ------------------------------------------------------------------------ */

/* Rows are processed in blocks so that each row of the weight matrix is loaded
 * once per block instead of once per row.
 *
 * The naive loop -- for each row, walk the whole weight matrix -- has good
 * locality within a single dot product and terrible locality across them. The
 * qkv projection's weights are 110 KB against a 32 KB L1, so every row evicts
 * them and the next row reloads them: 1024 rows x 110 KB is about 113 MB of
 * traffic for 28 MFLOP of arithmetic. The arithmetic was never the cost.
 *
 * With ROW_BLOCK rows in flight, one weight element is fetched and used
 * ROW_BLOCK times before moving on, cutting weight traffic by that factor. The
 * block of input rows is small enough to sit in L1 alongside it.
 *
 * The summation order is unchanged: each output element still accumulates over
 * i from 0 upward. That is deliberate and load-bearing -- floating-point
 * addition is not associative, so reordering it would change results, and the
 * training loop's determinism is what proves this optimisation did not alter
 * the mathematics. The loss curve after this change is bit-identical to the
 * curve before it. */
#define ROW_BLOCK 8
#define OUT_BLOCK 16

void linear_forward(float *out, const float *x, const float *weight,
                    const float *bias, int rows, int in_features,
                    int out_features)
{
    for (int r0 = 0; r0 < rows; r0 += ROW_BLOCK) {
        int block = rows - r0 < ROW_BLOCK ? rows - r0 : ROW_BLOCK;

        for (int o = 0; o < out_features; ++o) {
            const float *wo = weight + (size_t)o * in_features;
            float acc[ROW_BLOCK];

            /* The whole array, not just the live prefix: the tail is never read,
             * but proving that to the optimiser costs more than the eight
             * stores, and -O3 rejects the build otherwise. */
            for (int b = 0; b < ROW_BLOCK; ++b) {
                acc[b] = bias ? bias[o] : 0.0f;
            }
            for (int i = 0; i < in_features; ++i) {
                float w = wo[i];
                for (int b = 0; b < block; ++b) {
                    acc[b] += x[(size_t)(r0 + b) * in_features + i] * w;
                }
            }
            for (int b = 0; b < block; ++b) {
                out[(size_t)(r0 + b) * out_features + o] = acc[b];
            }
        }
    }
}

/* Split into three loops, one per output, rather than one fused loop.
 *
 * The fused version walks rows on the outside, which forces the whole weight
 * matrix and the whole weight-gradient matrix through cache on every row. Split
 * apart, each loop can be blocked for the array it actually writes.
 *
 * Splitting does not change any summation order, which is what makes it safe:
 * dx[r][i] still accumulates over o ascending, and dweight[o][i] and dbias[o]
 * still accumulate over r ascending, exactly as when the loops were nested. */
void linear_backward(float *dx, float *dweight, float *dbias,
                     const float *dout, const float *x, const float *weight,
                     int rows, int in_features, int out_features)
{
    /* dx[r][i] = sum_o dy[r][o] * W[o][i]. Blocked over rows for the same
     * reason as the forward pass: one weight row serves ROW_BLOCK outputs. */
    if (dx) {
        for (int r0 = 0; r0 < rows; r0 += ROW_BLOCK) {
            int block = rows - r0 < ROW_BLOCK ? rows - r0 : ROW_BLOCK;

            for (int o = 0; o < out_features; ++o) {
                const float *wo = weight + (size_t)o * in_features;

                for (int b = 0; b < block; ++b) {
                    float g = dout[(size_t)(r0 + b) * out_features + o];
                    float *dxr = dx + (size_t)(r0 + b) * in_features;
                    for (int i = 0; i < in_features; ++i) {
                        dxr[i] += g * wo[i];
                    }
                }
            }
        }
    }

    /* dW[o][i] = sum_r dy[r][o] * x[r][i]. Blocked over outputs: OUT_BLOCK rows
     * of the gradient stay resident while the inputs stream past once, rather
     * than the inputs streaming once per output. */
    if (dweight) {
        for (int o0 = 0; o0 < out_features; o0 += OUT_BLOCK) {
            int block = out_features - o0 < OUT_BLOCK ? out_features - o0 : OUT_BLOCK;

            for (int r = 0; r < rows; ++r) {
                const float *xr = x + (size_t)r * in_features;
                const float *doutr = dout + (size_t)r * out_features;

                for (int b = 0; b < block; ++b) {
                    float g = doutr[o0 + b];
                    float *dwo = dweight + (size_t)(o0 + b) * in_features;
                    for (int i = 0; i < in_features; ++i) {
                        dwo[i] += g * xr[i];
                    }
                }
            }
        }
    }

    /* db[o] = sum_r dy[r][o]. */
    if (dbias) {
        for (int r = 0; r < rows; ++r) {
            const float *doutr = dout + (size_t)r * out_features;
            for (int o = 0; o < out_features; ++o) {
                dbias[o] += doutr[o];
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * LayerNorm
 * ------------------------------------------------------------------------ */

void layernorm_forward(float *out, float *mean, float *rstd, const float *x,
                       const float *weight, const float *bias,
                       int rows, int cols, float eps)
{
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * cols;

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            sum += xr[c];
        }
        float m = sum / (float)cols;

        /* Variance from the centred values rather than E[x^2] - E[x]^2. The
         * latter is one pass cheaper and subtracts two large nearly-equal
         * numbers, losing most of the significand when the mean is large
         * relative to the spread. */
        float variance = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float d = xr[c] - m;
            variance += d * d;
        }
        variance /= (float)cols;

        float inv = 1.0f / sqrtf(variance + eps);
        mean[r] = m;
        rstd[r] = inv;

        float *outr = out + (size_t)r * cols;
        for (int c = 0; c < cols; ++c) {
            float normalized = (xr[c] - m) * inv;
            outr[c] = normalized * weight[c] + bias[c];
        }
    }
}

void layernorm_backward(float *dx, float *dweight, float *dbias,
                        const float *dout, const float *x, const float *mean,
                        const float *rstd, const float *weight,
                        int rows, int cols)
{
    /* With n = cols, xhat = (x - mean) * rstd and g = dout * weight:
     *
     *     dx = rstd * (g - mean(g) - xhat * mean(g * xhat))
     *
     * The two mean terms are what makes this different from a plain scale, and
     * they are what a careless derivation drops. They exist because mean and
     * variance are themselves functions of every element of the row: changing
     * one input moves the statistics, which moves every other output. Dropping
     * them still trains -- slightly worse -- which is why this needs a test
     * rather than a look. */
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * cols;
        const float *doutr = dout + (size_t)r * cols;
        float m = mean[r], inv = rstd[r];

        float mean_g = 0.0f;         /* mean over the row of dout*weight */
        float mean_g_xhat = 0.0f;    /* mean over the row of dout*weight*xhat */

        for (int c = 0; c < cols; ++c) {
            float xhat = (xr[c] - m) * inv;
            float g = doutr[c] * weight[c];
            mean_g += g;
            mean_g_xhat += g * xhat;

            if (dweight) {
                dweight[c] += doutr[c] * xhat;
            }
            if (dbias) {
                dbias[c] += doutr[c];
            }
        }
        mean_g /= (float)cols;
        mean_g_xhat /= (float)cols;

        if (dx) {
            float *dxr = dx + (size_t)r * cols;
            for (int c = 0; c < cols; ++c) {
                float xhat = (xr[c] - m) * inv;
                float g = doutr[c] * weight[c];
                dxr[c] += inv * (g - mean_g - xhat * mean_g_xhat);
            }
        }
    }
}

/* ------------------------------------------------------------------------
 * GELU
 * ------------------------------------------------------------------------ */

#define SQRT_1_2  0.70710678118654752440f   /* 1/sqrt(2) */
#define SQRT_2_PI 0.79788456080286535588f   /* sqrt(2/pi) */

void gelu_forward(float *out, const float *x, int n)
{
    for (int i = 0; i < n; ++i) {
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * SQRT_1_2));
    }
}

void gelu_backward(float *dx, const float *dout, const float *x, int n)
{
    /* d/dx [0.5x(1 + erf(x/sqrt2))] = 0.5(1 + erf(x/sqrt2)) + x * phi(x)
     * where phi is the standard normal density. The second term is the one
     * that distinguishes GELU's gradient from a gated linear unit's. */
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        float cdf = 0.5f * (1.0f + erff(v * SQRT_1_2));
        float pdf = 0.5f * SQRT_2_PI * expf(-0.5f * v * v);
        dx[i] += dout[i] * (cdf + v * pdf);
    }
}

/* ------------------------------------------------------------------------
 * Softmax
 * ------------------------------------------------------------------------ */

void softmax_forward(float *out, const float *x, int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * cols;
        float *outr = out + (size_t)r * cols;

        /* Subtracting the row maximum is mathematically a no-op and numerically
         * essential: attention logits routinely exceed 88, where expf overflows
         * to infinity in float32 and the row becomes NaN. */
        float maximum = xr[0];
        for (int c = 1; c < cols; ++c) {
            if (xr[c] > maximum) {
                maximum = xr[c];
            }
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(xr[c] - maximum);
            outr[c] = e;
            sum += e;
        }

        float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            outr[c] *= inv;
        }
    }
}

void softmax_backward(float *dx, const float *dout, const float *out,
                      int rows, int cols)
{
    for (int r = 0; r < rows; ++r) {
        const float *doutr = dout + (size_t)r * cols;
        const float *outr = out + (size_t)r * cols;
        float *dxr = dx + (size_t)r * cols;

        float dot = 0.0f;
        for (int c = 0; c < cols; ++c) {
            dot += doutr[c] * outr[c];
        }
        for (int c = 0; c < cols; ++c) {
            dxr[c] += outr[c] * (doutr[c] - dot);
        }
    }
}

/* ------------------------------------------------------------------------
 * Cross entropy
 * ------------------------------------------------------------------------ */

float crossentropy_forward(float *probs, const float *logits,
                           const int *targets, int rows, int classes)
{
    softmax_forward(probs, logits, rows, classes);

    float total = 0.0f;
    for (int r = 0; r < rows; ++r) {
        float p = probs[(size_t)r * classes + targets[r]];

        /* The floor matters: a confident wrong prediction drives p to zero and
         * logf(0) is -inf, which poisons the mean and every gradient computed
         * from it. Clamping bounds one sample's contribution at ~30 instead. */
        if (p < 1e-13f) {
            p = 1e-13f;
        }
        total += -logf(p);
    }
    return total / (float)rows;
}

void crossentropy_backward(float *dlogits, const float *probs,
                           const int *targets, int rows, int classes)
{
    /* The softmax and the log cancel: d(loss)/d(logits) = (p - onehot) / rows.
     * Computed separately, the log's 1/p and the softmax's p would multiply
     * back to this, but only after each had been rounded. */
    float scale = 1.0f / (float)rows;

    for (int r = 0; r < rows; ++r) {
        const float *pr = probs + (size_t)r * classes;
        float *dr = dlogits + (size_t)r * classes;

        for (int c = 0; c < classes; ++c) {
            dr[c] += scale * pr[c];
        }
        dr[targets[r]] -= scale;
    }
}

/* ------------------------------------------------------------------------
 * Multi-head causal self-attention
 * ------------------------------------------------------------------------ */

/* Offsets into the (B, T, 3C) projection buffer. `which` is 0 for q, 1 for k,
 * 2 for v. Written once and used by both passes, because getting this indexing
 * wrong in only one of them produces a gradient that is subtly incorrect rather
 * than obviously so. */
static size_t qkv_offset(int b, int t, int which, int head, int T, int C,
                         int head_dim)
{
    return ((size_t)(b * T + t) * 3 + which) * C + (size_t)head * head_dim;
}

void attention_forward(float *out, float *qkv, float *att,
                       const float *x, const float *qkv_w, const float *qkv_b,
                       int B, int T, int C, int n_head)
{
    int head_dim = C / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    linear_forward(qkv, x, qkv_w, qkv_b, B * T, C, 3 * C);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_head; ++h) {
            float *att_head = att + ((size_t)b * n_head + h) * T * T;

            for (int i = 0; i < T; ++i) {
                const float *q = qkv + qkv_offset(b, i, 0, h, T, C, head_dim);
                float *att_row = att_head + (size_t)i * T;

                /* Causality is enforced by never computing the scores above the
                 * diagonal, rather than by computing them and adding -inf. The
                 * result is identical and it halves the work, but the entries
                 * still have to be zeroed: the backward pass and the oracle
                 * comparison both read the full row. */
                float maximum = -INFINITY;
                for (int j = 0; j <= i; ++j) {
                    const float *k = qkv + qkv_offset(b, j, 1, h, T, C, head_dim);
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += q[d] * k[d];
                    }
                    /* The 1/sqrt(head_dim) scale keeps the logits' variance
                     * independent of head size. Without it softmax saturates as
                     * the model widens and the gradients vanish. */
                    dot *= scale;
                    att_row[j] = dot;
                    if (dot > maximum) {
                        maximum = dot;
                    }
                }

                float sum = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    float e = expf(att_row[j] - maximum);
                    att_row[j] = e;
                    sum += e;
                }
                float inv = 1.0f / sum;
                for (int j = 0; j <= i; ++j) {
                    att_row[j] *= inv;
                }
                for (int j = i + 1; j < T; ++j) {
                    att_row[j] = 0.0f;
                }

                /* Merge straight into the (B, T, C) output. Writing to a
                 * (B, n_head, T, head_dim) buffer and transposing afterwards
                 * would need a second pass over the same data. */
                float *outi = out + ((size_t)b * T + i) * C + (size_t)h * head_dim;
                for (int d = 0; d < head_dim; ++d) {
                    outi[d] = 0.0f;
                }
                for (int j = 0; j <= i; ++j) {
                    const float *v = qkv + qkv_offset(b, j, 2, h, T, C, head_dim);
                    float weight = att_row[j];
                    for (int d = 0; d < head_dim; ++d) {
                        outi[d] += weight * v[d];
                    }
                }
            }
        }
    }
}

void attention_backward(float *dx, float *dqkv_w, float *dqkv_b,
                        float *dqkv, float *datt,
                        const float *dout, const float *x, const float *qkv_w,
                        const float *qkv, const float *att,
                        int B, int T, int C, int n_head)
{
    int head_dim = C / n_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < B; ++b) {
        for (int h = 0; h < n_head; ++h) {
            const float *att_head = att + ((size_t)b * n_head + h) * T * T;
            float *datt_head = datt + ((size_t)b * n_head + h) * T * T;

            for (int i = 0; i < T; ++i) {
                const float *att_row = att_head + (size_t)i * T;
                float *datt_row = datt_head + (size_t)i * T;
                const float *douti = dout + ((size_t)b * T + i) * C
                                   + (size_t)h * head_dim;

                /* y[i] = sum_j att[i][j] * v[j], so the incoming gradient splits
                 * into one term for the weights and one for the values. */
                for (int j = 0; j <= i; ++j) {
                    const float *v = qkv + qkv_offset(b, j, 2, h, T, C, head_dim);
                    float *dv = dqkv + qkv_offset(b, j, 2, h, T, C, head_dim);

                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        dot += douti[d] * v[d];
                        dv[d] += att_row[j] * douti[d];
                    }
                    datt_row[j] = dot;
                }

                /* Softmax backward, restricted to the unmasked prefix. The
                 * masked entries need no special handling: att is zero there, so
                 * their contribution to the row's dot product and to their own
                 * gradient is zero either way. */
                float dot = 0.0f;
                for (int j = 0; j <= i; ++j) {
                    dot += datt_row[j] * att_row[j];
                }

                float *dq = dqkv + qkv_offset(b, i, 0, h, T, C, head_dim);
                const float *q = qkv + qkv_offset(b, i, 0, h, T, C, head_dim);

                for (int j = 0; j <= i; ++j) {
                    float dscore = att_row[j] * (datt_row[j] - dot) * scale;

                    const float *k = qkv + qkv_offset(b, j, 1, h, T, C, head_dim);
                    float *dk = dqkv + qkv_offset(b, j, 1, h, T, C, head_dim);

                    for (int d = 0; d < head_dim; ++d) {
                        dq[d] += dscore * k[d];
                        dk[d] += dscore * q[d];
                    }
                }
            }
        }
    }

    linear_backward(dx, dqkv_w, dqkv_b, dqkv, x, qkv_w, B * T, C, 3 * C);
}

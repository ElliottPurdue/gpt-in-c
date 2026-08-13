#include "ops.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------------
 * Linear
 * ------------------------------------------------------------------------ */

void linear_forward(float *out, const float *x, const float *weight,
                    const float *bias, int rows, int in_features,
                    int out_features)
{
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * in_features;
        float *outr = out + (size_t)r * out_features;

        for (int o = 0; o < out_features; ++o) {
            const float *wo = weight + (size_t)o * in_features;
            float sum = bias ? bias[o] : 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += xr[i] * wo[i];
            }
            outr[o] = sum;
        }
    }
}

void linear_backward(float *dx, float *dweight, float *dbias,
                     const float *dout, const float *x, const float *weight,
                     int rows, int in_features, int out_features)
{
    /* y[r][o] = sum_i x[r][i] * W[o][i] + b[o], so
     *     dx[r][i] = sum_o dy[r][o] * W[o][i]
     *     dW[o][i] = sum_r dy[r][o] * x[r][i]
     *     db[o]    = sum_r dy[r][o]
     *
     * The two loops are separated by which output they write, not by which
     * input they read, so neither needs a temporary. */
    for (int r = 0; r < rows; ++r) {
        const float *doutr = dout + (size_t)r * out_features;
        const float *xr = x + (size_t)r * in_features;
        float *dxr = dx ? dx + (size_t)r * in_features : NULL;

        for (int o = 0; o < out_features; ++o) {
            float g = doutr[o];
            const float *wo = weight + (size_t)o * in_features;

            if (dxr) {
                for (int i = 0; i < in_features; ++i) {
                    dxr[i] += g * wo[i];
                }
            }
            if (dweight) {
                float *dwo = dweight + (size_t)o * in_features;
                for (int i = 0; i < in_features; ++i) {
                    dwo[i] += g * xr[i];
                }
            }
            if (dbias) {
                dbias[o] += g;
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

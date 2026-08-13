#include "optim.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

int adamw_init(adamw *opt, size_t count)
{
    memset(opt, 0, sizeof(*opt));

    opt->m = (float *)calloc(count, sizeof(float));
    opt->v = (float *)calloc(count, sizeof(float));
    if (!opt->m || !opt->v) {
        adamw_free(opt);
        return 0;
    }

    opt->count = count;
    opt->lr = ADAMW_DEFAULT_LR;
    opt->beta1 = ADAMW_DEFAULT_BETA1;
    opt->beta2 = ADAMW_DEFAULT_BETA2;
    opt->eps = ADAMW_DEFAULT_EPS;
    opt->weight_decay = ADAMW_DEFAULT_DECAY;
    opt->step = 0;
    return 1;
}

void adamw_free(adamw *opt)
{
    free(opt->m);
    free(opt->v);
    memset(opt, 0, sizeof(*opt));
}

void adamw_step(adamw *opt, float *params, const float *grads)
{
    opt->step++;

    /* Computed once per step rather than per parameter. powf of a scalar is
     * cheap, but not thirty thousand times cheap. */
    float bias1 = 1.0f - powf(opt->beta1, (float)opt->step);
    float bias2 = 1.0f - powf(opt->beta2, (float)opt->step);

    for (size_t i = 0; i < opt->count; ++i) {
        float g = grads[i];

        opt->m[i] = opt->beta1 * opt->m[i] + (1.0f - opt->beta1) * g;
        opt->v[i] = opt->beta2 * opt->v[i] + (1.0f - opt->beta2) * g * g;

        float m_hat = opt->m[i] / bias1;
        float v_hat = opt->v[i] / bias2;

        /* Decoupled: the decay multiplies the parameter directly and never
         * enters m or v, so it is not rescaled by the gradient history. */
        params[i] -= opt->lr * opt->weight_decay * params[i];
        params[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);
    }
}

float clip_grad_norm(float *grads, size_t count, float max_norm)
{
    /* Accumulated in double. The sum runs over every parameter in the model, and
     * in float32 the running total grows large enough that individual squared
     * terms stop changing it -- the classic case where a sum of many small
     * positive numbers silently stalls. */
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        total += (double)grads[i] * (double)grads[i];
    }

    float norm = (float)sqrt(total);
    if (norm > max_norm && norm > 0.0f) {
        float scale = max_norm / norm;
        for (size_t i = 0; i < count; ++i) {
            grads[i] *= scale;
        }
    }
    return norm;
}

/* AdamW.
 *
 * Adam keeps two running averages per parameter -- of the gradient, and of the
 * gradient squared -- and divides one by the square root of the other. The
 * effect is a per-parameter learning rate: a weight whose gradient has been
 * consistently large gets a smaller step than one whose gradient is small but
 * steady. That is why it trains transformers where plain SGD needs careful
 * tuning per layer.
 *
 * The W is not a detail. Adam with L2 regularization added to the gradient does
 * something different from Adam with weight decay applied to the parameter,
 * because the L2 term passes through the same 1/sqrt(v) scaling as everything
 * else -- so parameters with large gradients get *less* regularization, which
 * is backwards. Decoupling it, as here, applies the same shrinkage to every
 * parameter regardless of its gradient history.
 *
 * The optimizer sees a flat float array and knows nothing about the model's
 * structure, which is the payoff for keeping parameters in one allocation.
 */

#ifndef OPTIM_H
#define OPTIM_H

#include <stddef.h>

typedef struct {
    float *m;           /* first moment, one per parameter */
    float *v;           /* second moment */
    size_t count;

    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    long step;          /* 1-based; drives the bias correction */
} adamw;

#define ADAMW_DEFAULT_LR      3e-4f
#define ADAMW_DEFAULT_BETA1   0.9f
#define ADAMW_DEFAULT_BETA2   0.999f
#define ADAMW_DEFAULT_EPS     1e-8f
#define ADAMW_DEFAULT_DECAY   0.1f

/* Returns 0 if the moment buffers could not be allocated. */
int adamw_init(adamw *opt, size_t count);
void adamw_free(adamw *opt);

/* One update. Reads grads, writes params, advances the moments.
 *
 * Both moments start at zero, which biases them toward zero for the first
 * several steps -- badly at step 1, where the estimate is (1 - beta) times the
 * true value, so 0.001 times it for the second moment. The bias correction
 * divides that factor out. Omitting it does not prevent training but makes the
 * first steps far too small, and it is invisible in a single-step test because
 * the correction is largest exactly there. */
void adamw_step(adamw *opt, float *params, const float *grads);

/* Scales gradients so their global L2 norm does not exceed max_norm, and
 * returns the norm before scaling.
 *
 * A single bad batch can produce a gradient hundreds of times the usual size,
 * and Adam does not save you from it: the update is bounded by roughly the
 * learning rate per step, but the second-moment estimate absorbs the spike and
 * then suppresses that parameter's updates for hundreds of steps afterwards.
 * Clipping the whole vector, rather than each element, preserves the gradient's
 * direction and changes only its length. */
float clip_grad_norm(float *grads, size_t count, float max_norm);

#endif /* OPTIM_H */

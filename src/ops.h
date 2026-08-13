/* Transformer primitives, forward and backward.
 *
 * Every backward pass here is hand-derived. That is the point of the project:
 * the forward pass of a transformer is short enough to write from the paper,
 * and autograd makes the backward pass free, so writing only the forward half
 * demonstrates nothing that reading the paper would not.
 *
 * CONVENTIONS
 *
 * Gradients ACCUMULATE into their output buffers. Callers zero them once per
 * step. This is what lets a parameter used in several places -- tied embeddings,
 * or a weight shared across positions -- collect gradient from each use without
 * the operations knowing about each other. It also means forgetting to zero
 * produces a slow, plausible-looking divergence rather than an obvious crash,
 * which is why the training loop zeroes in exactly one place.
 *
 * Backward functions take the forward pass's inputs and its saved outputs.
 * Nothing is recomputed that was already computed, and nothing is cached that
 * can be passed in; the caller owns all storage.
 *
 * Shapes are given in comments as (rows, cols) or (batch, time, channels).
 * There is no shape checking at runtime -- these are internal calls on
 * fixed-size buffers, and the test suite pins every shape against the oracle.
 */

#ifndef OPS_H
#define OPS_H

#include "tensor.h"

/* ---- Linear: y = x W^T + b ---------------------------------------------
 *
 * W is stored (out_features, in_features), matching PyTorch's nn.Linear, so
 * the forward pass walks both operands' rows contiguously. */
void linear_forward(float *out, const float *x, const float *weight,
                    const float *bias, int rows, int in_features,
                    int out_features);

void linear_backward(float *dx, float *dweight, float *dbias,
                     const float *dout, const float *x, const float *weight,
                     int rows, int in_features, int out_features);

/* ---- LayerNorm over the last axis --------------------------------------
 *
 * mean and rstd are written by the forward pass and required by the backward
 * pass. Recomputing them from x would be arithmetically identical but would
 * double the cost of the backward pass for no benefit. */
void layernorm_forward(float *out, float *mean, float *rstd, const float *x,
                       const float *weight, const float *bias,
                       int rows, int cols, float eps);

void layernorm_backward(float *dx, float *dweight, float *dbias,
                        const float *dout, const float *x, const float *mean,
                        const float *rstd, const float *weight,
                        int rows, int cols);

/* ---- GELU, exact ---------------------------------------------------------
 *
 * 0.5x(1 + erf(x/sqrt(2))), not the tanh approximation. The two differ by about
 * 1e-3, which is three orders of magnitude above the tolerance these tests run
 * at, so the choice is load-bearing rather than cosmetic. */
void gelu_forward(float *out, const float *x, int n);
void gelu_backward(float *dx, const float *dout, const float *x, int n);

/* ---- Softmax over the last axis ----------------------------------------- */
void softmax_forward(float *out, const float *x, int rows, int cols);

/* The Jacobian is dense -- every output in a row depends on every input in it --
 * but it collapses to dx = y * (dy - sum(dy * y)), which costs two passes and
 * no matrix. */
void softmax_backward(float *dx, const float *dout, const float *out,
                      int rows, int cols);

/* ---- Cross entropy over logits ------------------------------------------
 *
 * Fused with the softmax, because the composition's gradient is (p - onehot)/n
 * and computing it that way avoids both the division by a near-zero probability
 * and the catastrophic cancellation that a separate softmax and log would
 * produce for a confident wrong prediction. Returns the mean loss. */
float crossentropy_forward(float *probs, const float *logits,
                           const int *targets, int rows, int classes);

void crossentropy_backward(float *dlogits, const float *probs,
                           const int *targets, int rows, int classes);

#endif /* OPS_H */

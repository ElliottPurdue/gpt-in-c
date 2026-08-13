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

/* ---- Multi-head causal self-attention -----------------------------------
 *
 * Covers the fused QKV projection through to the merged output, but not the
 * output projection that follows it -- that is an ordinary linear, and keeping
 * it outside means this function has one job.
 *
 * LAYOUT. qkv is (B, T, 3C), holding q, k and v concatenated along the last
 * axis in that order, which is what a single projection produces and what
 * PyTorch's split(C, dim=2) expects. Within each of those C-wide sections the
 * heads are laid out contiguously, so head h of q at position t starts at
 * qkv[((b*T + t)*3 + 0)*C + h*head_dim]. That indexing is the whole of the
 * "transpose(1, 2)" the PyTorch version writes explicitly; no data moves.
 *
 * att is (B, n_head, T, T), holding the softmax output. Entries above the
 * diagonal are written as exact zero rather than left undefined, so the buffer
 * can be compared against PyTorch's masked result directly.
 *
 * The caller owns qkv and att. They are outputs of the forward pass and inputs
 * to the backward pass: recomputing them would cost a second forward pass, and
 * caching them inside would mean the library allocating.
 */
void attention_forward(float *out, float *qkv, float *att,
                       const float *x, const float *qkv_w, const float *qkv_b,
                       int B, int T, int C, int n_head);

/* dqkv and datt are scratch, sized like qkv and att. The caller zeroes them;
 * like every other backward here, gradients accumulate. */
void attention_backward(float *dx, float *dqkv_w, float *dqkv_b,
                        float *dqkv, float *datt,
                        const float *dout, const float *x, const float *qkv_w,
                        const float *qkv, const float *att,
                        int B, int T, int C, int n_head);

#endif /* OPS_H */

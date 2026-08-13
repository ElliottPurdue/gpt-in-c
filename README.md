# GPT in C

A transformer implemented from scratch in dependency-free C99 — forward pass
*and* hand-derived backward pass — with PyTorch used as a numerical oracle
rather than as a framework.

The forward pass of a transformer is short enough to write from the paper. The
backward pass is where the understanding is, and autograd makes it free, so an
implementation that writes only the forward half demonstrates nothing that
reading the paper would not. Every gradient here is derived by hand and checked
against `torch.autograd` element by element.

**Why an oracle.** An incorrect gradient almost never announces itself. It
usually still decreases the loss, just more slowly, so "it trains" is not
evidence of correctness — it is the failure mode. Each operation is therefore
pinned against autograd on its own, before any of it is assembled into a model.

---

## Status

Early. Working so far:

- `src/tensor.{h,c}` — flat row-major float32 tensors, no strides or broadcasting
- `src/ops.{h,c}` — linear, LayerNorm, GELU, softmax, fused softmax
  cross-entropy, and multi-head causal self-attention, each forward and backward
- `ref/reference.py` — the PyTorch model, and a full forward/backward dump
- `ref/units.py` — one isolated oracle case per operation
- `tests/` — 10 tests, no framework

Not yet built: the assembled model, AdamW, the tokenizer, and the training loop.

## Verification

Every operation is compared against PyTorch at absolute 1e-5 or relative 1e-4,
either sufficing. Both sides are float32 and accumulate in different orders, so
exact agreement is not available; these bounds are about two orders of magnitude
tighter than any real bug produces.

A test suite that passes proves nothing on its own, so the suite is checked by
breaking the implementation on purpose. **Twenty-five mutations, twenty-five
caught** — though not on the first attempt; see below.

| Mutation | Result |
|---|---|
| LayerNorm backward, drop both mean-correction terms | caught |
| LayerNorm backward, drop only the variance term | caught |
| LayerNorm backward, drop only the mean term | caught |
| LayerNorm forward, forget the weight scale | caught |
| GELU backward, drop the density term | caught |
| GELU forward, tanh approximation instead of erf | caught |
| Softmax backward, keep only the diagonal term | caught |
| Softmax forward, remove the max subtraction | caught (8 checks) |
| Linear backward, transpose the weight gradient | caught |
| Linear backward, forget the bias gradient | caught |
| Cross-entropy backward, forget the 1/rows scale | caught |
| Cross-entropy backward, forget to subtract the target | caught |
| Attention, leak the causal mask | caught |
| Attention, transpose q and k in the score | caught |
| Attention, drop the 1/sqrt(head_dim) scale, forward | caught |
| Attention, drop the 1/sqrt(head_dim) scale, backward | caught |
| Attention backward, drop the softmax dot term | caught |
| Attention backward, swap the dq and dk accumulation | caught |
| Attention backward, index dv by att[j][i] | caught |
| Attention, forget to zero the masked weights | caught *(2nd attempt)* |
| Attention, forget to zero the output accumulator | caught *(2nd attempt)* |

The LayerNorm ones matter most. Its backward pass has two terms that exist only
because the mean and variance are themselves functions of every element in the
row, and dropping them is the single most common error in a hand-written
transformer. A model with that bug still trains.

### Two mutations that survived, and what they exposed

Both concerned buffers the forward pass is supposed to overwrite. The tests
allocated them with `calloc` and so received clean zeroed memory every time,
which is exactly the state a training loop never provides — those buffers are
reused every step. Forgetting to zero the masked upper triangle of the attention
weights, or to reset the output accumulator, therefore worked on the first
forward pass and corrupted every one after it.

The tests now **poison every scratch buffer** with a sentinel before calling the
forward pass. Both mutations are caught, and it is worth noting that neither was
found by comparing against PyTorch: autograd was in perfect agreement, because
the oracle harness had the same clean-buffer assumption. Only deliberately
breaking the code surfaced them.

### What the masked attention gradient disagrees on

`datt` is the one tensor where this implementation and PyTorch legitimately
differ. Autograd forms the full `T x T` product, so it computes a nonzero
gradient for masked positions, where the attention weight is itself zero. This
implementation never forms the upper triangle at all.

Both are correct, because those entries are dead: softmax backward multiplies
each by its own attention weight, which is zero, so nothing they contain ever
reaches a parameter. The evidence is that `dx`, `dqkv_w` and `dqkv_b` all agree
with autograd to 1e-5 regardless. The test compares the causal region and
separately asserts the upper triangle is untouched, rather than quietly widening
a tolerance until the disagreement fits.

### Design notes

**Gradients accumulate, they do not overwrite.** A parameter used in several
places — tied embeddings, or a weight shared across positions — must collect
gradient from each use, and that only works if every backward function adds.
`test_gradients_accumulate_rather_than_overwrite` calls one twice and checks the
result doubles. The cost is that forgetting to zero produces a slow, plausible
divergence rather than a crash, so the training loop zeroes in exactly one place.

**Cross-entropy is fused with the softmax.** Their composition differentiates to
`(p - onehot) / n`, which avoids both the division by a near-zero probability and
the cancellation that a separate softmax and log produce for a confident wrong
prediction.

**Softmax subtracts the row maximum.** Mathematically a no-op, numerically
essential: `expf(89.0f)` is already infinity in float32, and attention logits
reach that range through ordinary training rather than adversarial input.
`test_softmax_survives_logits_that_would_overflow` uses logits of ±300.

**LayerNorm takes variance from centred values**, not `E[x²] − E[x]²`. The
latter is one pass cheaper and subtracts two large nearly-equal numbers, losing
most of the significand when the mean is large relative to the spread.

**Causality is enforced by not computing the future, not by masking it.** The
scores above the diagonal are never formed, which halves the work and makes a
leak structurally impossible rather than dependent on remembering to add `-inf`.
The entries are still written as zero, because the backward pass reads whole
rows. Two tests cover it: one asserts the weights are zero above the diagonal
and each row sums to one, the other edits a later token and checks that every
earlier output is bit-identical — the behavioural consequence, which the
structural check alone would not catch if values were mixed after the softmax.

**A leaking causal mask is the failure that looks most like success.** The model
reads the token it is being asked to predict, so training loss collapses toward
zero while generated text stays garbage. Every number printed during training
improves.

**Exact GELU, not the tanh approximation.** The two differ by about 1e-3, three
orders of magnitude above the tolerance these tests run at.

**The oracle is float32, not float64.** A float64 reference would be more
precise than the thing it is checking, and the tolerances would then be
measuring the C side's accumulation order rather than its correctness.

**No file I/O in `src/`.** The library has no stdio dependency, so the same
objects compile for a freestanding target. Loading oracle dumps lives in
`tests/`.

## Layout

```
src/        tensor.{h,c}  ops.{h,c}                    the library
ref/        reference.py  units.py                     PyTorch oracle
tests/      oracle.{h,c}  test.h  test_ops.c           host only
data/       generated oracle dumps, not committed
```

## Build and run

Requires a C99 compiler, `make`, and Python 3 with PyTorch for the oracle.

```
make oracle     # regenerate data/*.bin from ref/*.py
make test       # 10 tests
```

The oracle dumps are generated rather than committed — they are derived from
`ref/*.py`, and a binary blob in the history is something nobody can review.

## License

MIT.

# GPT in C

[![tests](https://github.com/ElliottPurdue/gpt-in-c/actions/workflows/ci.yml/badge.svg)](https://github.com/ElliottPurdue/gpt-in-c/actions/workflows/ci.yml)

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
- `src/model.{h,c}` — the assembled GPT: embeddings, pre-norm blocks, LM head,
  forward and backward
- `src/optim.{h,c}` — AdamW with decoupled weight decay, and gradient clipping
- `src/tokenizer.{h,c}` — character-level vocabulary, encode and decode
- `train.c` — data loading, batching, the training loop, and sampling
- `ref/reference.py` — the PyTorch model, and a full forward/backward dump
- `ref/units.py` — one isolated oracle case per operation
- `tests/` — 25 tests, no framework

The full model agrees with PyTorch on the loss, the logits, and **every one of
its 30,144 parameter gradients**, to 2e-5 absolute or 2e-4 relative.

**It trains.** On the repository's own source and prose as a corpus — 146,501
bytes, 105 distinct characters — a 3-layer, width-96, context-64 model of
362,016 parameters:

```
  expected initial loss  4.6540  (ln 105)

  step     1   train 4.6442   val 4.3820   |grad| 4.510
  step    50   train 2.8486   val 3.0872   |grad| 1.382
  step   100   train 2.5643   val 2.9018   |grad| 0.969
  step   200   train 2.5997   val 2.8195   |grad| 1.086
  step   300   train 2.0048   val 2.7374   |grad| 1.505

  300 steps in 187.0 s, 1780 tokens/s
```

(That run predates the matmul work below; the same 300 steps now take 132.8 s at
2,534 tokens/s, with identical losses.)

Sampling at temperature 0.8 after those 300 steps:

```
#dibes Nl Nize_o et);        ept thand ig tear = cens d ourwer s te sur/
ssemels ctelonthe the be in nf ye hera t ans t alo t capr ifemarg onamemrcrall
thamuan        * = izent = cononor & conde ang
```

Gibberish, but structured gibberish, and the structure is the evidence. It opens
with `#` like a preprocessor directive, places `);`, `*` and `&` plausibly,
produces `= cens` and `= izent` assignment shapes, reproduces the source's
eight-space indentation runs, and spells real English words from the comments —
`the`, `be`, `in`, `and`. Nothing above the character and short-word level, which
is what 300 steps on a 362K-parameter model buys.

For scale, 2.74 nats is about 3.95 bits per character; a well-trained character
model on English runs nearer 1 to 1.5. The loss was still falling when the run
stopped.

The first loss lands on `ln(vocab_size)` to three decimal places, which is where
an untrained model has to start: uniform over the vocabulary. Printing the
expected value next to it turns the first step into a check on the
initialisation rather than an unanchored number.

By step 300 training loss has pulled away from validation — 2.00 against 2.74 —
which is overfitting beginning, and expected with 362K parameters against 132K
training tokens. It is visible only because the validation split is contiguous;
see below.

### Making it faster, and proving it still computes the same thing

The naive matmul was the entire cost. Restructuring it is worth **1.42x** on the
full 300-step run: 1,784 to 2,534 tokens/s, 187.0 s down to 132.8 s.

Shorter 30-step runs, used to compare build flags:

| build | tokens/s | training loss at step 30 |
|---|---|---|
| naive triple loop, `-O2` | 1,653 | 2.9605 |
| blocked, `-O2` | 1,965 | 2.9605 |
| blocked, `-O3` | **2,572** | 2.9605 |

(Short runs measure high or low by 5-10% depending on warm-up, which is why the
headline figure comes from the 300-step runs rather than this table.)

**The loss column is the point.** Every value is bit-identical, because the
summation order never changed — and that claim is checkable rather than
rhetorical, since training is deterministic. Across all 300 steps, every
training loss, validation loss and gradient norm matches the pre-optimisation
run exactly:

```
step     1   train 4.6442   val 4.3820   |grad| 4.510      before and after
step   150   train 2.2869   val 2.8396   |grad| 1.685      before and after
step   300   train 2.0048   val 2.7374   |grad| 1.505      before and after
```

An optimisation that quietly altered the arithmetic would move that curve.

**What was actually slow.** Not the arithmetic. The naive loop walks the whole
weight matrix once per row, and the qkv projection's weights are 110 KB against
a 32 KB L1 — so 1,024 rows cost about 113 MB of memory traffic to do 28 MFLOP of
work. Processing rows in blocks of eight reuses each weight element eight times
before evicting it. The backward pass splits into three loops so each can be
blocked for the array it writes, which changes no summation order either: `dx`
still accumulates over outputs ascending, `dW` and `db` over rows ascending,
exactly as when the loops were nested.

**What did not help.** `-march=native` was *slower* (2,456), and gcc 6.3.0 on
this 32-bit toolchain does not vectorise the inner loop usefully. It would also
have enabled FMA, which fuses a multiply and add into one rounding step and so
would have changed results — losing the bit-identical property for a slowdown.

`-O3` also paid for itself twice: its stronger analysis caught a genuine bug in
the oracle loader, where a short read left a variable unwritten and the error
path printed it. `-O2` never noticed.

Threading is not done. OpenMP needs a pthread-capable toolchain, and the one
here is 32-bit MinGW 6.3.0 without one. Single-threaded gains transfer to the
microcontroller targets anyway, where there are no threads to use.

## Verification

Every operation is compared against PyTorch at absolute 1e-5 or relative 1e-4,
either sufficing. Both sides are float32 and accumulate in different orders, so
exact agreement is not available; these bounds are about two orders of magnitude
tighter than any real bug produces.

A test suite that passes proves nothing on its own, so the suite is checked by
breaking the implementation on purpose. That check is scripted rather than
described — `make mutate` applies each bug in turn, rebuilds, runs the suite,
restores the file, and exits non-zero if anything survives. CI runs it on every
push, so the claim below is verified rather than asserted. **Thirty-eight mutations, thirty-eight caught,
zero survivors** — though not all on the first attempt; see below.

Ten of those target the assembled model rather than individual operations:
dropping either residual connection, dropping the position embedding, post-norm
instead of pre-norm ordering, losing either residual path in the backward pass,
walking the layers forward instead of in reverse, and using the wrong
LayerNorm's saved statistics.

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
| AdamW, omit the bias correction | caught |
| AdamW, couple weight decay into the gradient | caught |
| AdamW, drop the second-moment normalisation | caught |
| Gradient clipping, scale by the norm instead of the limit | caught |
| Tokenizer, index the vocabulary by byte value | caught |
| Tokenizer, decode through the wrong table | caught |
| Linear forward, forget the bias | caught |

The full list lives in `tools/mutate.py`, and `make mutate` runs it in about a
minute. Every mutation must still **compile**: `-Werror` rejects an unused
variable, so deleting a term outright often fails the build rather than the
tests — which looks identical to being caught and is not. Terms are multiplied
by zero instead, and anything that fails to compile is reported as inconclusive
rather than counted as a pass. One mutation was scored that way on the first run
and had to be rewritten.

The LayerNorm ones matter most. Its backward pass has two terms that exist only
because the mean and variance are themselves functions of every element in the
row, and dropping them is the single most common error in a hand-written
transformer. A model with that bug still trains.

### A second, independent check on the gradients

The oracle is precise but shares an assumption with the code it checks: both
implement the same architecture from the same description, so a *misreading of
the architecture* would be reproduced on both sides and agree perfectly.

So the gradients are also checked against the definition of a derivative —
perturb a parameter, measure how the loss actually moves, compare. That check
knows nothing about transformers and cannot share the error.

It is deliberately coarse. A central difference has truncation error growing
with `eps²`, while float32 cancellation error grows as the loss difference
shrinks, and the two squeeze the usable range of `eps` from both sides. The
magnitude floor is derived rather than guessed: the difference moves the loss by
about `2·eps·grad`, float32 resolves a loss near 4.3 to roughly 5e-7, and
requiring several hundred times that gives `grad > 1e-2`. Below it the quotient
is mostly noise — a gradient of 1e-3 shifts the loss by 2e-5, which carries
barely two significant digits, and comparing that at 5% fails on rounding alone.

Setting the floor at 1e-3 initially produced exactly that: one sampled parameter
5.9% off, with PyTorch already confirming the same gradient to 2e-5. The fix was
the arithmetic above, not a wider tolerance.

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

**The build forces real float32 arithmetic**, with `-msse2 -mfpmath=sse`. This
is a 32-bit toolchain, where gcc defaults to the x87 unit and `FLT_EVAL_METHOD`
is 2: every float expression is evaluated in 80 bits and rounded to 32 only when
stored. That makes the library quietly more accurate than the float32 it claims
to be, hides error a real single-precision FPU would show, and — the symptom
that exposed it — breaks exact comparisons, since a value in memory and the same
value in a register are not bit-identical. A clipping test failed on
`grads[0] == 0.3f` while the bytes were provably unchanged.

The targets this code is written for, Cortex-M and Xtensa, have genuine 32-bit
FPUs and no excess precision, so the host build is pinned to match them. Every
gradient still agrees with PyTorch to 2e-5 without the extended-precision
crutch, which is a stronger result than the one measured before.

**The oracle is float32, not float64.** A float64 reference would be more
precise than the thing it is checking, and the tolerances would then be
measuring the C side's accumulation order rather than its correctness.

**The validation split is contiguous, not random.** A character model on a small
corpus will memorise, so training loss alone says nothing about whether it
learned anything. But sampling held-out windows at random from the same text
leaves each one overlapping its training neighbours by up to `block_size - 1`
characters, which leaks the answer: validation then tracks training no matter how
badly the model overfits, and the metric that exists to detect memorisation is
the one memorisation defeats. The last 10% is held out as a single block.

**Training is deterministic.** Two runs of the same command produce identical
losses at every step, bit for bit — the initialisation is a seeded xorshift and
the batch sampler is seeded too, with nothing reading the clock or the host's
`rand()`. That matters for what comes next: a regression introduced while
optimising the matmul shows up as a changed loss curve rather than disappearing
into run-to-run noise.

**The corpus is the repository itself.** Its own source, tests and prose,
concatenated by a Makefile target. No download, fully reproducible, and C gives a
character model plenty of structure to learn — matched braces, indentation,
comment delimiters, identifier conventions.

**No file I/O in `src/`.** The library has no stdio dependency, so the same
objects compile for a freestanding target. Loading oracle dumps lives in
`tests/`.

## Layout

```
src/        tensor.{h,c}  ops.{h,c}                    the library
ref/        reference.py  units.py                     PyTorch oracle
tests/      oracle.{h,c}  test.h  test_*.c             host only
tools/      mutate.py                                the mutation suite
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

# GPT in C

[![tests](https://github.com/ElliottPurdue/gpt-in-c/actions/workflows/ci.yml/badge.svg)](https://github.com/ElliottPurdue/gpt-in-c/actions/workflows/ci.yml)

A transformer implemented from scratch in dependency-free C99: forward pass
*and* hand-derived backward pass, with PyTorch used as a numerical oracle
rather than as a framework.

The forward pass of a transformer is short enough to write from the paper. The
backward pass is where the understanding is, and autograd makes it free, so an
implementation that writes only the forward half demonstrates nothing that
reading the paper would not. Every gradient here is derived by hand and checked
against `torch.autograd` element by element.

**Why an oracle.** An incorrect gradient almost never announces itself. It
usually still decreases the loss, just more slowly, so "it trains" is not
evidence of correctness. It is the failure mode. Each operation is therefore
pinned against autograd on its own, before any of it is assembled into a model.

---

## Status

Early. Working so far:

- `src/tensor.{h,c}`: flat row-major float32 tensors, no strides or broadcasting
- `src/ops.{h,c}`: linear, LayerNorm, GELU, softmax, fused softmax
  cross-entropy, and multi-head causal self-attention, each forward and backward
- `src/model.{h,c}`: the assembled GPT: embeddings, pre-norm blocks, LM head,
  forward and backward
- `src/optim.{h,c}`: AdamW with decoupled weight decay, and gradient clipping
- `src/tokenizer.{h,c}`: character-level vocabulary, encode and decode
- `train.c`: data loading, batching, the training loop, and sampling
- `ref/reference.py`: the PyTorch model, and a full forward/backward dump
- `ref/units.py`: one isolated oracle case per operation
- `tests/`: 25 tests, no framework

The full model agrees with PyTorch on the loss, the logits, and **every one of
its 30,144 parameter gradients**, to 2e-5 absolute or 2e-4 relative.

**It trains.** On the repository's own source as a corpus (150,335 bytes, 95
distinct characters), a 3-layer, width-96, context-64 model of 360,096
parameters:

```
  corpus      data/input.txt, 150335 bytes, 95 distinct characters
  fingerprint ee7d483c
  split       135301 train / 15034 validation tokens

  expected initial loss  4.5539  (ln 95)

  step     1   train 4.5547   val 4.2693   |grad| 4.449
  step    50   train 2.7917   val 2.7852   |grad| 0.854
  step   100   train 2.6888   val 2.5911   |grad| 1.007
  step   150   train 2.3092   val 2.5197   |grad| 1.403
  step   200   train 2.4653   val 2.4707   |grad| 1.507
  step   250   train 2.3263   val 2.4086   |grad| 1.283
  step   300   train 2.2246   val 2.3303   |grad| 1.527

  300 steps in 132.4 s, 2543 tokens/s
```

The corpus is built from the repository's source files by `make data/input.txt`,
so it moves when the code does, and the fingerprint is printed so two runs can
be shown to have seen the same bytes rather than assumed to have. Size and
vocabulary alone will not do it: an edit that adds one character and removes
another leaves both unchanged.

It earned its place immediately. The first CI run with it printed one fewer byte
per line than the same commit produced locally, and one fewer distinct character.
The tree checks out CRLF on Windows and LF on the Linux runner, so the corpus
carried one extra byte per line and one extra vocabulary entry, and every figure
below was reproducible only on the machine that produced it. The rule strips
carriage returns now.

With that fixed, the run reproduces across every toolchain it has been built on:

```
                    MinGW 6.3 32-bit    MinGW 16.2 64-bit    GCC 13 x86-64
                    Windows             Windows              Linux

  fingerprint       ee7d483c            ee7d483c             ee7d483c
  FLT_EVAL_METHOD   2, pinned to 0      0                    0
  step   1          4.5547  4.2693      4.5547  4.2693       4.5547  4.2693
  step 300          2.2246  2.3303      2.2246  2.3303       (20-step run)
  throughput        2,543 tok/s         5,929 tok/s          4,421 tok/s
```

Same digits on two architectures and two operating systems, under compilers ten
major versions apart, with only the speed differing. That is what `-msse2
-mfpmath=sse` in the Makefile buys. Only the 32-bit build needs it: it reports
`FLT_EVAL_METHOD` 2, meaning x87 evaluates every float expression at 80 bits and
rounds on store, and pinning it to SSE2 is what makes that column match the
other two. On x86-64 the flag is a no-op, because SSE2 is the baseline there and
`FLT_EVAL_METHOD` is already 0. The comment in the Makefile predicted this
before there was a 64-bit build to check it against.

Sampling at temperature 0.8 after those 300 steps:

```
");  %det foutef (watpt) * + (uti vaREL, ACTC) *          flutensize tut
+++conandi_frat (x.nat.sinenen_fffx., A`-->d))L, T  B {
moomendit ay g = ("nbet +cosize_ff(t)liear
```

Gibberish, but structured gibberish, and the structure is the evidence. It opens
on `");`, which is how a `printf` call ends and not a three-character sequence
reached by accident, then emits `%d`, a real conversion specifier. It produces
`->` and a dotted `x.nat.` member access, opens a brace, writes `= (` in the
shape of an assignment from a call, builds underscore-joined identifiers, and
leaves `T` and `B` standing alone as this code's own dimension variables. `REL`
and `ACTC` have the shape of the uppercase tolerance macros in the test suite.
Nothing above the character and short-token level, which is what 300 steps on a
360K-parameter model buys.

For scale, 2.33 nats is about 3.36 bits per character; a well-trained character
model on English runs nearer 1 to 1.5. Both curves were still falling when the
run stopped.

The first loss lands within 0.001 of `ln(vocab_size)`: 4.5547 against 4.5539. Above is the only direction available, since a randomly
initialised model is not exactly uniform and any departure from uniform costs
cross-entropy. Printing the expected value next to it turns the first step into
a check on the initialisation rather than an unanchored number, and the check is
worth having: a starting loss near zero means the targets have leaked into the
inputs, and one several times larger means the initialisation scale is wrong.

By step 300 training loss has pulled ahead of validation, 2.22 against 2.33. The
gap is real but small, which is the expected shape for 360K parameters against
135K training tokens: enough capacity to begin memorising, not enough steps to
have done much of it yet.

An earlier version of this corpus included README.md, and reported a far wider
gap, 2.00 against 2.74. Most of that was not overfitting. The validation split
is the last 10% of the corpus, taken contiguously, and the README was
concatenated last, so validation was roughly 15 KB of English prose while
training was almost entirely C. The model was being asked to generalise across a
change of language, and the penalty was being read as memorisation. Dropping the
README from the corpus removed both that confound and a worse one: the file
reporting these numbers was part of the data producing them, so correcting a
figure here changed the run it described.

### The 32-bit toolchain is costing 2.3x

The same 300 steps, same corpus, same machine: **2,543 tokens/s** on 32-bit
MinGW GCC 6.3 against **5,929** on 64-bit MinGW GCC 16.2. The 32-bit build has
half the registers, a compiler ten major versions older, and no pthread. The
loss curve is identical on both.

This number replaces a range. Earlier revisions compared the local 32-bit build
against a GitHub runner and read the gap as 1.8x, then as somewhere between 1.4x
and 1.8x once three runs came back at 3,547, 4,421 and 4,538 tokens/s on
identical code. That spread is the runner, not the code: CI runs on shared
hardware with no guarantee about what else is on it, so one measurement from one
samples the machine as much as the program. Two builds on one machine settle in
a single run what three CI runs could not.

Which is also why every matmul figure below is measured locally rather than in
CI, and why they are repeatable.

### Making it faster, and proving it still computes the same thing

The matmul was the entire cost. Restructuring it is worth **1.13x** on the full
300-step run: 2,244 to 2,543 tokens/s, 151.9 s down to 132.4 s.

The version it replaced is still in the tree, behind `-DGPTC_NAIVE_MATMUL`,
copied back verbatim from the commit before the change. `make bench-matmul`
builds both and runs them on the same corpus. Keeping it costs forty lines and
buys the only thing that makes the claims below checkable rather than asserted,
since both of them are claims about a comparison and a comparison needs the
other side present. A paraphrase would not have done: what is being measured is
exactly how those loops are written.

Shorter 30-step runs, used to compare build flags:

| build | tokens/s | training loss at step 30 |
|---|---|---|
| naive triple loop, `-O2` | 1,725 | 3.0251 |
| blocked, `-O2` | 1,810 | 3.0251 |
| naive triple loop, `-O3` | 2,192 | 3.0251 |
| blocked, `-O3` | **2,519** | 3.0251 |

(Short runs measure high or low by 5-10% depending on warm-up, which is why the
headline figure comes from the 300-step runs rather than this table.)

**An earlier revision of this section claimed 1.42x**, comparing 1,784 tokens/s
against 2,534. That was measured across two commits rather than one: the tree
had moved on between the two runs, so the figure credited the blocking with
changes that were not the blocking. Rebuilding both halves from the same tree,
on the same corpus, gives 1.13x. The correction is in the direction that matters,
and it is the reason the naive path is now a build flag instead of a commit
somebody has to go and find.

**The loss column is the point.** Every value is bit-identical, because the
summation order never changed, and that claim is checkable rather than
rhetorical, since training is deterministic. Across all 300 steps, every
training loss, validation loss and gradient norm matches:

```
step     1   train 4.5547   val 4.2693   |grad| 4.449      naive and blocked
step   150   train 2.3092   val 2.5197   |grad| 1.403      naive and blocked
step   300   train 2.2246   val 2.3303   |grad| 1.527      naive and blocked
```

So does the sampled text, byte for byte, which is the stronger statement of the
two. A loss is one number per step and could match by coincidence in the digits
printed; 200 characters drawn from the model's own distribution match only if
every weight does. An optimisation that quietly altered the arithmetic would
move both.

**What was actually slow.** Not the arithmetic. The naive loop walks the whole
weight matrix once per row, and the qkv projection's weights are 110 KB against
a 32 KB L1, so 1,024 rows cost about 113 MB of memory traffic to do 28 MFLOP of
work. Processing rows in blocks of eight reuses each weight element eight times
before evicting it. The backward pass splits into three loops so each can be
blocked for the array it writes, which changes no summation order either: `dx`
still accumulates over outputs ascending, `dW` and `db` over rows ascending,
exactly as when the loops were nested.

**What did not help.** `-march=native` was *slower* (2,455 against 2,519), and gcc 6.3.0 on
this 32-bit toolchain does not vectorise the inner loop usefully. It would also
have enabled FMA, which fuses a multiply and add into one rounding step and so
would have changed results, losing the bit-identical property for a slowdown.

`-O3` also paid for itself twice: its stronger analysis caught a genuine bug in
the oracle loader, where a short read left a variable unwritten and the error
path printed it. `-O2` never noticed.

Threading is not done. OpenMP needs a pthread-capable toolchain, and the one
here is 32-bit MinGW 6.3.0 without one. Single-threaded gains transfer to the
microcontroller targets anyway, where there are no threads to use.


### Threading it, without letting the arithmetic move

The matmul is threaded with OpenMP, under one rule: parallelism may split the
**output** of an accumulation across threads, never the index being accumulated.

That rules out the obvious thing. `#pragma omp parallel for reduction(+:sum)`
over the contraction index is what the pattern-matching answer looks like, and it
would combine per-thread partial sums in whatever order the threads happened to
finish. Floating-point addition is not associative, so the loss curve would move,
and the bit-identical property this whole section rests on would be gone.

Five loops carry a pragma, and the three in the matmul are threaded on three
different axes, because each writes an array that accumulates over a different
index:

| loop | threaded over | accumulates over |
|---|---|---|
| `linear_forward` | `r0`, row blocks | `i` |
| `linear_backward`, `dx` | `r0`, row blocks | `o` |
| `linear_backward`, `dweight` | `o0`, output blocks | `r` |
| `gelu_forward` | `i` | nothing, elementwise |
| `gelu_backward` | `i` | nothing, elementwise |

The `dx` loop is the one that catches people. Its accumulation index is `o`, the
reverse of the forward pass, so the `o` loop is the tempting one to thread and it
has more iterations to hand out. Threading it is wrong twice over: a write race
on `dx`, and a reordered sum. An `atomic` would repair only the first, which is
the trap, because the result would then be correct-looking, race-free, and
different.

`dbias` is left serial and says so. Its outer loop *is* the accumulation index,
so every repair reorders: `atomic` and `critical` leave the arrival order to the
scheduler, and a reduction combines partials in an unspecified order. It is
O(rows x out_features) against the other two at O(rows x in_features x
out_features), well under 1% of the work, so there is nothing to buy.

**Because the partition never splits a sum, the result cannot depend on the
thread count.** That is a claim, so CI checks it: `make verify-determinism` runs
the same training under 1, 2, 3, 4, 7, 16 and 33 threads and compares the output
of each to the serial build **byte for byte**.

It has to be a byte comparison. `make test` runs at 1e-5 absolute and 1e-4
relative, while a summation reordered over a few hundred float32 values moves a
result by roughly 1e-7. A build with the accumulation order destroyed passes all
25 tests. A tolerance-based suite cannot see the difference between this
optimisation and the bug it is one keystroke away from.

Two more guards sit alongside it, because a determinism sweep passes vacuously if
the threaded binary was not actually threaded. CI asserts that exactly five
OpenMP regions are emitted, by name, and greps the **preprocessed** source for
`reduction`, `atomic`, `critical` and `simd` (preprocessed, because the comments
explaining why those are absent contain the words). The trainer prints the thread
count it actually got, and the harness rejects a run that did not get the count
it asked for.

The pragmas are behind `#ifdef _OPENMP` rather than written bare, so a compiler
without `-fopenmp` sees nothing: `-Wall` enables `-Wunknown-pragmas` and
`-Werror` makes it fatal, and the 32-bit MinGW 6.3 build has no OpenMP at all.
Compiled without `-fopenmp`, `src/ops.c` produces assembly byte-identical to the
version before any of this, so the serial build is unchanged code rather than
code that merely behaves the same.

**Measured.** On 16 physical cores, 300 steps, 64-bit GCC 16.2:

| threads | time | tokens/s | vs serial |
|---|---|---|---|
| serial, no `-fopenmp` | 56.7 s | 5,929 | 1.00x |
| 1 | 69.0 s | 4,987 | 0.84x |
| 2 | 51.3 s | 6,572 | 1.11x |
| 4 | 30.7 s | 10,941 | 1.85x |
| 8 | 18.9 s | 17,783 | 3.00x |
| 16 | 15.2 s | 21,979 | 3.71x |
| 32 | 13.9 s | 24,013 | **4.05x** |

Every row above produced the same loss curve, gradient norms and sampled text as
the serial build, byte for byte, across all 300 steps.

**The single-threaded row is the honest part of this table.** Building with
`-fopenmp` and then running on one thread is 16% *slower* than not building with
it at all: 69.0 s against 56.7. The pragmas cost something even when no
parallelism happens, partly fork and join around every region and partly the
optimiser having less freedom inside a loop it must hand to a runtime. Threading
does not become a win until two threads, and the honest baseline for the
speedup is the serial build, not the one-thread run, which would have flattered
it to 4.96x.

**4.05x on 16 cores is not a disappointment, it is arithmetic.** Inverting
Amdahl's law on the 32-thread figure puts the parallel fraction at 0.78, so
about 22% of a step is still serial: attention, layernorm, softmax, the
optimiser, and the `dbias` loop that had to stay ordered. Threading the matmul
and GELU cannot buy more than that ceiling no matter how many cores are added,
which is exactly what the flattening between 16 and 32 threads shows. The next
real gain is in what is still serial, not in more threads.

Scaling is also sublinear well before the ceiling: 3.00x at 8 threads and 3.71x
at 16. `schedule(static)` splits `out_features / OUT_BLOCK` chunks across
threads, and for the projections that is 6 chunks against 16 threads, so most of
them idle. `OUT_BLOCK` appears nowhere in any summation order, which makes it a
free knob under the determinism claim, and lowering it is the obvious next
experiment.

For scale against the figures elsewhere in this file, which come from the 32-bit
MinGW 6.3 build: 2,543 tokens/s there, 5,929 for the same code on 64-bit GCC
16.2, and 24,013 threaded on 32. That is **9.4x** end to end, and the loss curve
never moved.
## Verification

CI runs the suite, the mutation pass, and a short training run on every push,
on a different toolchain from the one used to develop it -- x86-64 GCC 13 rather
than 32-bit MinGW 6.3. Everything below therefore holds on two compilers and two
word sizes, not one.

Every operation is compared against PyTorch at absolute 1e-5 or relative 1e-4,
either sufficing. Both sides are float32 and accumulate in different orders, so
exact agreement is not available; these bounds are about two orders of magnitude
tighter than any real bug produces.

A test suite that passes proves nothing on its own, so the suite is checked by
breaking the implementation on purpose. That check is scripted rather than
described. `make mutate` applies each bug in turn, rebuilds, runs the suite,
restores the file, and exits non-zero if anything survives. CI runs it on every
push, so the claim below is verified rather than asserted. **Thirty-eight mutations, thirty-eight caught,
zero survivors**, though not all on the first attempt; see below.

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
tests, which looks identical to being caught and is not. Terms are multiplied
by zero instead, and anything that fails to compile is reported as inconclusive
rather than counted as a pass. One mutation was scored that way on the first run
and had to be rewritten.

Two more were caught by the same rule years later, when the suite was first run
under gcc 16 instead of gcc 6.3. Both deleted the only *read* of a variable
rather than the only write, and the newer compiler rejects that under
`-Werror=unused-but-set-variable` where the older one did not. The mutations had
stopped testing anything and started reporting "did not compile", on one
toolchain only. Multiplying the term by zero instead fixed both, and the whole
suite is 38 of 38 on gcc 6.3 32-bit and gcc 16.2 64-bit alike. Reporting an
uncompilable mutation as inconclusive rather than caught is what made this
visible at all: as a pass it would have been indistinguishable from working.

A pattern that matches the source in more than one place is reported the same
way. This is not hypothetical: `src/ops.c` now carries the pre-optimisation
matmul behind `-DGPTC_NAIVE_MATMUL`, and one mutation's target line appears
identically in both copies. The harness patches the first match, which was the
copy the default build does not compile, so the binary was unchanged, the suite
passed, and the report blamed the tests for a gap that did not exist. A survivor
that is really a mis-aimed patch is worse than no result at all, so an ambiguous
pattern now refuses to run.

The LayerNorm ones matter most. Its backward pass has two terms that exist only
because the mean and variance are themselves functions of every element in the
row, and dropping them is the single most common error in a hand-written
transformer. A model with that bug still trains.

### A second, independent check on the gradients

The oracle is precise but shares an assumption with the code it checks: both
implement the same architecture from the same description, so a *misreading of
the architecture* would be reproduced on both sides and agree perfectly.

So the gradients are also checked against the definition of a derivative:
perturb a parameter, measure how the loss actually moves, compare. That check
knows nothing about transformers and cannot share the error.

It is deliberately coarse. A central difference has truncation error growing
with `eps²`, while float32 cancellation error grows as the loss difference
shrinks, and the two squeeze the usable range of `eps` from both sides. The
magnitude floor is derived rather than guessed: the difference moves the loss by
about `2·eps·grad`, float32 resolves a loss near 4.3 to roughly 5e-7, and
requiring several hundred times that gives `grad > 1e-2`. Below it the quotient
is mostly noise. A gradient of 1e-3 shifts the loss by 2e-5, which carries
barely two significant digits, and comparing that at 5% fails on rounding alone.

Setting the floor at 1e-3 initially produced exactly that: one sampled parameter
5.9% off, with PyTorch already confirming the same gradient to 2e-5. The fix was
the arithmetic above, not a wider tolerance.

### Two mutations that survived, and what they exposed

Both concerned buffers the forward pass is supposed to overwrite. The tests
allocated them with `calloc` and so received clean zeroed memory every time,
which is exactly the state a training loop never provides, since those buffers are
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
places (tied embeddings, or a weight shared across positions) must collect
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
earlier output is bit-identical. That is the behavioural consequence, which the
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
to be, hides error a real single-precision FPU would show, and breaks exact
comparisons, since a value in memory and the same value in a register are not
bit-identical. That last effect is what exposed it. A clipping test failed on
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
losses at every step, bit for bit. The initialisation is a seeded xorshift and
the batch sampler is seeded too, with nothing reading the clock or the host's
`rand()`. That matters for what comes next: a regression introduced while
optimising the matmul shows up as a changed loss curve rather than disappearing
into run-to-run noise.

**The corpus is the repository itself.** Its own source, tests and prose,
concatenated by a Makefile target. No download, fully reproducible, and C gives a
character model plenty of structure to learn: matched braces, indentation,
comment delimiters, identifier conventions.

**A checkpoint carries its vocabulary.** Token ids mean nothing without the
table that produced them, so the byte table is stored beside the weights; a
checkpoint loaded against a different corpus would otherwise generate confident
nonsense rather than fail. The config is stored ahead of the parameters and
checked on load, since a checkpoint from a different shape is otherwise just
bytes of the right length. The parameters themselves are one contiguous block by
construction, so writing them is a single `fwrite` and no serialiser needs to
know the model's structure.

**The CI log is read with `errors="replace"`.** It embeds the generated sample,
which is raw bytes the model emits in arbitrary order, including fragments of
the multi-byte UTF-8 sequences in the corpus. That is not valid UTF-8, so a
strict read raises `UnicodeDecodeError` on Linux while succeeding on Windows,
whose default codec accepts any byte. Four assertions failed on that decode
rather than on the property each claimed to check, which is the same shape of
error as a test that passes for the wrong reason.

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
make oracle        # regenerate data/*.bin from ref/*.py
make test          # 25 tests against the oracle
make train         # 500 steps on the repository's own source
make mutate        # break the library 38 ways, check the tests notice
make bench-matmul  # the blocked matmul against the one it replaced
make omp           # the same 25 tests, threaded (needs -fopenmp support)
make verify-determinism   # same bytes at 1, 2, 3, 4, 7, 16 and 33 threads
```

`make bench-matmul` builds both implementations from the same tree and runs them
on the same corpus, which is what makes the throughput and the bit-identical
losses reported above reproducible rather than remembered. `STEPS=300` for the
long run.

The oracle dumps are generated rather than committed, since they are derived from
`ref/*.py`, and a binary blob in the history is something nobody can review.

## License

MIT.

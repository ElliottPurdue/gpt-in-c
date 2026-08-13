"""Runs the mutation suite: break the library on purpose, check the tests notice.

A passing test suite is evidence about the tests only if the tests can fail. This
introduces one deliberate bug at a time, rebuilds, runs the suite, and records
whether it was caught -- then restores the file. A mutation that SURVIVES is a
hole in the tests, and is the interesting result.

Each mutation is a real error someone could plausibly write, not a random
character swap. Several are drawn from bugs that actually occurred while building
this: the LayerNorm mean terms, the attention buffers that a training loop reuses
but tests hand over freshly zeroed, and the measurement of a gradient against a
buffer that was never reset.

Every mutation must still COMPILE. `-Werror` rejects an unused variable, so
deleting a term outright often fails the build rather than the tests, which looks
identical to being caught and is not. Terms are therefore multiplied by zero
rather than removed. Anything that fails to compile is reported separately as
inconclusive, never as a pass.

    python tools/mutate.py            # run them all
    python tools/mutate.py --list     # just show them
"""

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent

# MinGW ships the tool as mingw32-make; CI and Unix use make. Picking it here
# keeps the script runnable on both without a wrapper.
MAKE = "mingw32-make" if shutil.which("mingw32-make") else "make"

# (file, description, original, replacement)
MUTATIONS = [
    # ---- LayerNorm ------------------------------------------------------
    ("src/ops.c", "layernorm backward: drop both mean-correction terms",
     "dxr[c] += inv * (g - mean_g - xhat * mean_g_xhat);",
     "dxr[c] += inv * (g - 0.0f * (mean_g + xhat * mean_g_xhat));"),
    ("src/ops.c", "layernorm backward: drop only the mean term",
     "dxr[c] += inv * (g - mean_g - xhat * mean_g_xhat);",
     "dxr[c] += inv * (g - 0.0f * mean_g - xhat * mean_g_xhat);"),
    ("src/ops.c", "layernorm backward: drop only the variance term",
     "dxr[c] += inv * (g - mean_g - xhat * mean_g_xhat);",
     "dxr[c] += inv * (g - mean_g - 0.0f * xhat * mean_g_xhat);"),
    ("src/ops.c", "layernorm forward: forget the weight scale",
     "outr[c] = normalized * weight[c] + bias[c];",
     "outr[c] = normalized * (0.0f * weight[c] + 1.0f) + bias[c];"),

    # ---- GELU -----------------------------------------------------------
    ("src/ops.c", "gelu backward: drop the density term",
     "dx[i] += dout[i] * (cdf + v * pdf);",
     "dx[i] += dout[i] * (cdf + 0.0f * v * pdf);"),
    ("src/ops.c", "gelu forward: tanh approximation instead of erf",
     "out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * SQRT_1_2));",
     "out[i] = 0.5f * x[i] * (1.0f + tanhf(SQRT_2_PI * (x[i] + 0.044715f * x[i] * x[i] * x[i])));"),

    # ---- Softmax --------------------------------------------------------
    ("src/ops.c", "softmax backward: keep only the diagonal term",
     "dxr[c] += outr[c] * (doutr[c] - dot);",
     "dxr[c] += outr[c] * doutr[c];"),
    ("src/ops.c", "softmax forward: remove the max subtraction",
     "float e = expf(xr[c] - maximum);",
     "float e = expf(xr[c] - 0.0f * maximum);"),

    # ---- Linear ---------------------------------------------------------
    ("src/ops.c", "linear backward: transpose the weight gradient",
     "dwo[i] += g * xr[i];",
     "dwo[i] += g * xr[in_features - 1 - i];"),
    ("src/ops.c", "linear backward: forget the bias gradient",
     "dbias[o] += doutr[o];",
     "dbias[o] += 0.0f * doutr[o];"),
    ("src/ops.c", "linear forward: forget the bias",
     "acc[b] = bias ? bias[o] : 0.0f;",
     "acc[b] = bias ? 0.0f * bias[o] : 0.0f;"),

    # ---- Cross entropy --------------------------------------------------
    ("src/ops.c", "crossentropy backward: forget the 1/rows scale",
     "float scale = 1.0f / (float)rows;",
     "float scale = 1.0f;"),
    ("src/ops.c", "crossentropy backward: forget to subtract the target",
     "dr[targets[r]] -= scale;",
     "dr[targets[r]] -= 0.0f * scale;"),

    # ---- Attention ------------------------------------------------------
    ("src/ops.c", "attention: drop the 1/sqrt(head_dim) scale, forward",
     "dot *= scale;",
     "dot *= (0.0f * scale + 1.0f);"),
    ("src/ops.c", "attention: drop the 1/sqrt(head_dim) scale, backward",
     "float dscore = att_row[j] * (datt_row[j] - dot) * scale;",
     "float dscore = att_row[j] * (datt_row[j] - dot) * (0.0f * scale + 1.0f);"),
    ("src/ops.c", "attention backward: drop the softmax dot term",
     "float dscore = att_row[j] * (datt_row[j] - dot) * scale;",
     "float dscore = att_row[j] * datt_row[j] * scale;"),
    ("src/ops.c", "attention backward: swap the dq and dk accumulation",
     "dq[d] += dscore * k[d];\n                        dk[d] += dscore * q[d];",
     "dq[d] += dscore * q[d];\n                        dk[d] += dscore * k[d];"),
    ("src/ops.c", "attention backward: index dv by att[j][i]",
     "dv[d] += att_row[j] * douti[d];",
     "dv[d] += att_head[(size_t)j * T + i] * douti[d];"),
    ("src/ops.c", "attention: leak the causal mask",
     "for (int j = 0; j <= i; ++j) {\n                    const float *k = qkv + qkv_offset(b, j, 1, h, T, C, head_dim);",
     "for (int j = 0; j < T; ++j) {\n                    const float *k = qkv + qkv_offset(b, j, 1, h, T, C, head_dim);"),
    ("src/ops.c", "attention: transpose q and k in the score",
     "return ((size_t)(b * T + t) * 3 + which) * C + (size_t)head * head_dim;",
     "return ((size_t)(b * T + t) * 3 + (which == 0 ? 1 : (which == 1 ? 0 : 2))) * C + (size_t)head * head_dim;"),
    ("src/ops.c", "attention: forget to zero the masked weights",
     "for (int j = i + 1; j < T; ++j) {\n                    att_row[j] = 0.0f;\n                }",
     "for (int j = i + 1; j < T; ++j) {\n                    if (att_row[j] == 0.0f) { att_row[j] = 0.0f; }\n                }"),
    ("src/ops.c", "attention: forget to zero the output accumulator",
     "outi[d] = 0.0f;",
     "outi[d] += 0.0f;"),

    # ---- Model ----------------------------------------------------------
    ("src/model.c", "forward: drop the position embedding",
     "out[i] = token[i] + position[i];",
     "out[i] = token[i] + 0.0f * position[i];"),
    ("src/model.c", "forward: drop the attention residual",
     "residual1[i] = residual[i] + attn_proj[i];",
     "residual1[i] = attn_proj[i];"),
    ("src/model.c", "forward: drop the MLP residual",
     "residual2[i] = residual1[i] + mlp_out[i];",
     "residual2[i] = mlp_out[i];"),
    ("src/model.c", "forward: post-norm instead of pre-norm ordering",
     "attention_forward(attn_out, qkv, att, ln1,",
     "attention_forward(attn_out, qkv, att, residual,"),
    ("src/model.c", "backward: lose the MLP residual gradient path",
     "memcpy(dresidual1, dresidual2, BTC * sizeof(float));",
     "memset(dresidual1, 0, BTC * sizeof(float));"),
    ("src/model.c", "backward: lose the attention residual gradient path",
     "memcpy(dresidual, dresidual1, BTC * sizeof(float));",
     "memset(dresidual, 0, BTC * sizeof(float));"),
    ("src/model.c", "backward: iterate layers forward instead of in reverse",
     "for (int l = L - 1; l >= 0; --l) {",
     "for (int l = 0; l < L; ++l) {"),
    ("src/model.c", "backward: use ln2 statistics for the ln1 gradient",
     "a->ln1_mean + (size_t)l * B * T,\n                           a->ln1_rstd + (size_t)l * B * T,",
     "a->ln2_mean + (size_t)l * B * T,\n                           a->ln2_rstd + (size_t)l * B * T,"),
    ("src/model.c", "backward: skip the token-embedding gradient",
     "dtoken[i] += d[i];",
     "dtoken[i] += 0.0f * d[i];"),
    ("src/model.c", "backward: skip the position-embedding gradient",
     "dposition[i] += d[i];",
     "dposition[i] += 0.0f * d[i];"),

    # ---- Optimizer ------------------------------------------------------
    ("src/optim.c", "adamw: omit the bias correction",
     "float m_hat = opt->m[i] / bias1;\n        float v_hat = opt->v[i] / bias2;",
     "float m_hat = opt->m[i] / (0.0f * bias1 + 1.0f);\n        float v_hat = opt->v[i] / (0.0f * bias2 + 1.0f);"),
    ("src/optim.c", "adamw: couple weight decay into the gradient path",
     "params[i] -= opt->lr * opt->weight_decay * params[i];",
     "params[i] -= 0.0f * opt->lr * opt->weight_decay * params[i];"),
    ("src/optim.c", "adamw: drop the second-moment normalisation",
     "params[i] -= opt->lr * m_hat / (sqrtf(v_hat) + opt->eps);",
     "params[i] -= opt->lr * m_hat / (0.0f * sqrtf(v_hat) + 1.0f);"),
    ("src/optim.c", "clipping: scale by the norm instead of the limit",
     "float scale = max_norm / norm;",
     "float scale = norm / max_norm;"),

    # ---- Tokenizer ------------------------------------------------------
    ("src/tokenizer.c", "tokenizer: index the vocabulary by byte value",
     "tok->to_token[byte] = tok->vocab_size;",
     "tok->to_token[byte] = byte;"),
    ("src/tokenizer.c", "tokenizer: decode through the wrong table",
     "out[i] = (t >= 0 && t < tok->vocab_size) ? tok->to_byte[t] : (unsigned char)'?';",
     "out[i] = (t >= 0 && t < tok->vocab_size) ? (unsigned char)t : (unsigned char)'?';"),
]


def run_tests():
    """Returns (compiled, failures)."""
    result = subprocess.run([MAKE, "test"], cwd=ROOT,
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.strip().endswith("failed checks"):
            return True, int(line.split(",")[1].strip().split()[0])
    return False, -1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for path, why, _, _ in MUTATIONS:
            print(f"  {path:18} {why}")
        return 0

    baseline_compiled, baseline_failures = run_tests()
    if not baseline_compiled or baseline_failures != 0:
        print("the suite does not pass before mutating; fix that first")
        return 2

    backups = {}
    for path, _, _, _ in MUTATIONS:
        if path not in backups:
            source = ROOT / path
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".bak")
            handle.close()
            shutil.copy2(source, handle.name)
            backups[path] = handle.name

    caught = survived = inconclusive = 0
    try:
        for path, why, old, new in MUTATIONS:
            source = ROOT / path
            text = source.read_text(encoding="utf-8")

            if old not in text:
                print(f"  PATTERN MISSING  {why}")
                inconclusive += 1
                continue

            source.write_text(text.replace(old, new, 1), encoding="utf-8")
            compiled, failures = run_tests()
            shutil.copy2(backups[path], source)

            if not compiled:
                print(f"  DID NOT COMPILE  {why}")
                inconclusive += 1
            elif failures > 0:
                print(f"  caught ({failures:2d})       {why}")
                caught += 1
            else:
                print(f"  SURVIVED         {why}   <-- gap in the tests")
                survived += 1
    finally:
        for path, backup in backups.items():
            shutil.copy2(backup, ROOT / path)
            pathlib.Path(backup).unlink(missing_ok=True)
        run_tests()   # leave the tree built and passing

    total = len(MUTATIONS)
    print(f"\n  {caught}/{total} caught, {survived} survived, "
          f"{inconclusive} inconclusive")
    return 1 if (survived or inconclusive) else 0


if __name__ == "__main__":
    sys.exit(main())

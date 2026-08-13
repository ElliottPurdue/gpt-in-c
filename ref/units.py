"""Per-operation oracle: one isolated case per op, forward and backward.

The full-model dump in reference.py catches disagreement but does not localise
it. A wrong LayerNorm gradient and a wrong attention gradient both show up as
"the loss gradient is off", and working backwards from that is slow. These cases
pin each operation on its own, with inputs chosen to exercise the parts that are
easy to get wrong.

Every case supplies an incoming gradient drawn at random rather than ones. A
gradient of all-ones hides bugs: any term that should be scaled by the incoming
gradient still produces the right answer when that gradient is 1, so a missing
multiplication survives the test.
"""

import pathlib

import torch
import torch.nn.functional as F

from reference import write_tensors

SEED = 4242


def case(name, inputs, output, extra_grads=()):
    """Runs backward and returns the tensors this case contributes."""
    incoming = torch.randn_like(output)
    output.backward(incoming)

    out = {f"{name}.out": output, f"{name}.dout": incoming}
    for key, tensor in inputs.items():
        out[f"{name}.{key}"] = tensor
        if tensor.grad is not None:
            out[f"{name}.d{key}"] = tensor.grad
    for key, tensor in extra_grads:
        out[f"{name}.{key}"] = tensor
    return out


def leaf(*shape, scale=1.0):
    return (torch.randn(*shape) * scale).requires_grad_(True)


def build():
    torch.manual_seed(SEED)
    tensors = {}

    # ---- LayerNorm -------------------------------------------------------
    # Inputs are scaled well away from unit variance. A normalisation whose
    # backward forgets the mean or variance terms still looks right when the
    # input is already standardised, which is exactly what randn gives.
    x = leaf(3, 5, 8, scale=4.0)
    weight = leaf(8)
    bias = leaf(8)
    tensors.update(case("layernorm",
                        {"x": x, "weight": weight, "bias": bias},
                        F.layer_norm(x, (8,), weight, bias, eps=1e-5)))

    # ---- GELU ------------------------------------------------------------
    # Spans the saturating tails and the region near zero where the exact and
    # tanh-approximate forms differ most.
    x = leaf(4, 16, scale=3.0)
    tensors.update(case("gelu", {"x": x}, F.gelu(x, approximate="none")))

    # ---- Softmax, last axis ---------------------------------------------
    # The Jacobian is dense: every output depends on every input in the row.
    # An implementation that only handles the diagonal term passes a
    # sum-reduced test and fails this one.
    x = leaf(3, 7, scale=2.0)
    tensors.update(case("softmax", {"x": x}, F.softmax(x, dim=-1)))

    # ---- Linear ----------------------------------------------------------
    # Non-square and non-matching dims throughout, so a transposed matmul
    # cannot silently produce a correctly shaped answer.
    x = leaf(6, 5)
    weight = leaf(9, 5)      # torch stores linear weight as (out, in)
    bias = leaf(9)
    tensors.update(case("linear",
                        {"x": x, "weight": weight, "bias": bias},
                        F.linear(x, weight, bias)))

    # ---- Causal self-attention ------------------------------------------
    # Two heads and four positions: small enough to inspect by hand, large
    # enough that the mask, the head split and the scale all matter. A missing
    # mask shows up here and nowhere else.
    B, T, n_head, head_dim = 2, 4, 2, 3
    C = n_head * head_dim
    x = leaf(B, T, C)
    qkv_w = leaf(3 * C, C)
    qkv_b = leaf(3 * C)

    qkv = F.linear(x, qkv_w, qkv_b)
    q, k, v = qkv.split(C, dim=2)
    q = q.view(B, T, n_head, head_dim).transpose(1, 2)
    k = k.view(B, T, n_head, head_dim).transpose(1, 2)
    v = v.view(B, T, n_head, head_dim).transpose(1, 2)

    att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    mask = torch.tril(torch.ones(T, T)).view(1, 1, T, T)
    att = att.masked_fill(mask == 0, float("-inf"))
    att = F.softmax(att, dim=-1)
    att.retain_grad()
    y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

    tensors.update(case("attention",
                        {"x": x, "qkv_w": qkv_w, "qkv_b": qkv_b}, y,
                        extra_grads=[("att", att)]))
    tensors["attention.datt"] = att.grad

    # ---- Cross entropy ---------------------------------------------------
    # Reduced to a scalar, so this one seeds its own backward rather than
    # taking an incoming gradient.
    logits = leaf(5, 11, scale=2.0)
    targets = torch.randint(0, 11, (5,))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    tensors["crossentropy.logits"] = logits
    tensors["crossentropy.targets"] = targets.float()
    tensors["crossentropy.loss"] = loss.reshape(1)
    tensors["crossentropy.dlogits"] = logits.grad

    # ---- AdamW ------------------------------------------------------------
    # Several steps, not one. The bias-correction terms decay with the step
    # count, so a single step cannot distinguish a correct implementation from
    # one that omits them -- at t=1 the correction is largest and most of the
    # plausible errors still land close. Five steps separates them.
    #
    # A fixed gradient would also hide errors in the second moment, so each step
    # gets a different one.
    torch.manual_seed(SEED + 1)
    param = torch.randn(24) * 0.5
    grads = [torch.randn(24) * 0.1 for _ in range(5)]

    tensors["adamw.initial"] = param.clone()
    for i, g in enumerate(grads):
        tensors[f"adamw.grad{i}"] = g

    work = param.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([work], lr=1e-2, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1)
    for i, g in enumerate(grads):
        optimizer.zero_grad()
        work.grad = g.clone()
        optimizer.step()
        tensors[f"adamw.after{i}"] = work.detach().clone()

    # ---- AdamW, several steps ------------------------------------------
    # One step is not enough to pin this. Adam's bias correction is largest at
    # step 1 and decays, so an implementation that omits it entirely is closest
    # to correct exactly where a single-step test would look. Running four steps
    # and dumping each lets the test follow the correction as it fades.
    torch.manual_seed(SEED + 1)
    weight = torch.randn(6, 4, requires_grad=True)
    start = weight.detach().clone()

    optimizer = torch.optim.AdamW([weight], lr=1e-2, betas=(0.9, 0.999),
                                  eps=1e-8, weight_decay=0.1)

    tensors["adamw.initial"] = start
    for step in range(4):
        # A fixed, deterministic "gradient" rather than a real one: this checks
        # the update rule, and coupling it to a model would only add a way for
        # the test to fail for an unrelated reason.
        grad = torch.sin(start * float(step + 1)) + 0.1 * float(step)
        weight.grad = grad.clone()
        tensors[f"adamw.grad{step}"] = grad
        optimizer.step()
        tensors[f"adamw.after{step}"] = weight.detach().clone()

    return tensors


def main():
    tensors = build()
    out = pathlib.Path(__file__).parent.parent / "data" / "units.bin"
    write_tensors(out, tensors)

    print(f"wrote {out}")
    names = sorted({n.split(".")[0] for n in tensors})
    print(f"  cases    {len(names)}: {', '.join(names)}")
    print(f"  tensors  {len(tensors)}")
    print(f"  bytes    {out.stat().st_size:,}")


if __name__ == "__main__":
    main()

"""PyTorch reference model, used as a numerical oracle for the C implementation.

This file is not the project. The project is the C code in src/; this exists so
that every number the C code produces can be checked against an implementation
whose correctness is not in question.

The forward pass of a transformer is easy to write and easy to get subtly wrong
in ways that still train -- a transposed index, a missing scale, a mask applied
one position late. The backward pass is worse, because an incorrect gradient
usually still decreases the loss, just slower, and there is nothing to notice.
So the C implementation is checked layer by layer, forward *and* backward,
against autograd, rather than being judged on whether its loss goes down.

What gets dumped:
  - every parameter, so the C model starts from identical weights
  - the input batch and the loss
  - every intermediate activation
  - every parameter gradient and every activation gradient

Tensors are written in a flat binary format that C can read with fread and no
parsing. Layout is row-major and contiguous, matching the C side exactly.
"""

import argparse
import pathlib
import struct

import torch
import torch.nn as nn
import torch.nn.functional as F

# Deliberately tiny. The oracle's job is to be exactly checkable, not to be a
# good language model -- small dimensions make a disagreement easy to localise,
# and every head, position and channel gets compared individually.
DEFAULT_CONFIG = dict(
    vocab_size=65,      # character-level, matching the Shakespeare corpus
    block_size=16,      # context length
    n_layer=2,
    n_head=4,
    n_embd=32,
)

MAGIC = b"GPTC"
VERSION = 1


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    Written with explicit reshapes and an explicit mask rather than
    scaled_dot_product_attention, because the C implementation has to reproduce
    each step and a fused kernel would hide the intermediates being compared.
    """

    def __init__(self, config):
        super().__init__()
        assert config["n_embd"] % config["n_head"] == 0
        self.n_head = config["n_head"]
        self.n_embd = config["n_embd"]
        self.head_dim = self.n_embd // self.n_head

        # One projection producing q, k and v, split afterwards. This is the
        # usual layout and the C code indexes into it the same way.
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        block = config["block_size"]
        self.register_buffer(
            "mask", torch.tril(torch.ones(block, block)).view(1, 1, block, block))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # The 1/sqrt(head_dim) scale keeps the logits' variance independent of
        # head size; without it softmax saturates as dimensions grow and the
        # gradients vanish.
        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embd = config["n_embd"]
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        # Exact GELU, not the tanh approximation. The two differ by ~1e-3, which
        # is far above the tolerance the gradient checks run at, so the C side
        # has to implement the same one.
        return self.c_proj(F.gelu(self.c_fc(x), approximate="none"))


class Block(nn.Module):
    """Pre-norm residual block: x + attn(ln(x)), then x + mlp(ln(x))."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config["n_embd"])
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config["n_embd"])
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config["vocab_size"], config["n_embd"])
        self.wpe = nn.Embedding(config["block_size"], config["n_embd"])
        self.blocks = nn.ModuleList([Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"])
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)

        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss


# --------------------------------------------------------------------------
# Binary dump
# --------------------------------------------------------------------------

def write_tensors(path, tensors):
    """Flat, self-describing, and readable with fread.

    Format:
        magic "GPTC", uint32 version, uint32 count
        per tensor:
            uint32 name_length, name bytes (no terminator)
            uint32 ndim, ndim x uint32 dims
            float32 data, row-major, contiguous
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(MAGIC)
        handle.write(struct.pack("<II", VERSION, len(tensors)))
        for name, tensor in tensors.items():
            array = tensor.detach().contiguous().float().cpu().numpy()
            encoded = name.encode("utf-8")
            handle.write(struct.pack("<I", len(encoded)))
            handle.write(encoded)
            handle.write(struct.pack("<I", array.ndim))
            handle.write(struct.pack(f"<{array.ndim}I", *array.shape))
            handle.write(array.tobytes(order="C"))
    return path


def collect(model, idx, targets):
    """One forward and backward pass, capturing everything on the way through.

    Activations are captured with hooks rather than by re-running pieces of the
    model, so what is dumped is exactly what the loss was computed from.
    """
    activations = {}

    def save(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            tensor.retain_grad()
            activations[name] = tensor
        return hook

    handles = [model.wte.register_forward_hook(save("act.wte")),
               model.wpe.register_forward_hook(save("act.wpe")),
               model.ln_f.register_forward_hook(save("act.ln_f"))]
    for i, block in enumerate(model.blocks):
        handles.append(block.ln_1.register_forward_hook(save(f"act.block{i}.ln_1")))
        handles.append(block.attn.register_forward_hook(save(f"act.block{i}.attn")))
        handles.append(block.ln_2.register_forward_hook(save(f"act.block{i}.ln_2")))
        handles.append(block.mlp.c_fc.register_forward_hook(save(f"act.block{i}.mlp.c_fc")))
        handles.append(block.mlp.register_forward_hook(save(f"act.block{i}.mlp")))
        handles.append(block.register_forward_hook(save(f"act.block{i}.out")))

    logits, loss = model(idx, targets)
    logits.retain_grad()
    model.zero_grad(set_to_none=False)
    loss.backward()

    for handle in handles:
        handle.remove()

    out = {}
    for name, parameter in model.named_parameters():
        out[f"param.{name}"] = parameter
        out[f"grad.{name}"] = parameter.grad
    for name, tensor in activations.items():
        out[name] = tensor
        if tensor.grad is not None:
            out[f"d{name}"] = tensor.grad
    out["input.idx"] = idx.float()
    out["input.targets"] = targets.float()
    out["act.logits"] = logits
    out["dact.logits"] = logits.grad
    out["loss"] = loss.reshape(1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # float64 would make the oracle more precise than the thing it is checking.
    # The C side is float32, so the reference is too, and the tolerances below
    # are set against float32 accumulation error rather than against exactness.
    torch.set_default_dtype(torch.float32)

    config = dict(DEFAULT_CONFIG)
    model = GPT(config)
    model.eval()   # no dropout anywhere, but pinned so the dump is reproducible

    T = config["block_size"]
    idx = torch.randint(0, config["vocab_size"], (args.batch, T))
    targets = torch.randint(0, config["vocab_size"], (args.batch, T))

    tensors = collect(model, idx, targets)

    here = pathlib.Path(__file__).parent
    out = pathlib.Path(args.out) if args.out else here.parent / "data" / "oracle.bin"
    write_tensors(out, tensors)

    total = sum(p.numel() for p in model.parameters())
    print(f"wrote {out}")
    print(f"  config      {config}")
    print(f"  parameters  {total:,}")
    print(f"  batch       {args.batch} x {T}")
    print(f"  loss        {tensors['loss'].item():.8f}")
    print(f"  tensors     {len(tensors)}")
    print(f"  bytes       {out.stat().st_size:,}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cross-check examples/flash_attn.vc against PyTorch's FlashAttention kernels.

The .vc demo verifies itself against its own host reference, which shares the
same author and the same assumptions. This script closes that loop against an
independent implementation: it reads the tensors the demo dumps and asks
torch.nn.functional.scaled_dot_product_attention to compute the same attention.

SDPA is pinned to two different backends, and they are genuinely different
code — verified by profiling, not assumed:

  FLASH_ATTENTION   aten::_scaled_dot_product_flash_attention_for_cpu on a
                    CPU-only host, the CUDA FA kernel's FlashAttentionKernel
                    on a GPU host. Tiled, online softmax — the same algorithm
                    as examples/flash_attn.vc, implemented independently.
  MATH              aten::_scaled_dot_product_attention_math — materializes
                    the N x N scores and softmaxes them in one shot. No tiling,
                    no running max, no rescale, so it is a ground truth that
                    does not share the algorithm under test.

Usage:

    python3 examples/compare_flash_attn.py /tmp/fa_tensors

where /tmp/fa_tensors was produced by:

    vcc examples/flash_attn.vc -o /tmp/fa
    /tmp/fa --dump /tmp/fa_tensors

Each `causal<N>.bin` in the directory is checked separately. Four comparisons
are reported per file:

  torch-vs-kernel   PyTorch FA vs the VC kernel's output   <- the actual claim
  torch-vs-hostref  PyTorch FA vs the demo's host reference
  kernel-vs-hostref the demo's own self-check, for scale
  torch-vs-math     PyTorch FA vs PyTorch's non-tiled math kernel

The expected magnitudes matter: the VC kernel accumulates in float32 while
both references accumulate in double, so ~3e-7 is the float rounding floor
here, not a defect. The assertions use 1e-5, an order of magnitude above that
floor and two below the tolerance the .vc demo applies to itself.

Requires torch. Exits 0 if every file passes, 1 otherwise. On a CPU-only
torch build everything still runs; torch.cuda.is_available() is printed so the
reader can tell which PyTorch kernel they got.
"""

import argparse
import os
import struct
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MAGIC = b"VCFA"
N_META = 4          # NHEAD, SEQ, HDIM, causal — all written as float32
N_BLOBS = 5         # Q, K, V, kernel O, host-reference O
BLOB_NAMES = ("q", "k", "v", "o_kernel", "o_hostref")

# Float rounding floor is ~1e-7 for these magnitudes; 1e-5 is an order of
# magnitude of headroom and still far tighter than the demo's own 1e-4.
TOL = 1.0e-5


def load(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:4] != MAGIC:
        raise ValueError(f"{path}: bad magic {raw[:4]!r}, not a VC flash_attn dump")
    if (len(raw) - 4) % 4:
        raise ValueError(f"{path}: body size {len(raw) - 4} is not a multiple of 4")

    # Read the whole body as float32, then slice. The header is four floats.
    body = struct.unpack(f"<{(len(raw) - 4) // 4}f", raw[4:])
    nhead, seq, hdim, causal = (int(v) for v in body[:N_META])
    elems = nhead * seq * hdim
    payload = body[N_META:]
    if len(payload) != N_BLOBS * elems:
        raise ValueError(
            f"{path}: expected {N_BLOBS} blobs of {elems} floats "
            f"({nhead}x{seq}x{hdim}), got {len(payload) // elems} blobs"
        )

    # Row-major [NHEAD][SEQ][HDIM] on disk; torch wants [batch, heads, seq, dim].
    # One batch here — NHEAD is the number of independent (batch x head) slices.
    shape = (1, nhead, seq, hdim)
    return {
        name: torch.tensor(payload[i * elems:(i + 1) * elems],
                           dtype=torch.float32).reshape(shape)
        for i, name in enumerate(BLOB_NAMES)
    }, causal, dict(nhead=nhead, seq=seq, hdim=hdim)


def torch_flash_attention(q, k, v, causal):
    """Run SDPA, pinned to the FlashAttention backend so it can't silently
    fall back to the math kernel (which would make this check circular)."""
    mask = None
    if causal:
        seq = q.shape[-2]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def torch_math_attention(q, k, v, causal):
    """Double-precision math-kernel reference — an independent ground truth
    for the case where the FlashAttention kernel itself is the thing in doubt."""
    mask = None
    if causal:
        seq = q.shape[-2]
        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                              attn_mask=mask)


def worst(a, b):
    return (a.double() - b.double()).abs().max().item()


def check(path, verbose, tol):
    tensors, causal, dims = load(path)
    q, k, v = tensors["q"], tensors["k"], tensors["v"]
    o_kernel, o_hostref = tensors["o_kernel"], tensors["o_hostref"]

    o_torch = torch_flash_attention(q, k, v, causal)
    o_math = torch_math_attention(q, k, v, causal)

    results = {
        "torch-vs-kernel": worst(o_torch, o_kernel),
        "torch-vs-hostref": worst(o_torch, o_hostref),
        "kernel-vs-hostref": worst(o_kernel, o_hostref),
        "torch-vs-math": worst(o_torch, o_math),
    }

    label = f"{os.path.basename(path)} causal={causal} " \
            f"({dims['nhead']}x{dims['seq']}x{dims['hdim']})"
    ok = True
    for name, err in results.items():
        # torch-vs-math is a sanity check on PyTorch's own kernel, not on VC.
        limit = tol if name != "torch-vs-math" else 1.0e-3
        status = "ok" if err <= limit else "FAIL"
        if err > limit:
            ok = False
        if verbose or err > limit:
            print(f"  {name:<18} max_abs_err={err:.3e}  [{status}]")

    print(f"{label}: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumpdir", help="directory written by `flash_attn --dump DIR`")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print every comparison, not just failures")
    ap.add_argument("--tol", type=float, default=TOL,
                    help=f"max abs error allowed (default {TOL:g})")
    args = ap.parse_args()

    if not os.path.isdir(args.dumpdir):
        print(f"error: {args.dumpdir} is not a directory", file=sys.stderr)
        return 2
    files = sorted(f for f in os.listdir(args.dumpdir) if f.endswith(".bin"))
    if not files:
        print(f"error: no .bin dumps in {args.dumpdir}", file=sys.stderr)
        return 2

    print(f"torch {torch.__version__} — "
          f"cuda_available={torch.cuda.is_available()}")
    ok = True
    for name in files:
        if not check(os.path.join(args.dumpdir, name), args.verbose, args.tol):
            ok = False

    print("compare_flash_attn: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

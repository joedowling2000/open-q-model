#!/usr/bin/env python3
"""llama.cpp's convert_lora_to_gguf.py, fixed for Qwen3.5 DeltaNet LoRA.

Qwen3.5-27B's linear-attention layers have 16 key heads and 48 value heads. The
GGUF converter reorders value heads from grouped to tiled order: rows of
in_proj_qkv (V part) and in_proj_z, and columns of out_proj. On a LoRA pair it
tries to do that with a reshape that changes the row size, which the LoRA tensor
wrapper cannot express, and it raises NotImplementedError. The 0.8B has equal
head counts, so the reorder never ran in testing.

The fix is exact, not an approximation. The update is W = B @ A. Permuting
W's rows is permuting B's rows, and permuting W's columns is permuting A's
columns. So the reorder is applied to the factor that owns that dimension.

    python scripts/train/convert_lora.py <adapter-dir> --base <hf-snapshot> \\
        --outfile out.gguf --outtype f16          # same arguments as upstream
    python scripts/train/convert_lora.py --selftest
"""

from __future__ import annotations

import runpy
import sys

LLAMA = "/home/joedowling/Projects/serving/llama.cpp"
sys.path.insert(0, LLAMA)

import torch  # noqa: E402
from conversion import qwen  # noqa: E402

_orig = qwen._LinearAttentionVReorderBase._reorder_v_heads


def _reorder_v_heads(tensor, dim, num_k_heads, num_v_per_k, head_dim):
    if not hasattr(tensor, "_lora_A"):
        return _orig(tensor, dim, num_k_heads, num_v_per_k, head_dim)
    A, B = tensor._lora_A, tensor._lora_B          # W = B @ A; A (rank, in), B (out, rank)
    if dim < 0:
        dim += 2
    if dim == 0:                                     # rows of W live in B
        B = _orig(B, 0, num_k_heads, num_v_per_k, head_dim)
    elif dim == 1:                                   # columns of W live in A
        A = _orig(A, 1, num_k_heads, num_v_per_k, head_dim)
    else:
        raise NotImplementedError(f"v-head reorder on dim {dim} of a LoRA pair")
    return type(tensor)(A, B)


qwen._LinearAttentionVReorderBase._reorder_v_heads = staticmethod(_reorder_v_heads)


def selftest() -> None:
    class Pair:  # the shape of convert_lora_to_gguf.LoraTorchTensor
        def __init__(self, A, B):
            self._lora_A, self._lora_B = A, B

    torch.manual_seed(0)
    k, r, hd, rank = 16, 3, 128, 32               # Qwen3.5-27B linear attention
    out, inp = k * r * hd, 5120
    A, B = torch.randn(rank, inp), torch.randn(out, rank)
    want = _orig(B @ A, 0, k, r, hd)
    p = _reorder_v_heads(Pair(A, B), 0, k, r, hd)
    assert torch.allclose(p._lora_B @ p._lora_A, want, atol=1e-4), "row reorder"
    A, B = torch.randn(rank, out), torch.randn(inp, rank)   # out_proj: columns are V heads
    want = _orig(B @ A, 1, k, r, hd)
    p = _reorder_v_heads(Pair(A, B), 1, k, r, hd)
    assert torch.allclose(p._lora_B @ p._lora_A, want, atol=1e-4), "column reorder"
    print("selftest ok: factor reorders reproduce the reordered product exactly")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        runpy.run_path(f"{LLAMA}/convert_lora_to_gguf.py", run_name="__main__")

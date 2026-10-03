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


# Qwen3-Coder-Next (Qwen3NextModel): in_proj_qkvz is reordered from per-head
# [q,k,v,z] interleaving to grouped q|k|v|z and split into ATTN_QKV and ATTN_GATE.
# That is a permutation and split of W's rows, which live in B. The permutation
# is computed by running the converter's own transform on row indices, so it
# matches llama.cpp by construction.
_orig_next_modify = qwen.Qwen3NextModel.modify_tensors


def _qkvz_row_index(hp) -> tuple[torch.Tensor, torch.Tensor]:
    hk, hv = hp["linear_key_head_dim"], hp["linear_value_head_dim"]
    nv, nk = hp["linear_num_value_heads"], hp["linear_num_key_heads"]
    split = [hk, hk, nv // nk * hv, nv // nk * hv]
    rows = sum(split) * nk
    idx = torch.arange(rows).view(1, rows)               # the transform's view of W.T, hidden_size = 1
    idx = idx.view(-1, nk, sum(split))
    q, k, v, z = torch.split(idx, split, dim=-1)
    qkv = torch.cat([q.reshape(1, -1), k.reshape(1, -1), v.reshape(1, -1)], dim=-1).reshape(-1)
    return qkv, z.reshape(-1)


def _next_modify(self, data_torch, name, bid):
    if hasattr(data_torch, "_lora_A") and "in_proj_qkvz.weight" in name:
        A, B = data_torch._lora_A, data_torch._lora_B
        qkv_idx, z_idx = _qkvz_row_index(self.hparams)
        yield (self.format_tensor_name(qwen.gguf.MODEL_TENSOR.ATTN_QKV, bid, ".weight"),
               type(data_torch)(A, B[qkv_idx]))
        yield (self.format_tensor_name(qwen.gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"),
               type(data_torch)(A, B[z_idx]))
        return
    yield from _orig_next_modify(self, data_torch, name, bid)


qwen.Qwen3NextModel.modify_tensors = _next_modify


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
    # Qwen3Next qkvz: the index permutation must reproduce the converter's dense transform.
    hp = {"linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_num_value_heads": 32,
          "linear_num_key_heads": 16, "hidden_size": 64}
    split = [128, 128, 256, 256]
    W = torch.randn(sum(split) * 16, 64)
    d = W.permute(1, 0).contiguous().view(-1, 16, sum(split))
    q, k, v, z = torch.split(d, split, dim=-1)
    qkv_dense = torch.cat([t.contiguous().view(64, -1) for t in (q, k, v)], -1).permute(1, 0)
    z_dense = z.contiguous().view(64, -1).permute(1, 0)
    qi, zi = _qkvz_row_index(hp)
    assert torch.equal(W[qi], qkv_dense) and torch.equal(W[zi], z_dense), "qkvz row permutation"
    print("selftest ok: factor reorders reproduce the reordered product exactly")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        runpy.run_path(f"{LLAMA}/convert_lora_to_gguf.py", run_name="__main__")

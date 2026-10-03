#!/usr/bin/env python3
"""Load Qwen3-Coder-Next for QLoRA on a 121 GB machine (amendment 12).

transformers stores each layer's 512 experts as two fused 3D parameters
(gate_up_proj 512x1024x2048, down_proj 512x2048x512): 77B of the model's 79.7B.
bitsandbytes' transformers integration only quantises nn.Linear, so standard
QLoRA leaves the experts in bf16 (~154 GB) and the model does not fit.

Here every expert matrix is quantised to NF4 on its own as it is streamed from
the checkpoint (which stores experts unfused), and the experts module is
replaced by QuantExperts, whose forward is the reference Qwen3NextExperts loop
with each routed expert's weights dequantised on the fly. The experts stay
frozen. LoRA goes on attention, the DeltaNet projections and the shared expert,
all ordinary Linear layers kept in bf16 (~3B parameters).

Memory: ~40 GB of NF4 experts + ~6 GB bf16 rest, against ~160 GB in bf16.

    python scripts/train/quant_moe.py --check 3     # layer 3: NF4 vs bf16 reference
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import bitsandbytes.functional as BF
import torch
from safetensors import safe_open
from torch import nn

SNAP = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/*/"))


class QuantExperts(nn.Module):
    """Drop-in for Qwen3NextExperts with NF4 weights, frozen.

    Each layer's experts are stacked and quantised as two NF4 blocks. In forward
    they are dequantised once (~3 GB bf16, transient) and every routed token is
    processed in one grouped matmul (torch._grouped_mm), with tokens sorted by
    expert. Benchmarked on layer 3, 4096 tokens, fwd+bwd: 173 ms, against 879 ms
    for a per-expert dequantise-and-loop and 2410 ms for a fused-weight loop.
    """

    def __init__(self, config):
        super().__init__()
        self.E, self.I, self.H = config.num_experts, config.moe_intermediate_size, config.hidden_size
        self.K = config.num_experts_per_tok
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_act]
        self.q_gate_up = None     # NF4 of (E * 2I, H): rows [gate; up] per expert, as gate_up_proj
        self.q_down = None        # NF4 of (E * H, I)

    def set_layer(self, gate_up: torch.Tensor, down: torch.Tensor):
        """gate_up (E, 2I, H), down (E, H, I), bf16 on GPU."""
        self.q_gate_up = BF.quantize_4bit(gate_up.reshape(-1, self.H).contiguous(), quant_type="nf4",
                                          compress_statistics=True)
        self.q_down = BF.quantize_4bit(down.reshape(-1, self.I).contiguous(), quant_type="nf4",
                                       compress_statistics=True)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        G = BF.dequantize_4bit(*self.q_gate_up).view(self.E, 2 * self.I, self.H).to(hidden_states.dtype)
        D = BF.dequantize_4bit(*self.q_down).view(self.E, self.H, self.I).to(hidden_states.dtype)
        flat = top_k_index.reshape(-1)
        order = torch.argsort(flat)
        tok = order // top_k_index.shape[1]
        offs = torch.cumsum(torch.bincount(flat, minlength=self.E), 0).to(torch.int32)
        h = torch._grouped_mm(hidden_states[tok], G.transpose(1, 2), offs=offs)
        gate, up = h.chunk(2, dim=-1)
        y = torch._grouped_mm(self.act_fn(gate) * up, D.transpose(1, 2), offs=offs)
        y = y * top_k_weights.reshape(-1)[order, None].to(y.dtype)
        return torch.zeros_like(hidden_states).index_add_(0, tok, y.to(hidden_states.dtype))


def expert_keys(index: dict) -> dict:
    """{(layer, expert): {gate|up|down: key}} from the checkpoint index."""
    out: dict = defaultdict(dict)
    for k in index:
        parts = k.split(".")
        if ".mlp.experts." in k and parts[-1] == "weight":
            layer, e, proj = int(parts[2]), int(parts[5]), parts[6].replace("_proj", "")
            out[(layer, e)][proj] = k
    return out


def load(snapshot: Path = SNAP, layers: list[int] | None = None, num_layers: int | None = None):
    """Build the model with NF4 experts. `num_layers` truncates the model (plumbing
    tests only); `layers` limits which checkpoint layers are loaded."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(snapshot)
    if num_layers:
        cfg.num_hidden_layers = num_layers
        if getattr(cfg, "layer_types", None):
            cfg.layer_types = cfg.layer_types[:num_layers]
        layers = list(range(num_layers))
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
    for layer in model.model.layers:
        layer.mlp.experts = QuantExperts(cfg)
    model.to_empty(device="cuda")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    ek = expert_keys(index)
    expert_files = set()
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    by_file = defaultdict(list)
    for k, f in index.items():
        by_file[f].append(k)
    loaded = 0
    for f, keys in sorted(by_file.items()):
        st = safe_open(str(snapshot / f), "pt", device="cpu")
        for k in keys:
            if ".mlp.experts." in k:
                expert_files.add(f)
                continue
            if layers is not None and ".layers." in k and int(k.split(".")[2]) not in layers:
                continue
            target = params.get(k) if k in params else buffers.get(k)
            if target is None:
                continue
            with torch.no_grad():
                target.copy_(st.get_tensor(k).to(target.dtype))
            loaded += 1
    handles: dict = {}
    def get(k):
        f = index[k]
        if f not in handles:
            handles[f] = safe_open(str(snapshot / f), "pt", device="cpu")
        return handles[f].get_tensor(k)
    E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
    for li in range(cfg.num_hidden_layers):
        if layers is not None and li not in layers:
            continue
        gu = torch.empty(E, 2 * I, H, dtype=torch.bfloat16, device="cuda")
        dn = torch.empty(E, H, I, dtype=torch.bfloat16, device="cuda")
        for e in range(E):
            p = ek[(li, e)]
            gu[e] = torch.cat([get(p["gate"]), get(p["up"])], 0).to("cuda", torch.bfloat16)
            dn[e] = get(p["down"]).to("cuda", torch.bfloat16)
        model.model.layers[li].mlp.experts.set_layer(gu, dn)
        del gu, dn
        torch.cuda.empty_cache()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg, loaded


def check(layer: int) -> None:
    """NF4 experts vs the bf16 reference module on one real layer."""
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextExperts
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(SNAP)
    index = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    ek = expert_keys(index)
    ref = Qwen3NextExperts(cfg).to("cuda", torch.bfloat16)
    q = QuantExperts(cfg)
    gate_w = safe_open(str(SNAP / index[f"model.layers.{layer}.mlp.gate.weight"]), "pt").get_tensor(
        f"model.layers.{layer}.mlp.gate.weight").to("cuda", torch.bfloat16)
    with torch.no_grad():
        for e in range(cfg.num_experts):
            t = {p: safe_open(str(SNAP / index[k]), "pt").get_tensor(k) for p, k in ek[(layer, e)].items()}
            ref.gate_up_proj[e] = torch.cat([t["gate"], t["up"]], 0).to("cuda", torch.bfloat16)
            ref.down_proj[e] = t["down"].to("cuda", torch.bfloat16)
        q.set_layer(ref.gate_up_proj.data, ref.down_proj.data)
        torch.manual_seed(0)
        x = torch.randn(256, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
        logits = x @ gate_w.T
        w, idx = torch.topk(torch.softmax(logits.float(), -1), cfg.num_experts_per_tok, dim=-1)
        w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
        a, b = ref(x, idx, w), q(x, idx, w)
        rel = ((a - b).float().norm() / a.float().norm()).item()
        cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()
    print(f"layer {layer}: NF4 vs bf16 experts: relative error {rel:.4f}, mean cosine {cos:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int)
    a = ap.parse_args()
    if a.check is not None:
        check(a.check)

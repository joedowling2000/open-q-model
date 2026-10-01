#!/usr/bin/env python3
"""Merge LoRA adapters into a copy of the original Qwen3.5 checkpoint files.

Why not PEFT's merge_and_unload + save_pretrained: re-saving through
transformers rewrites the layout (the original has 1199 tensors, transformers
exposes fused ones), and llama.cpp then rejects the file ("missing tensor
blk.24.attn_norm"; see eval_checkpoint.sh). So the arithmetic is done on the
original shards themselves, W <- W + (alpha / r) * B @ A, and every other byte
is left as it was.

Why merge at all: llama.cpp applies run-time LoRA as extra matmuls on every
layer. With two stacked adapters (gate A + SFT) decoding fell to ~1 tok/s per
slot. RL is mostly generation, so it starts from the round 3 model merged into
one Q8_0 GGUF, and only its own small adapter is applied at run time
(amendment 9).

Adapters are applied in order, each to the result of the previous one, which is
what training did: SFT trained on base + gate A merged.

    python scripts/train/merge_into_checkpoint.py --out /path/merged \\
        out/lora-cpt/final out/lora-sft-r3/final
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SNAP = next(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/*/"))


def load_deltas(adapter: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(str(adapter / "adapter_model.safetensors"), "pt")
    out = {}
    for k in f.keys():
        if not k.endswith("lora_A.weight"):
            continue
        module = k[: -len(".lora_A.weight")]
        target = module.replace("base_model.model.model.", "model.language_model.") + ".weight"
        out[target] = (f.get_tensor(k), f.get_tensor(module + ".lora_B.weight"), scale)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None, help="checkpoint dir to merge into (default: Qwen3.5-27B)")
    ap.add_argument("adapters", nargs="+")
    args = ap.parse_args()

    global SNAP
    if args.base:
        SNAP = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    deltas = [load_deltas(Path(a)) for a in args.adapters]
    wanted = {t for d in deltas for t in d}
    index = json.loads((SNAP / "model.safetensors.index.json").read_text())["weight_map"]
    missing = wanted - set(index)
    if missing:
        raise SystemExit(f"{len(missing)} adapter targets not in the checkpoint, e.g. {sorted(missing)[:3]}")

    patched = 0
    for shard in sorted(set(index.values())):
        f = safe_open(str(SNAP / shard), "pt")
        tensors = {}
        for name in f.keys():
            w = f.get_tensor(name)
            if name in wanted:
                acc = w.to(torch.float32)
                for d in deltas:                     # in order: gate A, then SFT
                    if name in d:
                        A, B, scale = d[name]
                        acc += scale * (B.to(torch.float32) @ A.to(torch.float32))
                w = acc.to(w.dtype)
                patched += 1
            tensors[name] = w
        save_file(tensors, str(out / shard), metadata=f.metadata())
        print(f"{shard}: done", flush=True)
    for p in SNAP.iterdir():                          # config, tokenizer, index, ...
        if not p.name.endswith(".safetensors"):
            shutil.copy(p, out / p.name)
    expected = len(wanted)
    print(f"patched {patched} tensors (expected {expected})")
    return 0 if patched == expected else 1


if __name__ == "__main__":
    raise SystemExit(main())

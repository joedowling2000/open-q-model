#!/usr/bin/env python3
"""LoRA continued pretraining on the q corpus.

A plain training loop rather than Trainer: on this box the things that matter
are gradient checkpointing, accumulation and when we can stop, and those are
clearer written out than configured.

Memory, measured on a 121 GB GB10: a 27B in bf16 is ~54 GB of weights; LoRA
adds a few hundred MB of trainable parameters and optimiser state, so the run
fits with room for activations if checkpointing is on and micro-batch is 1.
Throughput is bandwidth-bound — a step reads the weights several times, so
expect seconds per step, not milliseconds.

    python scripts/train/train_lora.py --smoke            # 0.8B, a few steps
    python scripts/train/train_lora.py --model Qwen/Qwen3.5-27B
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/joedowling/Projects/qeval")


def batches(blocks: np.ndarray, size: int, shuffle: bool, seed: int = 0):
    idx = np.arange(len(blocks))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for i in range(0, len(idx) - size + 1, size):
        yield torch.from_numpy(blocks[idx[i:i + size]].astype(np.int64))


@torch.no_grad()
def evaluate(model, blocks: np.ndarray, device, max_batches: int = 20) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(batches(blocks, 1, shuffle=False)):
        if i >= max_batches:
            break
        ids = batch.to(device)
        loss = model(input_ids=ids, labels=ids).loss
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B")
    ap.add_argument("--data", default="data/cpt")
    ap.add_argument("--out", default="out/lora-cpt")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny model, 6 steps: proves the loop, not the result")
    args = ap.parse_args()

    if args.smoke:
        args.model, args.epochs, args.eval_every = "Qwen/Qwen3.5-0.8B", 0.01, 3
        args.out = "out/lora-smoke"

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    data = ROOT / args.data
    train = np.load(data / "train.npy")
    val = np.load(data / "val.npy")
    print(f"data: {train.shape[0]} train blocks, {val.shape[0]} val, "
          f"seq {train.shape[1]}", flush=True)

    print(f"loading {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        # Attention projections only. Adapting the MLP as well roughly doubles
        # the trainable parameters for a marginal gain on a corpus this size.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable {trainable/1e6:.1f}M of {total/1e9:.1f}B "
          f"({100*trainable/total:.3f}%)", flush=True)

    device = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    steps_per_epoch = max(1, len(train) // (args.micro_batch * args.accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(args.warmup, 1)) *
                       0.5 * (1 + math.cos(math.pi * min(s / total_steps, 1.0))))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    print(f"baseline val loss: {evaluate(model, val, device):.4f}", flush=True)

    step, micro, running, t0 = 0, 0, 0.0, time.time()
    history = []
    for epoch in range(math.ceil(args.epochs)):
        for batch in batches(train, args.micro_batch, shuffle=True, seed=epoch):
            ids = batch.to(device)
            loss = model(input_ids=ids, labels=ids).loss / args.accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.accum:
                continue

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            rate = (time.time() - t0) / step
            print(f"step {step}/{total_steps} loss {running:.4f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {rate:.1f}s/step", flush=True)
            history.append({"step": step, "loss": running})
            running = 0.0

            if step % args.eval_every == 0 or step == total_steps:
                vl = evaluate(model, val, device)
                print(f"  val loss {vl:.4f}", flush=True)
                history.append({"step": step, "val_loss": vl})
            if step % args.save_every == 0:
                model.save_pretrained(out / f"step-{step}")
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    final_val = evaluate(model, val, device)
    model.save_pretrained(out / "final")
    (out / "history.json").write_text(json.dumps(
        {"args": vars(args), "history": history, "final_val_loss": final_val}, indent=1))
    print(f"final val loss {final_val:.4f}; adapter saved to {out/'final'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

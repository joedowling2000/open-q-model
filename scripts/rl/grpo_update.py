#!/usr/bin/env python3
"""One GRPO policy update from a round's rollouts (amendment 4).

The policy is the merged round 4 checkpoint (bf16) plus an RL LoRA, which is
the only thing trained. Per group (one prompt, G attempts) the advantage is
reward minus the group mean ("Dr. GRPO": no division by the group's std, which
over-weights near-unanimous groups). Groups with no spread carry no signal and
are skipped. The loss is -advantage x the summed log-probability of the
attempt's tokens, divided by a constant 512 (the generation cap), so long and
short answers are not weighted differently through length.

Rollouts come from the Q8 GGUF of the same policy; the update is in bf16. The
mismatch is small, and each batch is used for one pass only (no PPO epochs),
so there is no ratio to clip.

    python scripts/rl/grpo_update.py --round 1 --adapter-in none --adapter-out out/rl/r001
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

ROOT = Path("/home/joedowling/Projects/qeval")
BASE = ROOT / "out/merged/qwen3.5-27b-r4"
TOKENIZER = ("/home/joedowling/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
             "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654")
NORM = 512.0
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--adapter-in", required=True, help="previous RL adapter dir, or 'none'")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=32, help="samples per optimiser step")
    ap.add_argument("--run", default="rl", help="corpus/<run>/ holds the rollouts")
    ap.add_argument("--kl", type=float, default=0.0,
                    help="KL penalty to the reference policy (round 4 = the adapter disabled); "
                         "added after the round-14 collapse (incident 7)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--fresh-optimizer", action="store_true",
                    help="do not load the previous adapter's Adam state")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / f"corpus/{args.run}/rollouts_r{args.round:03d}.jsonl").open()]
    samples = []
    for r in rows:
        rw = r.get("rewards")
        if not rw or len(set(rw)) < 2:
            continue
        mean = sum(rw) / len(rw)
        for c, x in zip(r["completions"], rw):
            samples.append((r["prompt"], c, x - mean))
    random.Random(args.round).shuffle(samples)
    print(f"{len(samples)} samples from {sum(1 for r in rows if r.get('rewards') and len(set(r['rewards'])) > 1)} "
          f"groups with spread", flush=True)
    if not samples:
        print("no signal this round; adapter unchanged", flush=True)
        return 0

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    if args.adapter_in == "none":
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    else:
        model = PeftModel.from_pretrained(model, args.adapter_in, is_trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.99))
    opt_path = Path(args.adapter_in) / "optimizer.pt" if args.adapter_in != "none" else None
    if opt_path and opt_path.exists() and not args.fresh_optimizer:
        opt.load_state_dict(torch.load(opt_path))

    model.train()
    t0, step, done, loss_acc = time.time(), 0, 0, 0.0
    for prompt, completion, adv in samples:
        p_ids = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                        add_generation_prompt=True, return_dict=False, enable_thinking=False)
        c_ids = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + c_ids], device="cuda")
        logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
        logp = torch.log_softmax(logits, -1).gather(1, ids[0, len(p_ids):, None]).squeeze(1)
        loss = -(adv * logp.sum() / NORM)
        if args.kl > 0:
            # k3 estimator, per token: exp(ref - logp) - (ref - logp) - 1 >= 0
            with torch.no_grad(), model.disable_adapter():
                ref_logits = model(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
                ref = torch.log_softmax(ref_logits, -1).gather(1, ids[0, len(p_ids):, None]).squeeze(1)
            d = ref - logp
            loss = loss + args.kl * (torch.exp(d) - d - 1).sum() / NORM
        loss = loss / args.accum
        loss.backward()
        loss_acc += loss.item()
        done += 1
        if done % args.accum == 0 or done == len(samples):
            torch.nn.utils.clip_grad_norm_(params, args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            print(f"step {step} ({done}/{len(samples)}) loss {loss_acc:.4f} "
                  f"{(time.time() - t0) / done:.1f}s/sample", flush=True)
            loss_acc = 0.0
    out = Path(args.adapter_out)
    model.save_pretrained(out)
    torch.save(opt.state_dict(), out / "optimizer.pt")
    print(f"GRPO_DONE round {args.round}: {step} steps, adapter -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

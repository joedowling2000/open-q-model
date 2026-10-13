#!/usr/bin/env python3
"""Q-HumanEval samples from Claude models, for a comparison under our protocol.

Prompts are the harness's own (q_humaneval template), sent as one user message,
like the harness's litellm backend. Protocol as close to ours as the API allows:
one-shot, no tools, thinking disabled, max_tokens 512. Temperature cannot be set
on these models (the API rejects non-default sampling parameters), so they run at
the API default of 1.0; this deviation is reported with every result.

Output: the harness's solution jsonl (task_id, sample_index, completion,
model_name), scored with `python -m src.cli execute`. Unlike the harness, a
failed request is never written as an empty completion (incident 3): the script
stops instead, so nothing silently scores as a failure.

Never used for training (lineage rule): the output stays in results/frontier/.

    python scripts/frontier/claude_gen.py --model claude-sonnet-5 --n 2 --limit 20 --mode direct
    python scripts/frontier/claude_gen.py --model claude-opus-4-8 --n 30 --mode batch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anthropic

ROOT = Path("/home/joedowling/Projects/qeval")
HARNESS = ROOT / "q-evaluation-harness"
OUT = ROOT / "results/frontier"
PRICE = {"claude-opus-4-8": (5, 25), "claude-opus-5": (5, 25), "claude-sonnet-5": (2, 10),
         "claude-haiku-4-5": (1, 5)}                                   # $ per MTok in/out


def problems():
    sys.path.insert(0, str(HARNESS))
    from src.cli import get_loader, get_prompt_template
    from src.datasets.registry import get_dataset_config
    c = get_dataset_config("q-humaneval")
    t = get_prompt_template(c["prompt_template"])
    return [(p["task_id"], t.format(p)) for p in get_loader(c["format"]).load(str(HARNESS / c["path"]))]


def params(model: str, prompt: str, max_tokens: int) -> dict:
    p = {"model": model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    if model != "claude-haiku-4-5":
        p["thinking"] = {"type": "disabled"}       # Opus 5 / Sonnet 5 think by default; Opus 4.8 accepts it
    else:
        p["temperature"] = 0.8                     # Haiku 4.5 still accepts sampling parameters
    return p


def text_of(msg) -> str:
    if msg.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in msg.content if b.type == "text")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(PRICE))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--limit", type=int, default=None, help="first N problems (pilot)")
    ap.add_argument("--mode", choices=["direct", "batch"], default="batch")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    probs = problems()[: a.limit]
    reqs = {f"t{tid}-s{i}": (tid, i, params(a.model, prompt, a.max_tokens))
            for tid, prompt in probs for i in range(a.n)}
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")     # needed only for keys not scoped to a workspace
    client = anthropic.Anthropic(max_retries=8, default_headers={"anthropic-workspace-id": ws} if ws else None)
    results, usage, stops = {}, [0, 0], {}

    def keep(cid, msg):
        results[cid] = text_of(msg)
        usage[0] += msg.usage.input_tokens; usage[1] += msg.usage.output_tokens
        stops[msg.stop_reason] = stops.get(msg.stop_reason, 0) + 1

    if a.mode == "direct":
        def one(cid):
            return cid, client.messages.create(**reqs[cid][2])
        with ThreadPoolExecutor(8) as ex:
            for cid, msg in ex.map(one, reqs):
                keep(cid, msg)
    else:
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        batch = client.messages.batches.create(requests=[
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p)) for cid, (_, _, p) in reqs.items()])
        print(f"batch {batch.id} submitted: {len(reqs)} requests", flush=True)
        (OUT / "batches.log").parent.mkdir(parents=True, exist_ok=True)
        with (OUT / "batches.log").open("a") as fh:
            fh.write(f"{datetime.now().isoformat()} {a.model} n={a.n} {batch.id}\n")
        while (b := client.messages.batches.retrieve(batch.id)).processing_status != "ended":
            print(f"  {b.processing_status}: {b.request_counts.processing} processing, "
                  f"{b.request_counts.succeeded} done", flush=True)
            time.sleep(120)
        bad = []
        for r in client.messages.batches.results(batch.id):
            if r.result.type == "succeeded":
                keep(r.custom_id, r.result.message)
            else:
                bad.append((r.custom_id, r.result.type))
        if bad:
            print(f"STOPPED: {len(bad)} requests did not succeed, e.g. {bad[:5]}; "
                  f"nothing written (batch {batch.id})", flush=True)
            return 1

    missing = [c for c in reqs if c not in results]
    empty = [c for c in reqs if c in results and not results[c].strip()]
    if missing:
        print(f"STOPPED: {len(missing)} requests missing; nothing written", flush=True)
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"{a.model}{a.tag}" + (f"-pilot{a.limit}x{a.n}" if a.limit else "")
    with (OUT / f"{name}.jsonl").open("w") as fh:
        for cid, (tid, i, _) in reqs.items():
            fh.write(json.dumps({"task_id": tid, "sample_index": i, "completion": results[cid],
                                 "model_name": a.model, "generated_at": datetime.now().isoformat()}) + "\n")
    pin, pout = PRICE[a.model]
    cost = (usage[0] * pin + usage[1] * pout) / 1e6 * (0.5 if a.mode == "batch" else 1.0)
    summary = {"model": a.model, "requests": len(reqs), "input_tokens": usage[0], "output_tokens": usage[1],
               "mean_output": round(usage[1] / len(reqs), 1), "stop_reasons": stops,
               "empty_or_refused": len(empty), "cost_usd": round(cost, 2), "mode": a.mode,
               "out": str(OUT / f"{name}.jsonl")}
    print("CLAUDE_GEN", json.dumps(summary), flush=True)
    with (OUT / "usage.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

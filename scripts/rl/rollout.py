#!/usr/bin/env python3
"""One RL round's rollouts: sample prompts, generate G attempts each, score them.

Prompt choice (amendment 10 v): each problem's weight is its source weight
(MBPP and Morgan Stanley 1.5) times a learnability factor from its pass rate in
earlier rounds. 2.0 if it has been solved sometimes but not always, 1.0 if
unseen, 0.25 if always or never solved. Problems with no spread teach GRPO
nothing.

Reward (amendment 10 iv): 1.0 if the function passes all fresh inputs from the
problem's generator; otherwise 0.2 x the fraction passed; 0 if it does not load
or defines the wrong name. Generation uses the evaluation's settings: one-shot,
temperature 0.8, 512 tokens.

    python scripts/rl/rollout.py --url http://127.0.0.1:8107/v1 --model policy \\
        --round 1 --prompts 128 --group 8
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
import qcheck  # noqa: E402

HISTORY = ROOT / "corpus/rl_history.json"     # id -> [n_attempts, n_full_passes]


def learnability(h) -> float:
    if not h:
        return 1.0
    n, k = h
    return 2.0 if 0 < k < n else 0.25


def generate(url: str, model: str, prompt: str, n: int) -> list[str]:
    body = json.dumps({"model": model, "n": n, "temperature": 0.8, "max_tokens": 512,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return [c["message"]["content"] for c in json.load(r)["choices"]]


def extract(text: str) -> str | None:
    m = re.findall(r"```(?:q|k)?\s*\n(.*?)```", text, re.S)
    return (m[-1] if m else text).strip() or None


def reward(code: str | None, rec: dict, seeds) -> float:
    if not code or not re.search(rf"(^|\n)\s*{re.escape(rec['fname'])}\s*:", code):
        return 0.0
    try:
        res = qcheck.check(code + f"\nsolve:{rec['fname']};", rec["python_src"], rec["argnames"], seeds)
    except Exception:
        return 0.0
    if res["ok"]:
        return 1.0
    if res.get("reason"):          # did not load
        return 0.0
    return 0.2 * (1 - res["n_fail"] / res["n_cases"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--prompts", type=int, default=128)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    pool = [json.loads(l) for l in (ROOT / "corpus/rl_pool.jsonl").open()]
    hist = json.loads(HISTORY.read_text()) if HISTORY.exists() else {}
    rng = random.Random(1000 + args.round)
    w = [p["weight"] * learnability(hist.get(p["id"])) for p in pool]
    chosen, seen = [], set()
    while len(chosen) < args.prompts:
        p = rng.choices(pool, weights=w)[0]
        if p["id"] not in seen:
            seen.add(p["id"])
            chosen.append(p)
    seeds = range(2_000_000 + 1000 * args.round, 2_000_000 + 1000 * args.round + args.cases)

    def one(p):
        try:
            outs = generate(args.url, args.model, p["prompt"], args.group)
        except Exception as e:
            return {"id": p["id"], "error": type(e).__name__}
        rewards = [reward(extract(o), p, seeds) for o in outs]
        return {"id": p["id"], "prompt": p["prompt"], "completions": outs, "rewards": rewards}

    out = ROOT / f"corpus/rl/rollouts_r{args.round:03d}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(args.workers) as pool_, out.open("w") as fh:
        for r in pool_.map(one, chosen):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows.append(r)
    for r in rows:
        if "rewards" in r:
            n, k = hist.get(r["id"], [0, 0])
            hist[r["id"]] = [n + len(r["rewards"]), k + sum(x == 1.0 for x in r["rewards"])]
    HISTORY.write_text(json.dumps(hist))
    rs = [x for r in rows if "rewards" in r for x in r["rewards"]]
    spread = sum(1 for r in rows if "rewards" in r and len(set(r["rewards"])) > 1)
    summary = {"round": args.round, "prompts": len(rows), "rollouts": len(rs),
               "mean_reward": round(sum(rs) / max(1, len(rs)), 4),
               "full_pass_rate": round(sum(x == 1.0 for x in rs) / max(1, len(rs)), 4),
               "groups_with_spread": spread, "errors": sum("error" in r for r in rows)}
    print("ROLLOUT_SUMMARY", json.dumps(summary), flush=True)
    with (ROOT / "corpus/rl/summary.jsonl").open("a") as fh:
        fh.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

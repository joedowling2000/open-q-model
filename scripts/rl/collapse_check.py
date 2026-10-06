#!/usr/bin/env python3
"""Incident 7's automatic stop: run after each round's rollouts, before training.

Exit 1 (stop) if more than 1% of the round's completions are degenerate (junk
or runs of one repeated word, the round-15 signature), or if, on problems seen in
any earlier round of either RL run, mean reward fell by more than 0.03 against
their most recent earlier score. Rounds 6-13 rose +0.06 to +0.16 on this measure;
round 14 fell 0.04 and round 15 0.26.

    python scripts/rl/collapse_check.py --run rl2 --round 3
"""
import argparse, json, re, sys
from pathlib import Path
ROOT = Path("/home/joedowling/Projects/qeval")


def degenerate(c: str) -> bool:
    words = re.findall(r"\S+", c)
    if not words:
        return True
    non_ascii = sum(ord(ch) > 127 for ch in c) / max(1, len(c))
    top = max(words.count(w) for w in set(words)) / len(words)
    return non_ascii > 0.05 or (len(words) > 40 and top > 0.3)


def rollouts(path):
    return {json.loads(l)["id"]: json.loads(l) for l in path.open() if "rewards" in json.loads(l)}


ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--round", type=int, required=True)
a = ap.parse_args()
cur = rollouts(ROOT / f"corpus/{a.run}/rollouts_r{a.round:03d}.jsonl")
comps = [c for r in cur.values() for c in r["completions"]]
deg = sum(map(degenerate, comps)) / max(1, len(comps))
prev = {}
for f in sorted((ROOT / "corpus/rl").glob("rollouts_r*.jsonl"))[:13]:      # healthy rounds of run 1
    for i, r in rollouts(f).items():
        prev[i] = sum(r["rewards"]) / len(r["rewards"])
for q in range(1, a.round):
    p = ROOT / f"corpus/{a.run}/rollouts_r{q:03d}.jsonl"
    if p.exists():
        for i, r in rollouts(p).items():
            prev[i] = sum(r["rewards"]) / len(r["rewards"])
rep = [i for i in cur if i in prev]
delta = (sum(sum(cur[i]["rewards"]) / len(cur[i]["rewards"]) for i in rep) / len(rep)
         - sum(prev[i] for i in rep) / len(rep)) if rep else 0.0
verdict = "STOP" if deg > 0.01 or delta < -0.03 else "OK"
print(f"COLLAPSE_CHECK round {a.round}: degenerate {100*deg:.2f}%, same-problem reward change {delta:+.3f} "
      f"on {len(rep)} problems -> {verdict}", flush=True)
sys.exit(1 if verdict == "STOP" else 0)

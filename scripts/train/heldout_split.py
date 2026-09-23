#!/usr/bin/env python3
"""Fix the internal held-out set (amendment 7) before any distillation runs.

15% of the medium and hard original questions in the Q study bank are set aside
and never trained on, distilled from, or used as RL prompts. They are scored the
way the benchmark is — generate q, run it against the Python reference on fresh
generated inputs — so progress between Q-HumanEval runs is visible on problems
harder than the benchmark's.

Near-duplicates move together. Questions that share identical q, identical
Python, or a `similar_reference_ids` link form one group, and a group is held
out or kept whole; otherwise the answer to a held-out question reaches training
through its twin. Family exercises whose q matches a held-out question are
excluded from training too.

The split is deterministic (fixed seed, sorted inputs), stratified by
difficulty and topic, and its id list and hash are committed.

    python scripts/train/heldout_split.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
BANK = ROOT / "corpus/q_study_v5"
OUT = ROOT / "corpus/heldout_v5.json"
FRACTION = 0.15
SEED = 20260923


def load(name: str) -> list[dict]:
    return [json.loads(l) for l in (BANK / name).open()]


def groups(records: list[dict]) -> dict[str, str]:
    """Union-find over the three near-duplicate relations; returns id -> root."""
    parent = {r["id"]: r["id"] for r in records}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_key: dict[tuple, str] = {}
    for r in records:
        keys = [("py", r["python_src"].strip())]
        if r["solutions"]:
            keys.append(("q", r["solutions"][0]["code"].strip()))
        for k in keys:
            if k in by_key:
                union(r["id"], by_key[k])
            else:
                by_key[k] = r["id"]
        for other in r.get("similar_reference_ids", []):
            if other in parent:
                union(r["id"], other)
    return {r["id"]: find(r["id"]) for r in records}


def main() -> int:
    questions = load("questions.jsonl")
    reserve = load("reserve_exercises.jsonl")
    everything = questions + reserve
    root = groups(everything)

    members: dict[str, list[dict]] = defaultdict(list)
    for r in everything:
        members[root[r["id"]]].append(r)

    # Candidate groups: every group containing a medium/hard original. Strata by
    # (difficulty, topic) of the group's hardest original.
    rank = {"medium": 1, "hard": 2}
    strata: dict[tuple, list[str]] = defaultdict(list)
    for g, rs in members.items():
        mh = [r for r in rs if r["question_type"] == "original"
              and r.get("difficulty") in rank]
        if not mh:
            continue
        top = max(mh, key=lambda r: (rank[r["difficulty"]], r["id"]))
        strata[(top["difficulty"], top["topic"])].append(g)

    rng = random.Random(SEED)
    held_groups: set[str] = set()
    for key in sorted(strata):
        gs = sorted(strata[key])
        rng.shuffle(gs)
        n_items = sum(len(members[g]) for g in gs)
        target, taken = round(FRACTION * n_items), 0
        for g in gs:
            if taken >= target:
                break
            held_groups.add(g)
            taken += len(members[g])

    held = sorted(r["id"] for g in held_groups for r in members[g])
    held_originals = [r for g in held_groups for r in members[g]
                      if r["question_type"] == "original"]
    summary = {
        "purpose": "PREREGISTRATION amendment 7: internal held-out set; never trained on, "
                   "distilled from, or used as RL prompts",
        "bank": "corpus/q_study_v5",
        "bank_sha256": {n: hashlib.sha256((BANK / n).read_bytes()).hexdigest()
                        for n in ("questions.jsonl", "reserve_exercises.jsonl")},
        "seed": SEED,
        "fraction": FRACTION,
        "n_heldout": len(held),
        "by_type": {t: sum(1 for g in held_groups for r in members[g]
                           if r["question_type"] == t)
                    for t in ("original", "family exercise")},
        "by_difficulty": {d: sum(1 for r in held_originals if r.get("difficulty") == d)
                          for d in ("easy", "medium", "hard")},
        "with_verified_q": sum(1 for g in held_groups for r in members[g] if r["solutions"]),
        "ids": held,
    }
    summary["ids_sha256"] = hashlib.sha256("\n".join(held).encode()).hexdigest()
    OUT.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "ids"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

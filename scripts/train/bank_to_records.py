#!/usr/bin/env python3
"""Convert the Q study bank into the record format assemble_sft.py reads.

Exclusions (amendments 6 and 7): the held-out set, anything that failed
verify_bank.py, and the three known failures named in amendment 6.

Weighting: family exercises are templated one-liners from a composition
generator — 2,000 of them against ~560 usable originals. Left alone they would
dominate. Each original yields all four tasks; a stratified sample of 1,000
family exercises yields desc2q and py2q only.

The Python shown to the model is the reference `solve` plus the helpers it
calls. The input generator and generator-only helpers are dropped: they are
test machinery, not part of the problem.

    python scripts/train/bank_to_records.py
"""

from __future__ import annotations

import ast
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
BANK = ROOT / "corpus/q_study_v5"
OUT = BANK / "sft_records.jsonl"
KNOWN_BAD = {"question_0556", "question_0267", "question_0414"}
FAMILY_SAMPLE = 1000
SEED = 20260923


def reference_only(src: str) -> str:
    """Keep imports, `solve`, and the top-level functions `solve` reaches."""
    tree = ast.parse(src)
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    keep, todo = set(), ["solve"]
    while todo:
        name = todo.pop()
        if name in keep or name not in funcs:
            continue
        keep.add(name)
        todo += [n.id for n in ast.walk(funcs[name]) if isinstance(n, ast.Name)]
    body = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.FunctionDef) and n.name in keep)
            or (isinstance(n, ast.Assign) and not isinstance(n.value, ast.Lambda))]
    return ast.unparse(ast.Module(body=body, type_ignores=[]))


def main() -> int:
    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    ver = json.loads((ROOT / "corpus/q_study_v5_verification.json").read_text())["results"]
    recs = [json.loads(l) for n in ("questions.jsonl", "reserve_exercises.jsonl")
            for l in (BANK / n).open()]

    dropped = Counter()
    originals, family = [], defaultdict(list)
    for r in recs:
        if not r["solutions"]:
            continue
        if r["id"] in held:
            dropped["held out"] += 1
            continue
        if r["id"] in KNOWN_BAD or not ver.get(r["id"], {}).get("ok"):
            dropped["failed verification"] += 1
            continue
        (originals.append(r) if r["question_type"] == "original"
         else family[r["family"]].append(r))

    rng = random.Random(SEED)
    per_family = FAMILY_SAMPLE // len(family)
    fam_kept = []
    for name in sorted(family):
        pool = sorted(family[name], key=lambda r: r["id"])
        rng.shuffle(pool)
        fam_kept += pool[:per_family]
    dropped["family not sampled"] = sum(len(v) for v in family.values()) - len(fam_kept)

    with OUT.open("w") as fh:
        for r, tasks in [(r, ["desc2q", "py2q", "q2py", "q2desc"]) for r in originals] + \
                        [(r, ["desc2q", "py2q"]) for r in fam_kept]:
            sol = r["solutions"][0]
            fh.write(json.dumps({
                "id": r["id"],
                "description": r["description"],
                "python_solution": reference_only(r["python_src"]),
                "q_solution": sol["code"],
                "tasks": tasks,
                "topic": r["topic"],
                "difficulty": r.get("difficulty"),
                "source": "q_study_v5:" + r["question_type"],
                "authors": sol.get("authors"),
            }) + "\n")
    print(json.dumps({"originals": len(originals),
                      "by_difficulty": dict(Counter(r["difficulty"] for r in originals)),
                      "family_kept": len(fam_kept), "dropped": dict(dropped)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

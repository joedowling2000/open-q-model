#!/usr/bin/env python3
"""Round 2 SFT records: the Q study bank plus teacher-distilled problems,
with amendment 3's verified vector rewrites applied.

Where a loop solution has a verified vector rewrite, the vector version becomes
the target of every task (desc2q, py2q, ...), and the loop version is kept as
`q_solution_imperative`, which gives assemble_sft.py a `vectorise` sample. Where
no rewrite verified, the loop version stands; the loop share is reported, since
control 6 makes it a stated quality measure.

    python scripts/train/round2_records.py        # -> corpus/round2_records.jsonl
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
LOOPY = re.compile(r"\b(while|do)\[")


def main() -> int:
    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    rw_path = ROOT / "corpus/vector_rewrites.jsonl"
    rewrites = {}
    if rw_path.exists():
        for l in rw_path.open():
            r = json.loads(l)
            if r["ok"]:
                rewrites[r["id"]] = r["q"]

    dq_path = ROOT / "corpus/disqualified.json"
    disqualified = set(json.loads(dq_path.read_text())["disqualified"]) if dq_path.exists() else set()
    records, stats = [], Counter()

    def add(rec: dict, source: str) -> None:
        if rec["id"] in held:
            stats["refused: held out"] += 1
            return
        if rec["id"] in disqualified:          # release_check.py: flaky generator
            stats["refused: failed release check"] += 1
            return
        q = rec["q_solution"]
        if rec["id"] in rewrites and LOOPY.search(q):
            rec = {**rec, "q_solution": rewrites[rec["id"]], "q_solution_imperative": q}
            stats["vector rewrite applied"] += 1
        stats["loop target" if LOOPY.search(rec["q_solution"]) else "vector target"] += 1
        stats[source] += 1
        records.append(rec)

    for l in (ROOT / "corpus/q_study_v5/sft_records.jsonl").open():
        add(json.loads(l), "q_study_v5")
    for l in (ROOT / "corpus/distilled.jsonl").open():
        d = json.loads(l)
        if not d["ok"]:
            continue
        add({"id": d["id"], "description": d["description"],
             "python_solution": d["python_solution"], "q_solution": d["q_solution"],
             "tasks": ["desc2q", "py2q", "q2py", "q2desc"],
             "topic": d.get("topic"), "difficulty": d.get("difficulty"),
             "source": "distilled:" + str(d.get("source")), "teacher": d.get("teacher")},
            "distilled")

    with (ROOT / "corpus/round2_records.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    stats["records"] = len(records)
    stats["loop share of targets"] = round(stats["loop target"] / len(records), 3)
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

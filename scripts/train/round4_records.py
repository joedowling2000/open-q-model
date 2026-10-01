#!/usr/bin/env python3
"""Round 4 records (amendment 10): round 3's problems plus solved MBPP problems,
each also rendered in the benchmark's prompt format when the rename verifies.

    python scripts/train/round4_records.py      # -> corpus/round4_records.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/train"))
sys.path.insert(0, str(ROOT / "scripts/synth"))
import qcheck  # noqa: E402
from harness_format import render  # noqa: E402


def sources() -> dict:
    recs = {}
    for f in ("questions.jsonl", "reserve_exercises.jsonl"):
        for l in (ROOT / "corpus/q_study_v5" / f).open():
            r = json.loads(l)
            recs[r["id"]] = r
    for f in ("distilled.jsonl", "generated_problems.jsonl", "mbpp_rebuilt.jsonl"):
        p = ROOT / "corpus" / f
        if p.exists():
            for l in p.open():
                r = json.loads(l)
                if r.get("id") and r.get("python_src"):
                    recs[r["id"]] = r
    return recs


def main() -> int:
    recs = sources()
    rows = [json.loads(l) for l in (ROOT / "corpus/round4_base_records.jsonl").open()]

    def one(rec):
        src = recs[rec["id"]]
        if src.get("function_name"):
            rec = {**rec, "function_name": src["function_name"]}
        # Round 4 continues from merged round 3, which already learned the other
        # tasks: it trains on the prose prompt and the harness-format prompt only.
        rec = {**rec, "tasks": ["desc2q"]}
        try:
            out = render(rec, src["python_src"], qcheck.arg_order(src))
        except Exception:
            out = None
        if out:
            rec = {**rec, "harness_prompt": out[0], "harness_response": out[1]}
        return rec

    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(one, rows))
    stats = Counter("rendered" if r.get("harness_prompt") else "original format only" for r in rows)
    with (ROOT / "corpus/round4_records.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({"records": len(rows), **stats}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

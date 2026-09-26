#!/usr/bin/env python3
"""The standing release check from incident 2, run before any SFT assembly.

A problem's verification is only as good as its generator, and a generator that
raises on some seeds was verified on luck: it missed the acceptance seeds and
would miss an audit's. Each record's generator and reference are run on 200
seeds that no acceptance, audit or rewrite has used. Any crash disqualifies the
record from training, and the list is written for round2_records.py to exclude.

First run, 2026-09-26: 2 of 2129 round 2 records failed, after one of them
crashed the vector rewrite pass.

    python scripts/train/release_check.py corpus/round2_records.jsonl
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
import qcheck  # noqa: E402

SEEDS = range(1200000, 1200200)
OUT = ROOT / "corpus/disqualified.json"


def sources() -> dict:
    recs = {}
    for f in ("questions.jsonl", "reserve_exercises.jsonl"):
        for l in (ROOT / "corpus/q_study_v5" / f).open():
            r = json.loads(l)
            recs[r["id"]] = r
    for f in ("distilled.jsonl", "generated_problems.jsonl"):
        p = ROOT / "corpus" / f
        if p.exists():
            for l in p.open():
                r = json.loads(l)
                if r.get("id") and r.get("python_src"):
                    recs[r["id"]] = r
    return recs


def crashes(rec: dict) -> str | None:
    order = qcheck.arg_order(rec)
    g: dict = {}
    try:
        exec(rec["python_src"], g)
    except Exception as e:
        return f"does not load: {type(e).__name__}"
    for s in SEEDS:
        try:
            x = g["gen_input"](random.Random(s))
            if not isinstance(x, dict):
                x = {order[0]: x} if len(order) == 1 else dict(zip(order, x))
            g["solve"](**{k: x[k] for k in order})
        except Exception as e:
            return f"seed {s}: {type(e).__name__}: {str(e)[:60]}"
    return None


def main() -> int:
    ids = [json.loads(l)["id"] for l in (ROOT / sys.argv[1]).open()]
    recs = sources()
    bad = {i: why for i in ids if (why := crashes(recs[i]))}
    OUT.write_text(json.dumps({"seeds": [SEEDS.start, SEEDS.stop - 1],
                               "checked": len(ids), "disqualified": bad}, indent=1))
    print(f"release check: {len(ids)} records, {len(bad)} disqualified")
    for i, why in bad.items():
        print(" ", i, why)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

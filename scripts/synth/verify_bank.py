#!/usr/bin/env python3
"""Re-verify every q solution in the Q study bank on fresh inputs (amendment 6).

The bank's "verified" status came from its authors' saved test results. Nothing
is trained on until it has passed our own check here: 60 inputs from each
problem's generator, at seeds neither the authors nor any earlier audit used.

    python scripts/synth/verify_bank.py            # writes corpus/q_study_v5_verification.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import qcheck  # noqa: E402

ROOT = Path("/home/joedowling/Projects/qeval")
BANK = ROOT / "corpus/q_study_v5"
OUT = ROOT / "corpus/q_study_v5_verification.json"
SEEDS = range(500000, 500060)  # disjoint from 0-99, 100000-100099 and all earlier audits


def main() -> int:
    recs = [json.loads(l) for n in ("questions.jsonl", "reserve_exercises.jsonl")
            for l in (BANK / n).open()]
    recs = [r for r in recs if r["solutions"]]

    def one(r):
        try:
            res = qcheck.check(r["solutions"][0]["code"], r["python_src"],
                               qcheck.arg_order(r), SEEDS)
        except Exception as e:  # a reference or generator that raises
            res = {"ok": False, "reason": f"reference error: {type(e).__name__}"}
        return r["id"], res

    with ThreadPoolExecutor(16) as pool:
        results = dict(pool.map(one, recs))
    summary = Counter("pass" if v["ok"] else v.get("reason", "disagrees")
                      for v in results.values())
    OUT.write_text(json.dumps({"seeds": [SEEDS.start, SEEDS.stop - 1],
                               "summary": dict(summary), "results": results},
                              indent=1, default=str))
    print(dict(summary))
    for k, v in results.items():
        if not v["ok"]:
            print(" ", k, v.get("reason") or f"fails {v['n_fail']}/{v['n_cases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Re-verify every accepted problem on inputs it has never seen.

Acceptance used inputs drawn from each problem's own generator at one seed. That
proves the q solution matched the Python reference on *those* inputs. It does
not prove the pair agree in general — and the whole argument for this dataset
over Morgan Stanley's is that theirs was accepted on five fixed cases, one of
which hid a solution that is simply wrong.

So the same claim has to survive being turned on my own data: draw a fresh seed,
generate several times as many inputs, and re-run the differential test. A
problem that disagrees here was accepted on luck and must be quarantined.

No GPU, no model — just q, Python and the stored generators.

    python scripts/synth/audit_fresh.py --cases 60
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (ROOT, agrees, make_inputs, python_reference,  # noqa: E402
                      q_candidate)


def audit(rec: dict, cases: int, seed: int) -> tuple[str, dict]:
    inputs = make_inputs(rec["generator"], cases, seed=seed)
    if not inputs:
        return "generator_failed", {}
    expected = python_reference(rec["python_solution"], inputs)
    if expected is None:
        return "reference_failed", {}

    keep = [i for i, e in enumerate(expected)
            if not (isinstance(e, dict) and "__error__" in e)]
    if not keep:
        return "reference_errored_everywhere", {}
    n_err = len(expected) - len(keep)
    inputs = [inputs[i] for i in keep]
    expected = [expected[i] for i in keep]

    got = q_candidate(rec["q_solution"], inputs)
    if got is None:
        return "q_ran_badly", {}
    if len(got) != len(expected):
        return "q_wrong_arity", {}
    bad = [i for i, (e, g) in enumerate(zip(expected, got)) if not agrees(e, g)]
    if bad:
        i = bad[0]
        return "DISAGREED", {"n_bad": len(bad), "n_tested": len(expected),
                             "input": inputs[i], "expected": expected[i],
                             "got": got[i]}
    return "agreed", {"n_tested": len(expected), "n_reference_errors": n_err}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="corpus/qdataset.jsonl")
    ap.add_argument("--report", default="corpus/audit_fresh.json")
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260920)
    args = ap.parse_args()

    records = [json.loads(l) for l in (ROOT / args.src).open()]
    print(f"auditing {len(records)} problems on {args.cases} fresh inputs "
          f"(seed {args.seed}; acceptance used a different seed)", flush=True)

    counts: dict[str, int] = {}
    failures = []
    for i, rec in enumerate(records, 1):
        verdict, detail = audit(rec, args.cases, args.seed)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict != "agreed":
            failures.append({"id": rec["id"], "verdict": verdict,
                             "topic": rec.get("topic"),
                             "description": rec["description"][:200],
                             "q_solution": rec["q_solution"][:800], **detail})
        print(f"  [{i}/{len(records)}] {verdict}: "
              f"{' '.join(rec['description'].split())[:60]}", flush=True)

    report = {"n": len(records), "cases": args.cases, "seed": args.seed,
              "counts": counts, "failures": failures}
    (ROOT / args.report).write_text(json.dumps(report, indent=2, default=str))

    agreed = counts.get("agreed", 0)
    print(f"\nheld up on fresh inputs: {agreed}/{len(records)} "
          f"({agreed / len(records):.1%})")
    print(json.dumps(counts, indent=2))
    print("AUDIT_DONE", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

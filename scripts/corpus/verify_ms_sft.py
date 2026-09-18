#!/usr/bin/env python3
"""Check Morgan Stanley's released q solutions actually run and pass their tests.

Their pipeline built these by rejection sampling against a real q interpreter, so
they should pass. Two things are worth checking before training on them:

1. Do they still pass, here, under KDB-X? (Tooling drift, dialect differences.)
2. Do they pass because they are *correct*, or because five test cases are not
   many? The first row's solution returns false whenever the array sum is
   non-zero — wrong for the stated problem — yet its sample test has sum zero.

Runs each solution + test in a fresh q process so a crash or infinite loop costs
one problem, not the run.

    python scripts/corpus/verify_ms_sft.py --sample 60
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import tempfile
from pathlib import Path

Q = "/home/joedowling/.kx/bin/q"
ENV = {"QHOME": "/home/joedowling/.kx", "QLIC": "/home/joedowling/.kx",
       "PATH": "/home/joedowling/.kx/bin:/usr/bin:/bin", "HOME": "/home/joedowling"}


def run_q(script: str, timeout: float = 10.0) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as fh:
        fh.write(script + "\nexit 0;\n")
        path = fh.name
    try:
        r = subprocess.run([Q, path, "-q"], capture_output=True, text=True,
                           timeout=timeout, env=ENV)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--out", default="out/ms_sft_verification.json")
    args = ap.parse_args()

    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob("corpus/hf/sft-python-q-problems/**/*.parquet",
                                     recursive=True))])
    df = df.sample(n=min(args.sample, len(df)), random_state=0)

    results = []
    for _, row in df.iterrows():
        sol = str(row["q_solution"])
        passed = failed = errored = 0
        details = []
        for tc in row["test_cases"]:
            script = f"{sol}\n{tc['q_test_code']}"
            ok, out = run_q(script)
            expected = str(tc["q_expected_output"]).strip()
            got = out.strip().splitlines()[-1].strip() if out.strip() else ""
            if not ok or out.startswith("'"):
                errored += 1
                details.append({"expected": expected, "got": out[:80], "status": "error"})
            elif got == expected:
                passed += 1
            else:
                failed += 1
                details.append({"expected": expected, "got": got[:80], "status": "mismatch"})
        results.append({
            "problem_id": str(row.get("problem_id")), "leetcode_id": int(row["leetcode_id"]),
            "difficulty": str(row["difficulty"]),
            "passed": passed, "failed": failed, "errored": errored,
            "all_passed": failed == 0 and errored == 0,
            "details": details[:2],
        })
        mark = "ok " if results[-1]["all_passed"] else "FAIL"
        print(f"{mark} leetcode {results[-1]['leetcode_id']:>5} "
              f"{passed}/{passed+failed+errored} tests", flush=True)

    n_ok = sum(r["all_passed"] for r in results)
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"\n{n_ok}/{len(results)} solutions passed every one of their own tests")
    tot = sum(r["passed"] + r["failed"] + r["errored"] for r in results)
    print(f"test cases: {sum(r['passed'] for r in results)}/{tot} passed, "
          f"{sum(r['failed'] for r in results)} mismatched, "
          f"{sum(r['errored'] for r in results)} errored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

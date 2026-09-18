#!/usr/bin/env python3
"""Differentially test q solutions against a Python oracle on generated inputs.

The weakness in rejection-sampled data is that "verified" means "passed the five
test cases that came with it". Morgan Stanley's solution to LeetCode 1013 returns
false whenever the array sum is non-zero — wrong — and passes all five of its
tests because they all use zero-sum arrays.

This raises the bar: take each problem's Python reference, generate many fresh
inputs shaped like the original call, run both implementations, and compare. A q
solution is kept only if it agrees with Python everywhere.

Two details make the comparison reliable rather than a string-diff:

* q serialises its answer with `.j.j`, so results come back as JSON instead of
  `show` formatting (1b vs True, `1 2 3` vs [1,2,3]).
* Each problem runs as ONE q process and ONE Python process over all inputs,
  rather than a process per case — 678 spawns instead of 34,000.

    python scripts/synth/fuzz_verify.py --sample 100 --cases 25
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import subprocess
import tempfile
from pathlib import Path

Q = "/home/joedowling/.kx/bin/q"
QENV = {"QHOME": "/home/joedowling/.kx", "QLIC": "/home/joedowling/.kx",
        "PATH": "/home/joedowling/.kx/bin:/usr/bin:/bin", "HOME": "/home/joedowling"}
PY = "/usr/bin/python3"


# --------------------------------------------------------------- input shapes

def call_kwargs(test_code: str) -> dict | None:
    """Pull the keyword arguments out of `print(solve(arr = [...], k = 2))`."""
    try:
        tree = ast.parse(test_code.strip().splitlines()[0])
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "solve":
            out = {}
            for kw in node.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    return None
            return out or None
    return None


def numeric_domain(values) -> tuple[float, float]:
    """Observed numeric range across every seed case, flattened."""
    flat: list[float] = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            flat.append(float(v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for v in values:
        walk(v)
    if not flat:
        return -20.0, 20.0
    lo, hi = min(flat), max(flat)
    # Respect the sign convention the examples imply. LeetCode problems state
    # constraints like "1 <= nums[i] <= 10^9"; feeding negatives makes both
    # implementations undefined, and any disagreement then says nothing.
    if lo >= 1:
        lo = 1.0
    elif lo >= 0:
        lo = 0.0
    return lo, max(hi, lo + 1)


def mutate(value, rng: random.Random, dom: tuple[float, float], fixed_len: bool):
    """Generate a fresh value shaped like `value`, inside the observed domain."""
    lo, hi = dom
    if isinstance(value, bool):
        return rng.random() < 0.5
    if isinstance(value, int):
        return rng.randint(int(lo), max(int(lo), int(hi)))
    if isinstance(value, float):
        return round(rng.uniform(lo, hi), 3)
    if isinstance(value, str):
        if value and all(c.isalpha() for c in value):
            alphabet = sorted(set(value.lower())) or list("abcdefg")
            return "".join(rng.choice(alphabet) for _ in range(rng.randint(1, max(2, len(value)))))
        return value
    if isinstance(value, list):
        if not value:
            return []
        # Inner structures (edge pairs, [type,index,val] triples) must keep their
        # length; only the outer list is resized.
        n = len(value) if fixed_len else rng.randint(1, max(2, min(8, len(value) + 2)))
        return [mutate(value[i % len(value)], rng, dom, True) for i in range(n)]
    if isinstance(value, tuple):
        return tuple(mutate(v, rng, dom, True) for v in value)
    return value


# ------------------------------------------------------------------ q literals

def to_q(value) -> str:
    if isinstance(value, bool):
        return "1b" if value else "0b"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        # A one-character string is a char ATOM in q ("a"), not a string. The
        # list form needs enlist, or `count` and friends behave differently.
        lit = '"' + value.replace('"', '\\"') + '"'
        return f"enlist {lit}" if len(value) == 1 else lit
    if isinstance(value, (list, tuple)):
        if not value:
            return "()"
        if len(value) == 1:
            # Likewise (0) is the atom 0; a single-element list is `enlist 0`.
            return f"enlist {to_q(value[0])}"
        return "(" + ";".join(to_q(v) for v in value) + ")"
    raise TypeError(f"no q literal for {type(value).__name__}")


# ------------------------------------------------------------------- runners

def run_python(solution: str, inputs: list[dict], timeout: float) -> list | None:
    harness = [solution, "import json", "OUT=[]"]
    for kwargs in inputs:
        call = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
        harness.append(f"try:\n    OUT.append(solve({call}))\nexcept Exception as e:\n"
                       f"    OUT.append({{'__error__': type(e).__name__}})")
    harness.append("print('@@RESULT@@' + json.dumps(OUT, default=str))")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write("\n".join(harness))
        path = fh.name
    try:
        r = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout)
        for line in r.stdout.splitlines():
            if line.startswith("@@RESULT@@"):
                return json.loads(line[len("@@RESULT@@"):])
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


def run_q(solution: str, inputs: list[dict], timeout: float) -> list | None:
    lines = [solution, "out:();"]
    for kwargs in inputs:
        args = ";".join(to_q(v) for v in kwargs.values())
        # .Q.trp catches q errors so one bad input does not abort the batch.
        lines.append(f'out,:enlist @[{{.j.j solve[{args}]}};::;{{"__error__"}}];')
    lines.append('-1 "@@RESULT@@",.j.j out;')
    lines.append("exit 0;")
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as fh:
        fh.write("\n".join(lines))
        path = fh.name
    try:
        r = subprocess.run([Q, path, "-q"], capture_output=True, text=True,
                           timeout=timeout, env=QENV)
        for line in (r.stdout + r.stderr).splitlines():
            if line.startswith("@@RESULT@@"):
                raw = json.loads(line[len("@@RESULT@@"):])
                return [json.loads(x) if isinstance(x, str) and x != "__error__"
                        else {"__error__": "q"} for x in raw]
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, ValueError):
        return None
    finally:
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------- comparison

def agrees(py, qv) -> bool:
    if isinstance(py, dict) and "__error__" in py:
        return isinstance(qv, dict) and "__error__" in qv
    if isinstance(qv, dict) and "__error__" in qv:
        return False
    if isinstance(py, bool) or isinstance(qv, bool):
        return bool(py) == bool(qv)
    if isinstance(py, (int, float)) and isinstance(qv, (int, float)):
        return abs(float(py) - float(qv)) < 1e-6
    if isinstance(py, list) and isinstance(qv, list):
        return len(py) == len(qv) and all(agrees(a, b) for a, b in zip(py, qv))
    return str(py) == str(qv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--cases", type=int, default=25)
    ap.add_argument("--timeout", type=float, default=25.0)
    ap.add_argument("--out", default="out/fuzz_verify.json")
    args = ap.parse_args()

    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in
                    sorted(glob.glob("corpus/hf/sft-python-q-problems/**/*.parquet",
                                     recursive=True))])
    df = df.sample(n=min(args.sample, len(df)), random_state=1)

    rng = random.Random(0)
    results, skipped = [], 0
    for _, row in df.iterrows():
        seeds = [call_kwargs(tc["python_test_code"]) for tc in row["test_cases"]]
        seeds = [s for s in seeds if s]
        if not seeds:
            skipped += 1
            continue
        # Domain inferred per argument from the seed cases, so generated inputs
        # stay inside the problem's stated constraints.
        domains = {k: numeric_domain([s[k] for s in seeds if k in s]) for k in seeds[0]}
        inputs = list(seeds)
        while len(inputs) < args.cases:
            base = rng.choice(seeds)
            inputs.append({k: mutate(v, rng, domains.get(k, (-20.0, 20.0)), False)
                           for k, v in base.items()})

        py = run_python(str(row["python_solution"]), inputs, args.timeout)
        qv = run_q(str(row["q_solution"]), inputs, args.timeout)
        if py is None or qv is None or len(py) != len(qv):
            skipped += 1
            continue

        # An input the Python reference rejects is outside the problem's domain,
        # so it tests nothing: skip rather than score it as disagreement.
        valid = [i for i, p in enumerate(py)
                 if not (isinstance(p, dict) and "__error__" in p)]
        mismatches = [i for i in valid if not agrees(py[i], qv[i])]
        seeded = [i for i in mismatches if i < len(seeds)]
        rec = {
            "leetcode_id": int(row["leetcode_id"]), "difficulty": str(row["difficulty"]),
            "n_cases": len(inputs), "n_valid": len(valid), "n_mismatch": len(mismatches),
            "mismatch_on_original_tests": len(seeded),
            "example": ({"input": inputs[mismatches[0]],
                         "python": py[mismatches[0]], "q": qv[mismatches[0]]}
                        if mismatches else None),
        }
        results.append(rec)
        flag = "AGREE" if not mismatches else f"DIFFER on {len(mismatches)}/{len(valid)}"
        print(f"leetcode {rec['leetcode_id']:>5} {rec['difficulty']:<7} {flag}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1))
    agree = sum(1 for r in results if r["n_mismatch"] == 0)
    hidden = sum(1 for r in results
                 if r["n_mismatch"] and not r["mismatch_on_original_tests"])
    print(f"\n{agree}/{len(results)} q solutions match Python on all {args.cases} inputs")
    print(f"{hidden} passed their own five tests but fail on generated inputs")
    print(f"({skipped} skipped: unparseable call, timeout, or non-JSON result)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

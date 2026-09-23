#!/usr/bin/env python3
"""Rebuild Morgan Stanley's problems with our own references (amendment 6).

Their problem *descriptions* are permitted seeds; their Python and q are not
training material. For each problem, Qwen3.5-27B (an open general model) writes
a fresh `solve` and an input generator from the description alone. Morgan
Stanley's test cases then act as a check on our reference, never as training
text: a reference is kept only if it reproduces every one of their expected
outputs. The generator must also produce inputs the reference accepts, and the
answers must vary (no single answer in >90% of 60 draws, control 2).

The output is a problem bank in the same shape as the Q study bank, ready for
the teacher to solve:

    python scripts/synth/ms_reference.py --url http://127.0.0.1:8101/v1 \\
        --model qwen3.5-27b --out corpus/ms_rebuilt.jsonl
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
from pipeline import chat, fenced  # noqa: E402
import qcheck  # noqa: E402

PROMPT = """Here is a programming problem.

{description}

Write a Python reference solution and an input generator for it.

Reply with exactly two fenced blocks and nothing else:

```python
def solve({args}):
    <reference implementation>
```
```generator
def gen_input(rng):
    # return a dict of keyword arguments for solve(), always satisfying every
    # constraint in the problem, using rng (a random.Random) for variety.
    # Draw inputs that make the answer vary: mix cases where the answer is
    # True and False, empty and non-empty, small and large.
    return {{...}}
```

Rules: solve takes exactly the keyword arguments ({args}), is deterministic,
returns a JSON-serialisable value (lists, not tuples or sets), and runs in well
under a second on generated inputs. Keep generated inputs small (lists of at
most 30 elements, numbers of modest size). Standard library only."""


def ms_rows() -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "corpus/hf/sft-python-q-problems/data/*.parquet"))):
        rows += pq.read_table(p).to_pylist()
    return rows


def test_calls(row: dict) -> list[tuple[dict, str]] | None:
    """(kwargs, expected repr) for each Morgan Stanley test case."""
    out = []
    for tc in row["test_cases"]:
        try:
            call = ast.parse(tc["python_test_code"].strip().splitlines()[0]).body[0].value
            call = call.args[0] if getattr(call.func, "id", "") == "print" else call
            kwargs = {k.arg: ast.literal_eval(k.value) for k in call.keywords}
        except Exception:
            return None
        out.append((kwargs, tc["python_expected_output"].strip()))
    return out or None


def same_output(value, expected: str) -> bool:
    """Morgan Stanley stored `print(solve(...))` output, i.e. str(value)."""
    if str(value).strip() == expected:
        return True
    try:
        return value == ast.literal_eval(expected)
    except Exception:
        return False


def build(row: dict, url: str, model: str, attempts: int) -> dict:
    tests = test_calls(row)
    if not tests:
        return {"id": row["problem_id"], "ok": False, "reason": "tests unparseable"}
    args = list(tests[0][0])
    desc = row["problem_description"].strip()
    last = "no reply"
    for attempt in range(attempts):
        reply = chat(PROMPT.format(description=desc, args=", ".join(args)),
                     base_url=url, model=model, temperature=0.3 + 0.2 * attempt,
                     max_tokens=1500)
        py, gen = fenced(reply, "python"), fenced(reply, "generator")
        if not py or not gen:
            last = "missing block"
            continue
        src = py + "\n\n" + gen + "\n"
        g: dict = {}
        try:
            exec(src, g)
            if not all(same_output(g["solve"](**json.loads(json.dumps(kw))), exp)
                       for kw, exp in tests):
                last = "disagrees with Morgan Stanley's expected outputs"
                continue
            _, expected = qcheck.gen_inputs(src, args, range(700000, 700060))
        except Exception as e:
            last = f"raised {type(e).__name__}"
            continue
        share = qcheck.dominant_share(expected)
        if share > 0.9:
            last = f"generator not discriminating ({share:.2f})"
            continue
        return {"id": "ms__" + row["problem_id"], "ok": True, "attempt": attempt,
                "topic": ", ".join(row.get("tags") or []),
                "difficulty": (row.get("difficulty") or "").lower() or None,
                "description": desc, "argnames": args, "python_src": src,
                "dominant_share": share, "question_type": "ms-seed",
                "provenance": "description: morganstanley/sft-python-q-problems (seed, "
                              "disclosed); reference and generator: " + model}
    return {"id": "ms__" + row["problem_id"], "ok": False, "reason": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="corpus/ms_rebuilt.jsonl")
    args = ap.parse_args()

    rows = ms_rows()[: args.limit]
    done = set()
    out = ROOT / args.out
    if out.exists():  # resumable: a crash costs only the problems in flight
        done = {json.loads(l)["id"] for l in out.open()}
    todo = [r for r in rows if "ms__" + r["problem_id"] not in done]
    print(f"{len(rows)} problems, {len(done)} already done, {len(todo)} to go", flush=True)

    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for res in pool.map(lambda r: build(r, args.url, args.model, args.attempts), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(res["id"], "ok" if res["ok"] else res["reason"], flush=True)
    print("MS_REFERENCE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

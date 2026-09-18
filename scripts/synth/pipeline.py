#!/usr/bin/env python3
"""Synthetic q training data: generate, verify by execution, keep what survives.

Design follows one lesson from auditing Morgan Stanley's released data: five
fixed test cases per problem is not verification. Their LeetCode 1013 solution
returns false whenever the array sum is non-zero — wrong — and passes all five
of its tests because every one uses a zero-sum array.

Blind fuzzing does not fix that either: LeetCode problems carry cross-argument
constraints ("0 <= index < n", "edges reference nodes < n") that random inputs
violate, so most disagreements are invalid inputs rather than bugs.

So every problem here ships with **its own input generator**. Valid inputs exist
by construction, and a q solution is kept only if it agrees with the Python
reference on all of them.

    stage 1  model writes: description + python solution + gen_input()
    stage 2  reference self-check: the generator must produce inputs the
             reference handles without error (else the problem is discarded)
    stage 3  model writes q solutions, n samples at temperature
    stage 4  differential execution: q vs python over generated inputs
    stage 5  survivors become SFT records, deduped and contamination-checked

    python scripts/synth/pipeline.py --fixtures        # no GPU: prove 2,4,5
    python scripts/synth/pipeline.py --n-problems 50   # full run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
Q = "/home/joedowling/.kx/bin/q"
QENV = {"QHOME": "/home/joedowling/.kx", "QLIC": "/home/joedowling/.kx",
        "PATH": "/home/joedowling/.kx/bin:/usr/bin:/bin", "HOME": "/home/joedowling"}
PY = "/usr/bin/python3"

REPAIR_PROMPT = """This Python failed:

```python
{code}
```

Error:
```
{error}
```

Reply with exactly one corrected fenced ```python block and nothing else. Use only
the standard library, and keep the same function name and signature."""

PROBLEM_PROMPT = """Write one small programming problem suitable for testing a q/kdb+ programmer.

Reply with exactly three fenced blocks and nothing else:

```description
<one paragraph stating the task precisely, including any constraints>
```
```python
def solve(<args>):
    <reference implementation>
```
```generator
def gen_input(rng):
    # return a dict of keyword arguments for solve(), always satisfying the
    # stated constraints, using rng (a random.Random) for variety
    return {...}
```

Rules: solve() must be deterministic, return a JSON-serialisable value, and run
in well under a second. gen_input must never produce inputs that violate the
constraints. Prefer vector/table operations that suit q. Topic: {topic}"""

Q_PROMPT = """Write a q/kdb+ function that solves this problem.

{description}

Reply with exactly one fenced q block defining `solve`, and nothing else:

```q
solve:{{[{args}] ... }}
```
Use idiomatic q. Do not include tests or commentary."""

TOPICS = [
    "vector arithmetic", "string manipulation", "sorting and ranking",
    "time series windows", "table joins", "grouping and aggregation",
    "set operations", "running totals and deltas", "filtering with predicates",
    "type conversion and casting", "dictionaries and keyed lookup",
    "matrix and nested list handling",
]


# ------------------------------------------------------------------- helpers

def chat(prompt: str, *, base_url: str, model: str, temperature: float,
         max_tokens: int = 900) -> str:
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(f"{base_url}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


def fenced(text: str, tag: str) -> str | None:
    m = re.search(rf"```{tag}\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else None


def to_q_literal(value) -> str:
    if isinstance(value, bool):
        return "1b" if value else "0b"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        lit = '"' + value.replace('"', '\\"') + '"'
        return f"enlist {lit}" if len(value) == 1 else lit
    if isinstance(value, dict):
        if not value:
            return "()!()"
        keys = "`" + "`".join(str(k) for k in value)
        vals = "(" + ";".join(to_q_literal(v) for v in value.values()) + ")"
        return f"{keys}!{vals}"
    if isinstance(value, (list, tuple)):
        if not value:
            return "()"
        if len(value) == 1:                    # (0) is an atom; a list needs enlist
            return f"enlist {to_q_literal(value[0])}"
        return "(" + ";".join(to_q_literal(v) for v in value) + ")"
    raise TypeError(f"no q literal for {type(value).__name__}")


def run_python(code: str, timeout: float = 30.0) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(code)
        path = fh.name
    try:
        r = subprocess.run([PY, path], capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def run_q_script(code: str, timeout: float = 30.0) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".q", delete=False) as fh:
        fh.write(code + "\nexit 0;\n")
        path = fh.name
    try:
        r = subprocess.run([Q, path, "-q"], capture_output=True, text=True,
                           timeout=timeout, env=QENV)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    finally:
        Path(path).unlink(missing_ok=True)


def marker(out: str) -> list | None:
    for line in out.splitlines():
        if line.startswith("@@RESULT@@"):
            try:
                return json.loads(line[len("@@RESULT@@"):])
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------- stage 2 & 4

def make_inputs(generator: str, n: int, seed: int) -> list[dict] | None:
    """Stage 2: run the problem's own generator to get n valid input sets."""
    code = (f"import json, random\n{generator}\n"
            f"rng = random.Random({seed})\n"
            f"out = [gen_input(rng) for _ in range({n})]\n"
            "print('@@RESULT@@' + json.dumps(out, default=str))\n")
    ok, out = run_python(code)
    if not ok:
        return None
    got = marker(out)
    return got if isinstance(got, list) and all(isinstance(x, dict) for x in got) else None


def python_reference(solution: str, inputs: list[dict]) -> list | None:
    """Stage 2: the reference must handle its own generator's inputs."""
    lines = [solution, "import json", "OUT=[]"]
    for kw in inputs:
        call = ", ".join(f"{k}={v!r}" for k, v in kw.items())
        lines.append(f"try:\n    OUT.append(solve({call}))\n"
                     f"except Exception as e:\n    OUT.append({{'__error__': type(e).__name__}})")
    lines.append("print('@@RESULT@@' + json.dumps(OUT, default=str))")
    ok, out = run_python("\n".join(lines))
    return marker(out) if ok else None


def q_candidate(solution: str, inputs: list[dict]) -> list | None:
    """Stage 4: run the q candidate over the same inputs, results as JSON."""
    lines = [solution, "out:();"]
    for kw in inputs:
        args = ";".join(to_q_literal(v) for v in kw.values())
        lines.append(f'out,:enlist @[{{.j.j solve[{args}]}};::;{{"__error__"}}];')
    lines.append('-1 "@@RESULT@@",.j.j out;')
    ok, out = run_q_script("\n".join(lines))
    raw = marker(out) if ok else None
    if raw is None:
        return None
    parsed = []
    for x in raw:
        if isinstance(x, str) and x != "__error__":
            try:
                parsed.append(json.loads(x))
            except json.JSONDecodeError:
                parsed.append({"__error__": "json"})
        else:
            parsed.append({"__error__": "q"})
    return parsed


def agrees(py, qv) -> bool:
    if isinstance(py, dict) and "__error__" in py:
        return False                      # reference errors are filtered earlier
    if isinstance(qv, dict) and "__error__" in qv:
        return False
    if isinstance(py, bool) or isinstance(qv, bool):
        return bool(py) == bool(qv)
    if isinstance(py, (int, float)) and isinstance(qv, (int, float)):
        return abs(float(py) - float(qv)) < 1e-6
    if isinstance(py, list) and isinstance(qv, list):
        return len(py) == len(qv) and all(agrees(a, b) for a, b in zip(py, qv))
    return str(py) == str(qv)


# ------------------------------------------------------------------ stage 5

def humaneval_texts() -> list[str]:
    path = ROOT / "q-evaluation-harness/datasets/q_humaneval.jsonl"
    return [json.loads(l)["prompt"] for l in path.open()] if path.exists() else []


def contaminated(description: str, benchmark: list[str], threshold: float = 0.6) -> bool:
    """Reject anything resembling a benchmark problem, by token overlap."""
    words = set(re.findall(r"[a-z]{4,}", description.lower()))
    if not words:
        return False
    for text in benchmark:
        other = set(re.findall(r"[a-z]{4,}", text.lower()))
        if other and len(words & other) / min(len(words), len(other)) > threshold:
            return True
    return False


# --------------------------------------------------------------------- main

def process_problem(spec: dict, args, benchmark: list[str], stats: dict,
                    rejects: list) -> dict | None:
    desc, py_sol, gen = spec["description"], spec["python"], spec["generator"]

    seed = abs(hash(desc)) % 10_000
    inputs = make_inputs(gen, args.cases, seed=seed)
    if not inputs and spec.get("repair"):
        # One repair attempt: a generator that references undefined names is a
        # weak-model artefact, not a reason to throw the problem away.
        _, err = run_python(f"import random\n{gen}\ngen_input(random.Random(0))")
        fixed = spec["repair"](gen, err[-800:])
        if fixed:
            gen = fixed
            inputs = make_inputs(gen, args.cases, seed=seed)
            if inputs:
                stats["repaired"] += 1
    if not inputs:
        stats["generator_failed"] += 1
        rejects.append({"reason": "generator_failed", "description": desc[:300],
                        "generator": gen[:600]})
        return None

    expected = python_reference(py_sol, inputs)
    if expected is None:
        stats["reference_failed"] += 1
        return None
    valid = [i for i, e in enumerate(expected)
             if not (isinstance(e, dict) and "__error__" in e)]
    if len(valid) < max(5, args.cases // 2):
        stats["generator_invalid_inputs"] += 1
        return None
    inputs = [inputs[i] for i in valid]
    expected = [expected[i] for i in valid]

    if contaminated(desc, benchmark):
        stats["contaminated"] += 1
        return None

    # Discriminating power. Random inputs can be uninformative: for the
    # three-equal-parts problem, random arrays almost never partition, so a
    # solution that always answers "false" agrees with the reference on every
    # case. If the reference's own answers barely vary, the inputs cannot
    # separate a right solution from a wrong one, and passing them means little.
    distinct = {json.dumps(e, sort_keys=True) for e in expected}
    if len(distinct) < 2:
        stats["uninformative_inputs"] += 1
        return None
    commonest = max(sum(1 for e in expected if json.dumps(e, sort_keys=True) == d)
                    for d in distinct)
    if commonest / len(expected) > 0.9:
        stats["inputs_too_skewed"] += 1
        return None

    for attempt, q_sol in enumerate(spec["q_candidates"]):
        got = q_candidate(q_sol, inputs)
        if got is None or len(got) != len(expected):
            stats["q_ran_badly"] += 1
            continue
        bad = [i for i, (e, g) in enumerate(zip(expected, got)) if not agrees(e, g)]
        if bad:
            stats["q_disagreed"] += 1
            continue
        stats["kept"] += 1
        return {
            "description": desc, "python_solution": py_sol, "generator": gen,
            "q_solution": q_sol, "n_cases": len(inputs), "attempt": attempt,
            "inputs_sample": inputs[:3], "expected_sample": expected[:3],
            "id": hashlib.sha256(desc.encode()).hexdigest()[:16],
        }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true",
                    help="run stages 2/4/5 on hand-written problems, no model needed")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--samples", type=int, default=4, help="q attempts per problem")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--base-url", default=os.environ.get("SYNTH_BASE_URL",
                                                         "http://127.0.0.1:8100/v1"))
    ap.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "qwen3.5-27b"))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="corpus/synthetic.jsonl")
    args = ap.parse_args()

    benchmark = humaneval_texts()
    stats = {k: 0 for k in ["generator_failed", "reference_failed",
                            "generator_invalid_inputs", "contaminated",
                            "uninformative_inputs", "inputs_too_skewed",
                            "q_ran_badly", "q_disagreed", "kept", "spec_unparsed",
                            "repaired"]}
    rejects: list = []
    specs: list[dict] = []

    if args.fixtures:
        specs = json.loads((ROOT / "scripts/synth/fixtures.json").read_text())
    else:
        rng = random.Random()
        for i in range(args.n_problems):
            topic = TOPICS[i % len(TOPICS)]
            raw = chat(PROBLEM_PROMPT.replace("{topic}", topic), base_url=args.base_url,
                       model=args.model, temperature=args.temperature)
            desc, py_sol, gen = (fenced(raw, "description"), fenced(raw, "python"),
                                 fenced(raw, "generator"))
            if not (desc and py_sol and gen):
                stats["spec_unparsed"] += 1
                continue
            m = re.search(r"def solve\(([^)]*)\)", py_sol)
            arg_names = [a.split(":")[0].strip() for a in (m.group(1) if m else "").split(",") if a.strip()]
            q_prompt = Q_PROMPT.format(description=desc, args=";".join(arg_names))
            cands = [chat(q_prompt, base_url=args.base_url, model=args.model,
                          temperature=args.temperature) for _ in range(args.samples)]
            specs.append({"description": desc, "python": py_sol, "generator": gen,
                          "q_candidates": [fenced(c, "q") or c for c in cands],
                          "repair": lambda code, err: fenced(
                              chat(REPAIR_PROMPT.format(code=code, error=err),
                                   base_url=args.base_url, model=args.model,
                                   temperature=0.3), "python")})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    with out_path.open("a") as fh:
        for spec in specs:
            rec = process_problem(spec, args, benchmark, stats, rejects)
            if rec:
                fh.write(json.dumps(rec) + "\n")
                kept.append(rec)
            print(f"  {'KEPT' if rec else 'drop'}: "
                  f"{' '.join(spec['description'].split())[:70]}", flush=True)

    print("\nstage outcomes:")
    for k, v in stats.items():
        if v:
            print(f"  {k:26} {v}")
    print(f"kept {len(kept)}/{len(specs)} problems -> {out_path}")
    if rejects:
        rej = out_path.with_name(out_path.stem + "_rejects.jsonl")
        with rej.open("a") as fh:
            for r in rejects:
                fh.write(json.dumps(r) + "\n")
        print(f"kept {len(rejects)} rejected specs for inspection -> {rej}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

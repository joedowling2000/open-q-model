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
from concurrent.futures import ThreadPoolExecutor
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

The description must be unambiguous about boundary conditions: state explicitly
whether ranges/windows include their endpoints, what happens on empty or
single-element input, and how ties are broken.

Rules: solve() must be deterministic, return a JSON-serialisable value, and run
in well under a second. gen_input must never produce inputs that violate the
constraints. Use only the Python standard library — no pandas, no numpy.
Prefer vector/table operations that suit q.

Topic: {topic}
Difficulty: {level}"""

# The q author sees the Python reference, not just the prose. Translating a
# concrete implementation removes the spec ambiguity that produced most
# disagreements in the first trial — e.g. whether a sliding window includes both
# endpoints, which the description left open and the reference settles.
Q_PROMPT = """Translate this Python function into q/kdb+.

Problem: {description}

Reference implementation, whose behaviour you must match exactly, including edge
cases and boundary conditions:

```python
{python}
```

Reply with exactly one fenced q block defining `solve`, and nothing else:

```q
solve:{{[{args}] ... }}
```

Requirements:
- Match the reference's output exactly, including types (a list of strings stays
  a list of strings, not a single joined string).
- Write idiomatic vector q. Avoid `while` and `do` loops; prefer q's vector
  primitives and iterators (each, over, scan, mavg, deltas, where, group).
- No tests, no commentary, no explanation."""

REWRITE_PROMPT = """This q function is correct but written in an imperative style.

```q
{solution}
```

Rewrite it in idiomatic vector q with identical behaviour: no `while`, no `do`,
no manual index counters. Use q's iterators and primitives (each, over, scan,
where, group, sums, deltas, mavg, cut, flip) instead.

Reply with exactly one fenced q block defining `solve`, and nothing else."""

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


class UnrepresentableInput(TypeError):
    """An input value that cannot be faithfully expressed as a q literal."""


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
    # No faithful q literal exists for this Python value (None being the common
    # case). Guessing a mapping — 0N? 0n? :: ? — could make a correct solution
    # look wrong, so the problem is skipped instead.
    raise UnrepresentableInput(f"no q literal for {type(value).__name__}")


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


def q_candidate(solution: str, inputs: list[dict], debug: bool = False):
    """Stage 4: run the q candidate over the same inputs, results as JSON."""
    lines = [solution, "out:();"]
    for kw in inputs:
        args = ";".join(to_q_literal(v) for v in kw.values())
        lines.append(f'out,:enlist @[{{.j.j solve[{args}]}};::;{{"__error__"}}];')
    lines.append('-1 "@@RESULT@@",.j.j out;')
    ok, out = run_q_script("\n".join(lines))
    raw = marker(out) if ok else None
    if raw is None:
        return (None, out) if debug else None
    parsed = []
    for x in raw:
        if isinstance(x, str) and x != "__error__":
            try:
                parsed.append(json.loads(x))
            except json.JSONDecodeError:
                parsed.append({"__error__": "json"})
        else:
            parsed.append({"__error__": "q"})
    return (parsed, out) if debug else parsed


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

    try:
        to_q_literals_ok = all(to_q_literal(v) for kw in inputs for v in kw.values())
    except UnrepresentableInput:
        stats["input_not_representable"] += 1
        return None

    # Whether a solution uses loops is visible in the text, so order the
    # candidates vector-first and stop at the first that verifies. Verifying all
    # six to pick the nicest one was three times the q execution for the same
    # data (batch 2 yield fell from 13/hour to 6/hour); this gets the idiom
    # preference for free.
    ordered = sorted(enumerate(spec["q_candidates"]),
                     key=lambda t: (bool(re.search(r"\b(while|do)\[", t[1])), len(t[1])))
    passing: list[tuple[int, str, bool]] = []
    for attempt, q_sol in ordered:
        got, raw_out = q_candidate(q_sol, inputs, debug=True)
        if got is None or len(got) != len(expected):
            stats["q_ran_badly"] += 1
            rejects.append({"reason": "q_ran_badly", "description": desc[:200],
                            "q_solution": q_sol[:600], "q_output": (raw_out or "")[-600:]})
            continue
        bad = [i for i, (e, g) in enumerate(zip(expected, got)) if not agrees(e, g)]
        if bad:
            stats["q_disagreed"] += 1
            rejects.append({"reason": "q_disagreed", "description": desc[:200],
                            "q_solution": q_sol[:600],
                            "input": inputs[bad[0]], "expected": expected[bad[0]],
                            "got": got[bad[0]]})
            continue
        passing.append((attempt, q_sol, bool(re.search(r"\b(while|do)\[", q_sol))))
        break                      # ordered vector-first, so the first pass is the best

    if not passing:
        return None

    # Prefer vector style; among equals prefer the shorter solution.
    attempt, q_sol, uses_loops = sorted(passing, key=lambda t: (t[2], len(t[1])))[0]

    # If the only correct solutions are imperative, ask for a vector rewrite and
    # verify it the same way. qqWen learned q from LeetCode translations, so 79%
    # of accepted solutions arrived as while loops — training on those would
    # teach Python-shaped q. The rewrite is kept only if it passes; otherwise the
    # loop version stands.
    if uses_loops and spec.get("rewrite"):
        rewritten = spec["rewrite"](q_sol)
        if rewritten and not re.search(r"\b(while|do)\[", rewritten):
            got = q_candidate(rewritten, inputs)
            if got is not None and len(got) == len(expected) and all(
                    agrees(e, g) for e, g in zip(expected, got)):
                q_sol, uses_loops = rewritten, False
                stats["rewritten_to_vector"] += 1
    if uses_loops:
        stats["kept_but_imperative"] += 1
    stats["kept"] += 1
    if True:
        return {
            "description": desc, "python_solution": py_sol, "generator": gen,
            "q_solution": q_sol, "n_cases": len(inputs), "attempt": attempt,
            "n_passing_attempts": len(passing),
            "topic": spec.get("topic"), "difficulty": spec.get("difficulty"),
            "problem_model": spec.get("problem_model"), "q_model": spec.get("q_model"),
            "uses_loops": uses_loops,
            "inputs_sample": inputs[:3], "expected_sample": expected[:3],
            "id": hashlib.sha256(desc.encode()).hexdigest()[:16],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", action="store_true",
                    help="run stages 2/4/5 on hand-written problems, no model needed")
    ap.add_argument("--n-problems", type=int, default=20)
    ap.add_argument("--samples", type=int, default=4, help="q attempts per problem")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--base-url", default=os.environ.get("SYNTH_BASE_URL",
                                                         "http://127.0.0.1:8100/v1"),
                    help="endpoint of the PROBLEM author (general reasoning)")
    ap.add_argument("--model", default=os.environ.get("SYNTH_MODEL", "qwen3.5-27b"))
    # Writing a good problem is general reasoning; writing good q is domain
    # skill. Splitting them lets the best q model we have write the solutions.
    ap.add_argument("--q-url", default=os.environ.get("SYNTH_Q_URL",
                                                      "http://127.0.0.1:8101/v1"),
                    help="endpoint of the q author (best measured q model)")
    ap.add_argument("--q-model", default=os.environ.get("SYNTH_Q_MODEL", "qqwen-32b-rl"))
    ap.add_argument("--parallel", type=int, default=4,
                    help="problems written concurrently (keep <= server slots)")
    ap.add_argument("--difficulty", default="easy,easy,medium,medium,hard",
                    help="ladder cycled across problems")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="corpus/synthetic.jsonl")
    args = ap.parse_args()

    benchmark = humaneval_texts()
    stats = {k: 0 for k in ["generator_failed", "reference_failed",
                            "generator_invalid_inputs", "contaminated",
                            "uninformative_inputs", "inputs_too_skewed",
                            "q_ran_badly", "q_disagreed", "kept", "spec_unparsed",
                            "repaired", "input_not_representable", "crashed",
                            "q_unfenced", "kept_but_imperative",
                            "rewritten_to_vector"]}
    rejects: list = []
    specs: list[dict] = []

    if args.fixtures:
        specs = json.loads((ROOT / "scripts/synth/fixtures.json").read_text())
    else:
        rng = random.Random()
        levels = [d.strip() for d in args.difficulty.split(",") if d.strip()]

        def write_problem(i: int) -> tuple | None:
            """One problem spec. Retries once if the model ignores the format."""
            topic, level = TOPICS[i % len(TOPICS)], levels[i % len(levels)]
            prompt = PROBLEM_PROMPT.replace("{topic}", topic).replace("{level}", level)
            for attempt in range(2):
                raw = chat(prompt, base_url=args.base_url, model=args.model,
                           temperature=args.temperature)
                parts = (fenced(raw, "description"), fenced(raw, "python"),
                         fenced(raw, "generator"))
                if all(parts):
                    return (topic, level, *parts)
            return None

        # Both servers have several slots; issuing calls one at a time left them
        # mostly idle. Problems are written concurrently, then each problem's q
        # attempts are issued concurrently too.
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            written = list(pool.map(write_problem, range(args.n_problems)))

        for item in written:
            if item is None:
                stats["spec_unparsed"] += 1
                continue
            topic, level, desc, py_sol, gen = item
            m = re.search(r"def solve\(([^)]*)\)", py_sol)
            arg_names = [a.split(":")[0].strip() for a in (m.group(1) if m else "").split(",") if a.strip()]
            q_prompt = Q_PROMPT.format(description=desc, python=py_sol,
                                       args=";".join(arg_names))
            with ThreadPoolExecutor(max_workers=args.samples) as qpool:
                cands = list(qpool.map(
                    lambda _: chat(q_prompt, base_url=args.q_url, model=args.q_model,
                                   temperature=args.temperature),
                    range(args.samples)))
            parsed_q = [fenced(c, "q") for c in cands]
            stats["q_unfenced"] += sum(1 for x in parsed_q if x is None)
            specs.append({"topic": topic, "difficulty": level,
                          "problem_model": args.model, "q_model": args.q_model,
                          "description": desc, "python": py_sol, "generator": gen,
                          "q_candidates": [x or c for x, c in zip(parsed_q, cands)],
                          "rewrite": lambda sol: fenced(
                              chat(REWRITE_PROMPT.format(solution=sol),
                                   base_url=args.q_url, model=args.q_model,
                                   temperature=0.4), "q"),
                          "repair": lambda code, err: fenced(
                              chat(REPAIR_PROMPT.format(code=code, error=err),
                                   base_url=args.base_url, model=args.model,
                                   temperature=0.3), "python")})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept = []
    with out_path.open("a") as fh:
        for spec in specs:
            # One bad problem must not take the batch down with it: a TypeError
            # on a single generated input cost ~20 problems of GPU work.
            try:
                rec = process_problem(spec, args, benchmark, stats, rejects)
            except Exception as exc:
                stats["crashed"] += 1
                rejects.append({"reason": f"crashed: {type(exc).__name__}: {exc}",
                                "description": spec["description"][:300]})
                rec = None
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

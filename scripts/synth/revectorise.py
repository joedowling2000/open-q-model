#!/usr/bin/env python3
"""Rewrite imperative q solutions into vector q, keeping only verified rewrites.

Why this exists as a separate pass: the generation pipeline already had a
rewrite step, and it failed on all 85 of the 85 loop-shaped solutions it was
given. The reason is visible in hindsight — the rewriter was qqWen, the same
model whose LeetCode-translation training gave it the loop habit in the first
place. Asking it to devectorise its own reflex, once, at temperature 0.4, was
never going to work.

This pass changes three things and nothing else:

  * a different rewriter (pass --url/--model; the intended one is the q-adapted
    Qwen3.5-27B checkpoint, i.e. the model being improved rewriting its own
    training data — self-improvement under the pre-registered lineage rule, not
    distillation from a stronger outside model),
  * several attempts up a temperature ladder instead of one shot,
  * a prompt that carries the problem description and the Python reference, so
    the model rewrites the *intent* rather than transliterating the loop.

Verification is unchanged and non-negotiable: a rewrite is kept only if it
agrees with the Python reference on freshly generated inputs — more of them than
the original acceptance used, so a kept rewrite is held to a stricter standard
than the solution it replaces. Anything else leaves the loop version standing.

    python scripts/synth/revectorise.py --url http://127.0.0.1:8102/v1 \
        --model qwen3.5-27b-cpt --in corpus/qdataset.jsonl \
        --out corpus/qdataset.vector.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import (ROOT, agrees, chat, fenced, make_inputs,  # noqa: E402
                      python_reference, q_candidate)

LOOPY = re.compile(r"\b(while|do)\[")

PROMPT = """You are writing idiomatic q (kdb+).

Problem:
{description}

A correct Python reference:
```python
{python}
```

A correct but imperative q solution:
```q
{imperative}
```

Rewrite the q solution in idiomatic vector style with identical behaviour.
Hard requirements:
  * no `while`, no `do`, no manual index counters, no accumulator variables
    appended to in a loop;
  * express the computation with q's iterators and primitives — each, each-left,
    each-right, over (/), scan (\\), where, group, sums, deltas, prds, mavg,
    msum, cut, flip, til, except, inter, asc, iasc, rank, fills, next, prev;
  * keep the same function name `solve` and the same parameter names and order;
  * handle the empty and boundary cases the imperative version handles.

Think about what the computation *is*, not about translating the loop
statement by statement. Reply with exactly one fenced q block and nothing else.
"""


def rewrite_one(rec: dict, args) -> tuple[str | None, str]:
    """Return (verified vector solution, reason) for one record."""
    gen, py_sol = rec["generator"], rec["python_solution"]

    # Fresh inputs, and more of them than acceptance used: a rewrite that
    # survives this has been tested harder than the solution it replaces.
    inputs = make_inputs(gen, args.cases, seed=args.seed)
    if not inputs:
        return None, "generator failed"
    expected = python_reference(py_sol, inputs)
    if expected is None:
        return None, "reference failed"
    keep = [i for i, e in enumerate(expected)
            if not (isinstance(e, dict) and "__error__" in e)]
    if len(keep) < args.cases // 2:
        return None, "reference errored on most inputs"
    inputs = [inputs[i] for i in keep]
    expected = [expected[i] for i in keep]

    # Re-verify the *existing* solution on the fresh inputs first. If the loop
    # version disagrees here, the record itself is suspect and no rewrite of it
    # should be trusted — that is a finding, not a rewrite failure.
    base_got = q_candidate(rec["q_solution"], inputs)
    if base_got is None or len(base_got) != len(expected) or not all(
            agrees(e, g) for e, g in zip(expected, base_got)):
        return None, "ORIGINAL FAILED on fresh inputs"

    prompt = PROMPT.format(description=rec["description"],
                           python=py_sol, imperative=rec["q_solution"])
    for attempt, temp in enumerate(args.temps):
        try:
            cand = fenced(chat(prompt, base_url=args.url, model=args.model,
                               temperature=temp, max_tokens=1200), "q")
        except Exception as exc:
            return None, f"request failed: {type(exc).__name__}"
        if not cand:
            continue
        if LOOPY.search(cand):
            continue
        got = q_candidate(cand, inputs)
        if got is None or len(got) != len(expected):
            continue
        if all(agrees(e, g) for e, g in zip(expected, got)):
            return cand, f"verified on attempt {attempt + 1} (t={temp})"
    return None, "no attempt verified"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8102/v1")
    ap.add_argument("--model", default="qwen3.5-27b-cpt")
    ap.add_argument("--in", dest="src", default="corpus/qdataset.jsonl")
    ap.add_argument("--out", dest="dst", default="corpus/qdataset.vector.jsonl")
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--seed", type=int, default=99991)
    ap.add_argument("--temps", type=float, nargs="+",
                    default=[0.2, 0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = [json.loads(l) for l in (ROOT / args.src).open()]
    todo = [r for r in records if r.get("uses_loops")]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(records)} records, {len(todo)} imperative, "
          f"{args.cases} fresh cases each, {len(args.temps)} attempts", flush=True)

    fixed, reasons, suspect = {}, {}, []
    for i, rec in enumerate(todo, 1):
        vector, reason = rewrite_one(rec, args)
        reasons[reason.split(" on attempt")[0]] = \
            reasons.get(reason.split(" on attempt")[0], 0) + 1
        if reason.startswith("ORIGINAL FAILED"):
            suspect.append(rec["id"])
        if vector:
            fixed[rec["id"]] = (vector, reason)
        print(f"  [{i}/{len(todo)}] {'VECTOR' if vector else 'keep loop'}: "
              f"{reason} :: {' '.join(rec['description'].split())[:60]}", flush=True)

    out_path = ROOT / args.dst
    with out_path.open("w") as fh:
        for rec in records:
            if rec["id"] in fixed:
                vector, reason = fixed[rec["id"]]
                rec = dict(rec)
                rec["q_solution_imperative"] = rec["q_solution"]
                rec["q_solution"] = vector
                rec["uses_loops"] = False
                rec["revectorised_by"] = args.model
                rec["revectorise_note"] = reason
            rec["revectorise_suspect"] = rec["id"] in suspect
            fh.write(json.dumps(rec) + "\n")

    n_loop_before = sum(1 for r in records if r.get("uses_loops"))
    print(f"\nrewritten to vector: {len(fixed)}/{len(todo)}")
    print(f"imperative share: {n_loop_before}/{len(records)} -> "
          f"{n_loop_before - len(fixed)}/{len(records)}")
    if suspect:
        print(f"SUSPECT (original failed fresh inputs): {len(suspect)} -> {suspect}")
    print("reasons:", json.dumps(reasons, indent=2))
    print("REVECTORISE_DONE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

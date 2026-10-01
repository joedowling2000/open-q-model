#!/usr/bin/env python3
"""Amendment 10 (i): render a verified problem in the harness's own prompt format.

The harness (src/prompts/jinja_templates/q_humaneval.j2) sends, verbatim:

    \"\"\"You are an expert q/kdb+ programmer.
    Write idiomatic q function to complete the following task:
    // @overview
    // <description>
    //
    // @param x {int[]}  <what x is>
    //
    // @return {int}   <what comes back>
    name:{[x]
        / function body
        }
    \"\"\"

and every training prompt so far was prose asking for `solve`. This builds the
same shape from a training record: a named function, x/y/z parameters when
there are at most three (as in every benchmark prompt), and type tags in the
benchmark's own vocabulary (int, int[], string, string[], float, boolean, dict,
...) inferred from the problem's generated inputs and outputs.

The response is the record's verified q, renamed (solve -> name, original
parameters -> x/y/z) and verified again. A rename that would capture an
existing identifier, or that fails verification, is dropped, and that record
keeps only its original format. Only the template is borrowed; no benchmark
problem, test or solution is used.
"""

from __future__ import annotations

import json
import random
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
import qcheck  # noqa: E402

PREAMBLE = ('"""You are an expert q/kdb+ programmer.\n'
            'Write idiomatic q function to complete the following task:\n')
STOP = {"given", "list", "return", "write", "function", "that", "which", "with", "from",
        "each", "into", "their", "this", "your", "should", "integer", "integers", "string",
        "strings", "number", "numbers", "where", "when", "will", "have", "value", "values",
        "records", "record", "input", "output", "returns", "compute", "find"}
RESERVED = set(json.loads((ROOT / "scripts/synth/q_reserved.json").read_text()))


def type_tag(v) -> str:
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "string"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, (list, tuple)):
        inner = {type_tag(x) for x in v}
        if len(inner) == 1 and next(iter(inner)) in ("int", "float", "string", "boolean"):
            return next(iter(inner)) + "[]"
        if inner <= {"int", "float"} and inner:
            return "float[]"
        return "list"
    return "any"


def clean_description(text: str) -> str:
    """Drop Morgan Stanley's translation boilerplate ("### Format: ... starter code",
    "NOTE: This problem is described for Python ..."): it is their instruction to
    the q writer, not the problem, and it asks for `solve`."""
    for marker in ("### Format:", "NOTE: This problem is described for Python"):
        i = text.find(marker)
        if i > 0:
            text = text[:i]
    return text.strip()


def function_name(rec: dict) -> str:
    # Morgan Stanley and MBPP problems carry a real name; the rest get one from
    # the description's first distinctive words.
    rid = rec["id"]
    if rid.startswith("ms__"):
        name = rid[4:]
    elif rec.get("function_name"):
        name = rec["function_name"]
    else:
        words = [w for w in re.findall(r"[a-z]{4,}", clean_description(rec["description"]).lower())
                 if w not in STOP]
        name = "_".join(words[:3]) or "compute"
    name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40]
    if not name or name[0].isdigit() or name in RESERVED:
        name = "f_" + name
    return name


def render(rec: dict, python_src: str, order: list[str], seeds=range(1300000, 1300040)):
    """Return (prompt, response) in the harness format, or None."""
    q = rec["q_solution"]
    if not q.lstrip().startswith("solve:"):
        return None
    new_args = ["x", "y", "z"][: len(order)] if len(order) <= 3 else list(order)
    # Refuse a rename that would capture an identifier already in the code.
    tokens = set(re.findall(r"(?<![\w`.])[A-Za-z]\w*", q))
    if any(a in tokens and a not in order for a in new_args):
        return None
    name = function_name(rec)
    body = re.sub(r"^\s*solve:", f"{name}:", q, count=1)
    for old, new in zip(order, new_args):
        if old != new:
            body = re.sub(rf"(?<![\w`.]){re.escape(old)}(?![\w])", new, body)
    # Verify the renamed function (qcheck calls `solve`, so alias it).
    res = qcheck.check(body + f"\nsolve:{name};", python_src, order, seeds)
    if not res["ok"]:
        return None

    inputs, outputs = qcheck.gen_inputs(python_src, order, range(1300100, 1300103))
    overview = textwrap.wrap(" ".join(clean_description(rec["description"]).split()), 88)
    lines = ["// @overview"] + [f"// {l}" for l in overview] + ["//"]
    for arg, old in zip(new_args, order):
        lines.append(f"// @param {arg} {{{type_tag(inputs[0][old])}}}  {old.replace('_', ' ')}")
    lines += ["//", f"// @return {{{type_tag(outputs[0])}}}   result",
              f"{name}:{{[{';'.join(new_args)}]", "    / function body", "    }"]
    prompt = PREAMBLE + "\n".join(lines) + '\n"""'
    response = f"```q\n{body.strip()}\n```"
    return prompt, response


if __name__ == "__main__":
    # Smoke test on a few round 3 records.
    recs = {}
    for f in ("questions.jsonl", "reserve_exercises.jsonl"):
        for l in (ROOT / "corpus/q_study_v5" / f).open():
            r = json.loads(l)
            recs[r["id"]] = r
    for f in ("distilled.jsonl", "generated_problems.jsonl"):
        for l in (ROOT / "corpus" / f).open():
            r = json.loads(l)
            if r.get("id") and r.get("python_src"):
                recs[r["id"]] = r
    rows = [json.loads(l) for l in (ROOT / "corpus/round3_records.jsonl").open()]
    random.Random(0).shuffle(rows)
    ok = 0
    for rec in rows[:12]:
        src = recs[rec["id"]]
        out = render(rec, src["python_src"], qcheck.arg_order(src))
        ok += out is not None
        if out and ok <= 2:
            print(out[0]); print(out[1]); print("-" * 60)
    print(f"rendered {ok} of 12")

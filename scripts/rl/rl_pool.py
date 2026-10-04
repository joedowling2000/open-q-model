#!/usr/bin/env python3
"""RL prompt pool (amendments 4, 10): every usable problem, as a benchmark-format prompt.

A problem needs only a description, a Python reference and an input generator.
No q solution is required, which is why RL can use the generated problems that
nobody solved. Each is rendered in the harness's template: named function,
x/y/z parameters, type tags from its own generated inputs.

Excluded: held-out questions (amendment 7) and their textual near-variants
(incident 4), records that failed the release check or re-verification, and
anything with more than three arguments (no benchmark prompt has more).

    python scripts/rl/rl_pool.py       # -> corpus/rl_pool.jsonl
"""
from __future__ import annotations

import json
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
sys.path.insert(0, str(ROOT / "scripts/train"))
import qcheck  # noqa: E402
from harness_format import PREAMBLE, clean_description, function_name, type_tag  # noqa: E402

RESERVED = set(json.loads((ROOT / "scripts/synth/q_reserved.json").read_text()))
SOURCE_WEIGHT = {"mbpp": 1.5, "ms": 1.5, "gen": 1.0, "bank": 1.0}   # amendment 10 (v)


def words(t: str) -> frozenset:
    return frozenset(re.findall(r"[a-z]{4,}", t.lower()))


def prompt_for(rec: dict, order: list[str]) -> tuple[str, str] | None:
    if len(order) > 3:
        return None
    inputs, outputs = qcheck.gen_inputs(rec["python_src"], order, range(1500000, 1500003))
    name = function_name(rec)
    args = ["x", "y", "z"][: len(order)]
    over = textwrap.wrap(" ".join(clean_description(rec["description"]).split()), 88)
    lines = ["// @overview"] + [f"// {l}" for l in over] + ["//"]
    lines += [f"// @param {a} {{{type_tag(inputs[0][o])}}}  {o.replace('_', ' ')}" for a, o in zip(args, order)]
    lines += ["//", f"// @return {{{type_tag(outputs[0])}}}   result",
              f"{name}:{{[{';'.join(args)}]", "    / function body", "    }"]
    return PREAMBLE + "\n".join(lines) + '\n"""', name


def main() -> int:
    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    bank = [json.loads(l) for l in (ROOT / "corpus/q_study_v5/questions.jsonl").open()]
    held_w = [words(r["description"]) for r in bank if r["id"] in held]
    dq = set()
    p = ROOT / "corpus/disqualified.json"
    if p.exists():
        dq |= set(json.loads(p.read_text())["disqualified"])
    p = ROOT / "corpus/vector_rewrites.jsonl"
    if p.exists():
        dq |= {json.loads(l)["id"] for l in p.open() if json.loads(l).get("reason", "").startswith("ORIGINAL FAILED")}

    recs = [(r, "bank") for r in bank if r["question_type"] == "original"]
    for f, src in (("generated_problems.jsonl", "gen"), ("mbpp_rebuilt.jsonl", "mbpp"), ("ms_rebuilt.jsonl", "ms")):
        recs += [(json.loads(l), src) for l in (ROOT / "corpus" / f).open()]
    stats, out = Counter(), []
    for r, src in recs:
        if not r.get("ok", True) or not r.get("python_src") or not r.get("id"):
            continue
        if r["id"] in held or r["id"] in dq:
            stats["excluded: held out / disqualified"] += 1
            continue
        if any(len(words(r["description"]) & h) / max(1, len(words(r["description"]) | h)) > 0.40 for h in held_w):
            stats["excluded: near-variant of a held-out question"] += 1
            continue
        order = qcheck.arg_order(r)
        if any(a in RESERVED for a in order):
            stats["excluded: reserved argument name"] += 1
            continue
        try:
            pr = prompt_for(r, order)
        except Exception:
            pr = None
        if not pr:
            stats["excluded: >3 args or generator error"] += 1
            continue
        out.append({"id": r["id"], "source": src, "weight": SOURCE_WEIGHT[src], "prompt": pr[0],
                    "fname": pr[1], "argnames": order, "python_src": r["python_src"],
                    "difficulty": r.get("difficulty")})
        stats[f"kept: {src}"] += 1
    with (ROOT / "corpus/rl_pool.jsonl").open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(dict(stats), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

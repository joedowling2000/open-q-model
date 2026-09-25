#!/usr/bin/env python3
"""Amendment 8 (i): new problems from Qwen3.5-27B, weighted to the weak spots.

A problem is a description, a Python reference and an input generator; no q.
It feeds three later stages: the student attempts it first, the teacher only if
the student fails, and every kept problem joins the RL prompt pool whether or
not anyone solved it.

What is measured to be weak gets more weight: hard problems (held-out hard 4.6%
after round 1), strings and type conversion (thinnest topics in the Q study
bank), and dict-shaped outputs (where the teacher failed most). A random setting
and output shape per request keep the model from writing the same five problems.

Kept only if (all in a time-limited child process for the model-written code):
  - the reference and generator run on 60 inputs and return JSON-serialisable
    values; the generator draws discriminating inputs (control 2, <=90% share)
  - no argument name is a q reserved word (it could never be called from q)
  - the description is not close to a Q-HumanEval prompt (control 1), a
    held-out question (amendment 7), or any problem we already have

    python scripts/synth/gen_problems.py --url http://127.0.0.1:8101/v1 \\
        --model qwen3.5-27b --target 3000 --out corpus/generated_problems.jsonl
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import multiprocessing as mp
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
from pipeline import PROBLEM_PROMPT, chat, fenced, humaneval_texts  # noqa: E402
import qcheck  # noqa: E402

RESERVED = set(json.loads((ROOT / "scripts/synth/q_reserved.json").read_text()))
TOPICS = {  # weight
    "string manipulation": 2.0, "type conversion and parsing": 2.0,
    "dictionaries and keyed lookup": 1.5, "grouping and aggregation": 1.2,
    "time series windows": 1.2, "table joins": 1.2, "running totals and deltas": 1.0,
    "sorting and ranking": 1.0, "set operations": 1.0, "filtering with predicates": 1.0,
    "matrix and nested list handling": 1.0, "vector arithmetic": 0.8,
}
LEVELS = {"hard": 0.40, "medium": 0.45, "easy": 0.15}
SHAPES = {
    "a dict mapping string keys to values": 0.35,
    "a list (of numbers, strings, or small lists)": 0.35,
    "a single number or boolean": 0.15,
    "a single string": 0.15,
}
SETTINGS = ["trade and order records", "sensor telemetry", "shipping and logistics",
            "retail inventory", "log files and event streams", "sports results",
            "bank transactions", "school timetables", "weather readings",
            "network packets", "library loans", "energy metering", "text documents",
            "product catalogues", "survey responses", "flight schedules"]

EXTRA = """
Setting: {setting}. solve() should return {shape}.
Argument names must be ordinary descriptive words that are not q keywords or
built-ins (avoid names like log, count, sum, max, min, first, last, key, value,
type, string, where, group, x, y, z)."""


def pick(rng: random.Random, weights: dict) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def words(text: str) -> frozenset:
    return frozenset(re.findall(r"[a-z]{4,}", text.lower()))


def too_close(w: frozenset, pool: list[frozenset], threshold: float) -> bool:
    """Overlap relative to the smaller text: strict, used for the benchmark."""
    return any(o and len(w & o) / min(len(w), len(o)) > threshold for o in pool)


def jaccard_close(w: frozenset, pool: list[frozenset], threshold: float) -> bool:
    """Jaccard similarity. The min-normalised measure flags any short description
    that shares a handful of common words ("total", "weight") with a long one."""
    return any(o and len(w & o) / len(w | o) > threshold for o in pool)


def _validate(src: str, args: list, q) -> None:
    try:
        _, expected = qcheck.gen_inputs(src, args, range(950000, 950060))
        q.put(("ok", qcheck.dominant_share(expected)))
    except BaseException as e:
        q.put(("fail", f"raised {type(e).__name__}"))


def validate(src: str, args: list, timeout: float = 60) -> tuple[str, object]:
    ctx = mp.get_context("spawn")  # fork from a threaded process can deadlock
    q = ctx.Queue()
    p = ctx.Process(target=_validate, args=(src, args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return "fail", "timed out"
    return q.get() if not q.empty() else ("fail", "crashed")


def solve_args(py: str) -> list[str] | None:
    try:
        for node in ast.parse(py).body:
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                return [a.arg for a in node.args.args]
    except SyntaxError:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=int, default=3000, help="kept problems to reach")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--out", default="corpus/generated_problems.jsonl")
    args = ap.parse_args()

    out = ROOT / args.out
    kept_rows = [json.loads(l) for l in out.open()] if out.exists() else []
    kept = sum(r["ok"] for r in kept_rows)

    bench = [words(t) for t in humaneval_texts()]
    held_ids = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    bank = [json.loads(l) for l in (ROOT / "corpus/q_study_v5/questions.jsonl").open()]
    held = [words(r["description"]) for r in bank if r["id"] in held_ids]
    existing = [words(r["description"]) for r in bank if r["id"] not in held_ids]
    ms = ROOT / "corpus/ms_rebuilt.jsonl"
    if ms.exists():
        existing += [words(json.loads(l).get("description", "")) for l in ms.open()]
    existing += [words(r["description"]) for r in kept_rows if r["ok"]]
    seen_py = {hashlib.sha256(r["python_src"].encode()).hexdigest()
               for r in kept_rows if r["ok"]}
    print(f"{kept} kept so far; target {args.target}", flush=True)

    rng = random.Random(args.seed + len(kept_rows))  # resumable without repeating specs

    def one(i: int) -> dict:
        r = random.Random(rng.random())
        spec = {"topic": pick(r, TOPICS), "difficulty": pick(r, LEVELS),
                "shape": pick(r, SHAPES), "setting": r.choice(SETTINGS)}
        # replace, not format: the template contains a literal `return {...}`
        prompt = PROBLEM_PROMPT.replace("{topic}", spec["topic"]).replace(
            "{level}", spec["difficulty"]) + EXTRA.format(setting=spec["setting"],
                                                         shape=spec["shape"])
        try:
            reply = chat(prompt, base_url=args.url, model=args.model,
                         temperature=0.9, max_tokens=1800)
        except Exception as e:
            return {**spec, "ok": False, "reason": f"model error {type(e).__name__}"}
        desc, py, gen = (fenced(reply, "description"), fenced(reply, "python"),
                         fenced(reply, "generator"))
        if not (desc and py and gen):
            return {**spec, "ok": False, "reason": "missing block"}
        names = solve_args(py)
        if not names:
            return {**spec, "ok": False, "reason": "no solve()"}
        if any(n in RESERVED for n in names):
            return {**spec, "ok": False, "reason": "argument is a q reserved word"}
        src = py + "\n\n" + gen + "\n"
        status, info = validate(src, names)
        if status != "ok":
            return {**spec, "ok": False, "reason": info}
        if info > 0.9:
            return {**spec, "ok": False, "reason": f"not discriminating ({info:.2f})"}
        return {**spec, "ok": True, "description": desc, "argnames": names,
                "python_src": src, "dominant_share": info}

    i = len(kept_rows)
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        while kept < args.target:
            batch = list(pool.map(one, range(i, i + args.workers * 4)))
            i += len(batch)
            for res in batch:
                if res["ok"]:
                    w = words(res["description"])
                    h = hashlib.sha256(res["python_src"].encode()).hexdigest()
                    if too_close(w, bench, 0.6):
                        res = {**res, "ok": False, "reason": "close to Q-HumanEval"}
                    elif jaccard_close(w, held, 0.35):
                        res = {**res, "ok": False, "reason": "close to a held-out question"}
                    elif h in seen_py or jaccard_close(w, existing, 0.6):
                        res = {**res, "ok": False, "reason": "near-duplicate"}
                    else:
                        existing.append(w)
                        seen_py.add(h)
                        kept += 1
                        res["id"] = f"gen__{h[:12]}"
                        res["question_type"] = "generated"
                        res["provenance"] = f"problem, reference and generator: {args.model}"
                fh.write(json.dumps(res) + "\n")
                fh.flush()
            print(f"{i} attempted, {kept} kept", flush=True)
    print("GEN_PROBLEMS_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

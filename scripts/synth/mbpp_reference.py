#!/usr/bin/env python3
"""Amendment 10 (ii): MBPP problem statements as disclosed seeds.

MBPP (CC-BY-4.0, ~970 short function tasks) is the closest public set to
HumanEval in style. Its statements are seeds, exactly like Morgan Stanley's:
Qwen3.5-27B writes a fresh reference and input generator from the statement and
signature. MBPP's three assert tests only check that reference. MBPP's own code
is never used, shown or trained on.

Contamination is handled before any problem is used. MBPP and HumanEval share
some trivial tasks under different wording ("greatest common divisor", "is the
number even"). Word overlap cannot find those: nearly every MBPP statement
begins "Write a function to find ... the given ...", so unrelated tasks score
0.2-0.3 Jaccard. Instead, each MBPP problem's three nearest Q-HumanEval
problems (by content words, boilerplate removed) go to Qwen3.5-27B as a judge:
"would a correct solution to one also solve the other?". Any "yes", or any
reply that is not a clear "no", drops the MBPP problem. So does an exact
function-name match.

    python scripts/synth/mbpp_reference.py --url http://127.0.0.1:8101/v1 \\
        --model qwen3.5-27b --out corpus/mbpp_rebuilt.jsonl
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import multiprocessing as mp
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
from pipeline import chat, fenced  # noqa: E402
import qcheck  # noqa: E402

RESERVED = set(json.loads((ROOT / "scripts/synth/q_reserved.json").read_text()))
PROMPT = """Here is a programming problem.

{text}

The function is called with these positional arguments: ({args}).

Write a Python reference solution and an input generator for it.

Reply with exactly two fenced blocks and nothing else:

```python
def solve({args}):
    <reference implementation>
```
```generator
def gen_input(rng):
    # return a dict of keyword arguments for solve(), satisfying the problem's
    # implied constraints, using rng (a random.Random). Vary the inputs so the
    # answers vary. Keep them small (lists of at most 20 elements).
    return {{...}}
```

Rules: deterministic, JSON-serialisable return values (lists, not tuples or
sets), standard library only, well under a second per call."""


BOILER = {"write", "function", "python", "find", "given", "check", "whether", "list",
          "array", "string", "number", "numbers", "elements", "element", "return"}
JUDGE = """Task A:
{a}

Task B:
{b}

Would a correct implementation of task A also be a correct implementation of task B
(same inputs, same outputs), or are they essentially the same programming task?
Answer with exactly one word: yes or no."""


def words(t: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]{4,}", t.lower()) if w not in BOILER)


def overview(prompt: str) -> str:
    text = re.sub(r".*?@overview", "", prompt, flags=re.S).split("@param")[0]
    return " ".join(l.strip("/ ") for l in text.splitlines()).strip()


def signature(code: str) -> tuple[str, list[str]] | None:
    try:
        for node in ast.parse(code).body:
            if isinstance(node, ast.FunctionDef):
                return node.name, [a.arg for a in node.args.args]
    except SyntaxError:
        return None
    return None


def parse_tests(tests: list[str], fname: str) -> list[tuple[list, object]] | None:
    out = []
    for t in tests:
        try:
            node = ast.parse(t.strip()).body[0]
            cmp = node.test
            call, expected = cmp.left, cmp.comparators[0]
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", "") == fname):
                return None
            out.append(([ast.literal_eval(a) for a in call.args], ast.literal_eval(expected)))
        except Exception:
            return None
    return out or None


def _validate(src: str, tests: list, args: list, q) -> None:
    try:
        g: dict = {}
        exec(src, g)
        for pos, exp in tests:
            got = g["solve"](*json.loads(json.dumps(pos)))
            if json.loads(json.dumps(got)) != json.loads(json.dumps(exp)):
                q.put(("fail", "disagrees with MBPP's tests"))
                return
        _, expected = qcheck.gen_inputs(src, args, range(1400000, 1400060))
        q.put(("ok", qcheck.dominant_share(expected)))
    except BaseException as e:
        q.put(("fail", f"raised {type(e).__name__}"))


def validate(src: str, tests: list, args: list, timeout: float = 60):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_validate, args=(src, tests, args, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill()
        p.join()
        return "fail", "timed out"
    return q.get() if not q.empty() else ("fail", "crashed")


def candidates() -> tuple[list[dict], dict]:
    rows = []
    for p in sorted(glob.glob(str(ROOT / "corpus/hf/mbpp/*.parquet"))):
        rows += pq.read_table(p).to_pylist()
    he = [json.loads(l) for l in (ROOT / "q-evaluation-harness/datasets/q_humaneval.jsonl").open()]
    he_ov = [(h["entry_point"].lower(), overview(h["prompt"])) for h in he]
    he_words = [words(o) for _, o in he_ov]
    he_names = {n for n, _ in he_ov}
    kept, dropped = [], {"same function name as a Q-HumanEval entry point": 0,
                         "no signature or tests": 0, "argument is a q reserved word": 0}
    for r in rows:
        sig = signature(r["code"])
        tests = parse_tests(r["test_list"], sig[0]) if sig else None
        if not sig or not tests:
            dropped["no signature or tests"] += 1
            continue
        if sig[0].lower() in he_names:
            dropped["same function name as a Q-HumanEval entry point"] += 1
            continue
        if any(a in RESERVED for a in sig[1]):
            dropped["argument is a q reserved word"] += 1
            continue
        w = words(r["text"])
        near = sorted(range(len(he_ov)), key=lambda i: -(len(w & he_words[i]) / max(1, len(w | he_words[i]))))[:3]
        kept.append({"task_id": r["task_id"], "text": r["text"], "fname": sig[0],
                     "args": sig[1], "tests": tests, "near": [he_ov[i][1] for i in near]})
    return kept, dropped


def same_task(c: dict, url: str, model: str) -> bool:
    for other in c["near"]:
        reply = chat(JUDGE.format(a=c["text"], b=other), base_url=url, model=model,
                     temperature=0.0, max_tokens=5)
        if reply.strip().lower().rstrip(".") != "no":
            return True
    return False


def build(c: dict, url: str, model: str, attempts: int) -> dict:
    rid = f"mbpp__{c['task_id']}"
    try:
        if same_task(c, url, model):
            return {"id": rid, "ok": False, "reason": "judged the same task as a Q-HumanEval problem"}
    except Exception as e:
        return {"id": rid, "ok": False, "reason": f"judge error {type(e).__name__}"}
    last = "no reply"
    for attempt in range(attempts):
        try:
            reply = chat(PROMPT.format(text=c["text"], args=", ".join(c["args"])),
                         base_url=url, model=model, temperature=0.3 + 0.2 * attempt,
                         max_tokens=1200)
        except Exception as e:
            return {"id": rid, "ok": False, "reason": f"model error {type(e).__name__}"}
        py, gen = fenced(reply, "python"), fenced(reply, "generator")
        if not py or not gen:
            last = "missing block"
            continue
        src = py + "\n\n" + gen + "\n"
        status, info = validate(src, c["tests"], c["args"])
        if status != "ok":
            last = info
            continue
        if info > 0.9:
            last = f"not discriminating ({info:.2f})"
            continue
        return {"id": rid, "ok": True, "description": c["text"], "argnames": c["args"],
                "function_name": c["fname"], "python_src": src, "dominant_share": info,
                "question_type": "mbpp-seed", "difficulty": None, "topic": "mbpp",
                "provenance": f"statement: MBPP task {c['task_id']} (CC-BY-4.0, seed, disclosed); "
                              f"reference and generator: {model}"}
    return {"id": rid, "ok": False, "reason": last}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="corpus/mbpp_rebuilt.jsonl")
    ap.add_argument("--dry-run", action="store_true", help="report the filter only")
    args = ap.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)   # MBPP code has bare regex strings
    kept, dropped = candidates()
    print(f"{len(kept)} MBPP problems eligible; dropped {dropped}", flush=True)
    if args.dry_run:
        return 0
    out = ROOT / args.out
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [c for c in kept if f"mbpp__{c['task_id']}" not in done]
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for res in pool.map(lambda c: build(c, args.url, args.model, args.attempts), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(res["id"], "ok" if res["ok"] else res["reason"], flush=True)
    print("MBPP_REFERENCE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

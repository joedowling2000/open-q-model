#!/usr/bin/env python3
"""Amendment 3's vector rewrite, on the shared judge (qcheck).

revectorise.py predates qcheck and inherits the old verifier: inputs inlined
into the lambda (q's `limit`), 7-digit floats, no dict inputs, no memory cap.
This keeps its prompt and its rules and swaps in qcheck:

  * the existing loop solution is re-verified first; if it fails on the fresh
    inputs it is reported as suspect, not rewritten;
  * the rewriter gets several attempts up a temperature ladder;
  * a rewrite is kept only if it has no `while`/`do` and agrees with the
    Python reference on 40 fresh inputs, more than acceptance used;
  * the loop version is kept as `q_solution_imperative`, so the
    imperative -> vector pair becomes a training task (assemble_sft's vectorise).

Input records need id, description, python_src, argnames or signature, and
q_solution. Output: {id: {"q": vector code, "attempt": n}} or a reason.

    python scripts/synth/vector_rewrite.py --url http://127.0.0.1:8104/v1 \\
        --model qwen3.5-27b-sft-v5 --in corpus/distilled.jsonl \\
        --out corpus/vector_rewrites.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
sys.path.insert(0, str(ROOT / "scripts/train"))
import qcheck  # noqa: E402
from bank_to_records import reference_only  # noqa: E402
from pipeline import chat, fenced  # noqa: E402
from revectorise import PROMPT  # noqa: E402

LOOPY = re.compile(r"\b(while|do)\[")
SEEDS = range(1000000, 1000040)
TEMPS = (0.2, 0.5, 0.7, 0.9, 1.1)


def rewrite(rec: dict, url: str, model: str) -> dict:
    # One bad record must not end the pass: the first run died at 109 of 673
    # on a generator that raises for some seeds (randint on an empty range).
    try:
        return _rewrite(rec, url, model)
    except Exception as e:
        return {"id": rec["id"], "ok": False, "reason": f"reference error {type(e).__name__}"}


def _rewrite(rec: dict, url: str, model: str) -> dict:
    order = qcheck.arg_order(rec)
    base = qcheck.check(rec["q_solution"], rec["python_src"], order, SEEDS)
    if not base["ok"]:
        return {"id": rec["id"], "ok": False, "reason": "ORIGINAL FAILED on fresh inputs"}
    prompt = PROMPT.format(description=rec["description"],
                           python=reference_only(rec["python_src"]),
                           imperative=rec["q_solution"])
    for attempt, temp in enumerate(TEMPS, 1):
        try:
            cand = fenced(chat(prompt, base_url=url, model=model, temperature=temp,
                               max_tokens=1200), "q")
        except Exception as e:
            return {"id": rec["id"], "ok": False, "reason": f"model error {type(e).__name__}"}
        if not cand or not cand.lstrip().startswith("solve") or LOOPY.search(cand):
            continue
        if qcheck.check(cand, rec["python_src"], order, SEEDS)["ok"]:
            return {"id": rec["id"], "ok": True, "q": cand, "attempt": attempt}
    return {"id": rec["id"], "ok": False, "reason": "no attempt verified"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="src", nargs="+", required=True,
                    help="jsonl of records with q_solution (bank rows are flattened)")
    ap.add_argument("--out", default="corpus/vector_rewrites.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    recs = []
    for path in args.src:
        for line in (ROOT / path).open():
            r = json.loads(line)
            if r.get("ok") is False or r["id"] in held:
                continue
            if "q_solution" not in r and r.get("solutions"):   # Q study bank row
                r = {**r, "q_solution": r["solutions"][0]["code"]}
            if r.get("q_solution") and LOOPY.search(r["q_solution"]):
                recs.append(r)
    out = ROOT / args.out
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    print(f"{len(recs)} loop solutions, {len(done)} done, {len(todo)} to go", flush=True)
    kept = 0
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for res in pool.map(lambda r: rewrite(r, args.url, args.model), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            kept += res["ok"]
            print(res["id"], f"VECTOR (attempt {res['attempt']})" if res["ok"]
                  else res["reason"], flush=True)
    print(f"REWRITE_DONE kept {kept} of {len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

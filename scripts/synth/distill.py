#!/usr/bin/env python3
"""Step 2: q solutions from the teacher, kept only if they verify (amendments 5, 6).

The teacher (qqWen-72B-RL, served by llama.cpp) sees the description and the
Python reference and writes q. Each problem gets up to `--samples` attempts in
one request; every attempt is run against the reference on 60 fresh inputs by
qcheck, the same judge used everywhere else. Every distinct passing solution is
kept, and the record notes which of them loop, so the vector rewrite pass
(amendment 3) knows what to work on.

Problems whose reference is not discriminating (one answer in >90% of draws,
control 2) are skipped before the teacher is asked: such a problem cannot
separate right q from wrong, so anything it "verifies" is unearned.

Held-out questions (amendment 7) are refused outright, whatever the input file.

    python scripts/synth/distill.py --url http://127.0.0.1:8103/v1 \\
        --model qqwen-72b-rl --in corpus/q_study_v5/questions.jsonl \\
        --only-unsolved --out corpus/distilled.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
sys.path.insert(0, str(ROOT / "scripts/train"))
import qcheck  # noqa: E402
from bank_to_records import reference_only  # noqa: E402
from pipeline import Q_PROMPT  # noqa: E402

SEEDS = range(800000, 800060)
LOOPY = re.compile(r"\b(while|do)\[")


def ask(url: str, model: str, prompt: str, n: int, temperature: float,
        max_tokens: int) -> list[str]:
    body = json.dumps({"model": model, "n": n, "temperature": temperature,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return [c["message"]["content"] for c in json.load(r)["choices"]]


def extract(text: str) -> str | None:
    m = re.findall(r"```(?:q|k)?\s*\n(.*?)```", text, re.S)
    code = (m[-1] if m else text).strip()
    return code if code.startswith("solve") else None


def distill(rec: dict, args) -> dict:
    order = qcheck.arg_order(rec)
    try:
        _, expected = qcheck.gen_inputs(rec["python_src"], order, SEEDS)
    except Exception as e:
        return {"id": rec["id"], "ok": False, "reason": f"reference error {type(e).__name__}"}
    share = qcheck.dominant_share(expected)
    if share > 0.9:
        return {"id": rec["id"], "ok": False, "reason": f"not discriminating ({share:.2f})"}

    python = reference_only(rec["python_src"])
    prompt = Q_PROMPT.format(description=rec["description"], python=python,
                             args=";".join(order))
    try:
        replies = ask(args.url, args.model, prompt, args.samples, args.temperature,
                      args.max_tokens)
    except Exception as e:
        return {"id": rec["id"], "ok": False, "reason": f"teacher error {type(e).__name__}"}

    passing, failures = [], 0
    for code in dict.fromkeys(filter(None, map(extract, replies))):  # distinct, in order
        res = qcheck.check(code, rec["python_src"], order, SEEDS)
        if res["ok"]:
            passing.append({"q": code, "loops": bool(LOOPY.search(code))})
        else:
            failures += 1
    base = {"id": rec["id"], "attempts": len(replies), "failed": failures,
            "dominant_share": share, "teacher": args.model}
    if not passing:
        return {**base, "ok": False, "reason": "no attempt verified"}
    # Prefer a loop-free solution, then the shortest.
    passing.sort(key=lambda p: (p["loops"], len(p["q"])))
    return {**base, "ok": True, "description": rec["description"],
            "python_solution": python, "python_src": rec["python_src"],
            "argnames": order, "q_solution": passing[0]["q"],
            "q_alternatives": [p["q"] for p in passing[1:]],
            "uses_loops": passing[0]["loops"],
            "topic": rec.get("topic"), "difficulty": rec.get("difficulty"),
            "source": rec.get("question_type") or rec.get("source")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="src", nargs="+", required=True)
    ap.add_argument("--out", default="corpus/distilled.jsonl")
    ap.add_argument("--only-unsolved", action="store_true",
                    help="skip records that already carry a verified q solution")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    recs = []
    for path in args.src:
        for line in (ROOT / path).open():
            r = json.loads(line)
            if r.get("ok") is False:          # a failed ms_reference.py row
                continue
            if r["id"] in held:
                continue
            if args.only_unsolved and r.get("solutions"):
                continue
            recs.append(r)
    out = ROOT / args.out
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done][: args.limit]
    print(f"{len(recs)} eligible, {len(done)} done, {len(todo)} to go", flush=True)

    kept = 0
    with ThreadPoolExecutor(args.workers) as pool, out.open("a") as fh:
        for res in pool.map(lambda r: distill(r, args), todo):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            kept += res["ok"]
            print(res["id"], "KEPT" + (" (loops)" if res.get("uses_loops") else "")
                  if res["ok"] else res["reason"], flush=True)
    print(f"DISTILL_DONE kept {kept} of {len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

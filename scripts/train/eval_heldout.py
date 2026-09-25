#!/usr/bin/env python3
"""Score a served model on the amendment 7 held-out set.

For each of the 75 held-out questions, sample q from the description (the
desc2q prompt used in training), then run each sample against the Python
reference on 60 fresh generated inputs with qcheck. pass@1 is the unbiased
estimator over n samples, bootstrapped over questions.

The server is whatever eval_checkpoint.sh (or serve.sh) started; this script
only needs its OpenAI-compatible URL and model alias.

    python scripts/train/eval_heldout.py --url http://127.0.0.1:8102/v1 \\
        --model qwen3.5-27b-sft-v5 --n 10 --out results/heldout-qwen3.5-27b-sft-v5.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
sys.path.insert(0, str(ROOT / "scripts/train"))
import qcheck  # noqa: E402
from assemble_sft import desc2q  # noqa: E402

SEEDS = range(600000, 600060)


def complete(url: str, model: str, prompt: str, n: int, max_tokens: int) -> list[str]:
    body = json.dumps({"model": model, "n": n, "temperature": 0.8,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(url + "/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return [c["message"]["content"] for c in json.load(r)["choices"]]


def extract(text: str) -> str:
    m = re.findall(r"```(?:q|k)?\s*\n(.*?)```", text, re.S)
    return (m[-1] if m else text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="first N questions (testing)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    held = set(json.loads((ROOT / "corpus/heldout_v5.json").read_text())["ids"])
    recs = [json.loads(l) for l in (ROOT / "corpus/q_study_v5/questions.jsonl").open()]
    recs = [r for r in recs if r["id"] in held][: args.limit]

    def one(r):
        prompt = desc2q({"description": r["description"], "q_solution": ""})[0]
        samples = complete(args.url, args.model, prompt, args.n, args.max_tokens)
        results = [qcheck.check(extract(s), r["python_src"], qcheck.arg_order(r), SEEDS)
                   for s in samples]
        return {"id": r["id"], "difficulty": r["difficulty"], "topic": r["topic"],
                "n": len(samples), "correct": sum(x["ok"] for x in results),
                "samples": samples}

    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(one, recs))

    def pass1(rs):
        return sum(r["correct"] / r["n"] for r in rs) / len(rs)

    rng = random.Random(0)
    boots = sorted(pass1([rng.choice(rows) for _ in rows]) for _ in range(10000))
    # Incident 4: also report the clean subset (no close training variant).
    clean_ids = set(json.loads((ROOT / "corpus/heldout_v5_clean.json").read_text())["ids"])
    clean = [r for r in rows if r["id"] in clean_ids]
    summary = {"model": args.model, "n_questions": len(rows), "n_samples": args.n,
               "pass_at_1_clean": pass1(clean) if clean else None, "n_clean": len(clean),
               "pass_at_1": pass1(rows), "ci95": [boots[250], boots[9750]],
               "by_difficulty": {d: pass1([r for r in rows if r["difficulty"] == d])
                                 for d in ("medium", "hard")}}
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("HELDOUT_RESULT", json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

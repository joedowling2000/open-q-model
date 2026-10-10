#!/usr/bin/env python3
"""Score a served policy on the amendment 15 style held-out set.

200 benchmark-style problems never drawn in RL. Harness-format prompt, 5
samples, temperature 0.8, 512 tokens. A sample passes if it is fully correct
(reward 1.0) on 30 fresh inputs. pass@1 is averaged over problems, with a
bootstrap CI.

    python scripts/rl/eval_style.py --url http://127.0.0.1:8108/v1 --model policy --out results/style-X.json
"""
import argparse, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/rl")); sys.path.insert(0, str(ROOT / "scripts/synth")); sys.path.insert(0, str(ROOT / "scripts/train"))
from rollout import generate, extract, reward          # noqa: E402
from rl_pool import prompt_for                         # noqa: E402
import qcheck                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5)
a = ap.parse_args()
ids = set(json.loads((ROOT / "corpus/heldout_style.json").read_text())["ids"])
recs = [json.loads(l) for l in (ROOT / "corpus/hstyle_problems.jsonl").open()]
recs = [r for r in recs if r.get("id") in ids]
seeds = range(3_000_000, 3_000_030)

def one(r):
    order = qcheck.arg_order(r)
    prompt, fname = prompt_for(r, order)
    rec = {"fname": fname, "python_src": r["python_src"], "argnames": order}
    outs = generate(a.url, a.model, prompt, a.n)
    ok = [reward(extract(o), rec, seeds) == 1.0 for o in outs]
    return {"id": r["id"], "n": len(ok), "correct": sum(ok)}

with ThreadPoolExecutor(8) as p:
    rows = list(p.map(one, recs))
p1 = lambda rs: sum(r["correct"] / r["n"] for r in rs) / len(rs)
rng = random.Random(0)
b = sorted(p1([rng.choice(rows) for _ in rows]) for _ in range(10000))
s = {"model": a.model, "n_problems": len(rows), "n_samples": a.n, "pass_at_1": p1(rows), "ci95": [b[250], b[9750]]}
Path(a.out).write_text(json.dumps({"summary": s, "rows": rows}, indent=1))
print("STYLE_RESULT", json.dumps(s), flush=True)

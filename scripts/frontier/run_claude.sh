#!/bin/bash
# Full Q-HumanEval run for one Claude model via the Batch API, then grading and
# a comparison with our final model. Usage: run_claude.sh MODEL [N]
set -uo pipefail
cd /home/joedowling/Projects/qeval
set -a; . ~/.qeval-keys.env; set +a
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
M=$1; N=${2:-30}; F=results/frontier/$M
$QPY scripts/frontier/claude_gen.py --model $M --n $N --mode batch || { echo "GENERATION_FAILED"; exit 1; }
( cd q-evaluation-harness && taskset -c 0-4,10-14 $QPY -m src.cli execute "../$F.jsonl" q-humaneval \
    -o "../$F-scored.json" ) > "$F-exec.log" 2>&1 || { echo "GRADING_FAILED"; exit 1; }
python3 - "$F" <<'PY'
import json, random, sys
f = sys.argv[1]
d = json.load(open(f + "-scored.json"))
per = lambda x: {r["task_id"]: r["num_correct"] / r["num_samples"] for r in x["per_problem_results"]}
a, b = per(d), per(json.load(open("results/qwen3.5-27b-final-scored.json")))
k = sorted(a); rng = random.Random(0); m = lambda v: sum(v) / len(v)
ci = sorted(m([a[rng.choice(k)] for _ in k]) for _ in range(10000))
dd = sorted(m([b[t] - a[t] for t in (rng.choice(k) for _ in k)]) for _ in range(10000))
print(f"FRONTIER_RESULT {f}: pass@1 {100*d['pass_at_1']:.2f}% [{100*ci[250]:.1f}, {100*ci[9750]:.1f}], "
      f"pass@10 {100*d['pass_at_10']:.1f}%, errored {d['errored_solutions']}; "
      f"our final minus this: {100*m([b[t]-a[t] for t in k]):+.2f} [{100*dd[250]:+.1f}, {100*dd[9750]:+.1f}]")
PY

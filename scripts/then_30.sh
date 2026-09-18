#!/bin/bash
# Wait for the 10-sample pass, validate it, then reach 30 samples per problem by
# generating 20 more and merging — not by regenerating all 30. At ~7.5 h per
# model per 10 samples, reusing the first ten saves about a day.
#
# Waits purely on the ALL_EVALS_DONE marker: an earlier version also watched
# whether the systemd unit was running, and aborted the moment the 10-sample
# pass was restarted for benchmarking.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ROOT=/home/joedowling/Projects/qeval
MODELS="qwen3.5-27b gemma-4-31b-it qqwen-32b-rl"
PY=/home/joedowling/venvs/qeval/bin/python

echo "waiting for the 10-sample pass..."
until grep -aq "ALL_EVALS_DONE" logs/run_all.log; do sleep 120; done

echo "=== validating the 10-sample pass ==="
$PY scripts/validate.py 10 $MODELS || {
  echo "NOT_STARTING_30 — validation failed; fix the 10-sample run first"; exit 1; }

echo "=== generating 20 more samples per problem ==="
N_SAMPLES=20 bash scripts/run_all_n.sh || echo "TOPUP_PASS_HAD_FAILURES"

echo "=== merging to 30 samples and grading ==="
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1
for name in $MODELS; do
  [ -f "results/$name.jsonl" ] && [ -f "results/$name-n20.jsonl" ] || {
    echo "SKIP_MERGE $name (missing inputs)"; continue; }
  $PY scripts/merge_samples.py "results/$name.jsonl" "results/$name-n20.jsonl" \
      "results/$name-n30.jsonl" || { echo "MERGE_FAILED $name"; continue; }
  ( cd q-evaluation-harness && $PY -m src.cli execute \
      "$ROOT/results/$name-n30.jsonl" q-humaneval \
      -o "$ROOT/results/$name-n30-scored.json" ) > "logs/exec-$name-n30.log" 2>&1 || {
    echo "EXECUTE_FAILED $name-n30"; continue; }
  $PY - "results/$name-n30-scored.json" "$name" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metrics", d)
print("RESULT30", sys.argv[2], json.dumps({k: v for k, v in m.items()
      if str(k).lower().startswith("pass")}), flush=True)
PY
done

$PY scripts/validate.py 30 $MODELS
echo THIRTY_SAMPLE_PASS_DONE

#!/bin/bash
# Finish the third model on its own, after its conversion failed the first time.
#
# qqWen ships config.json with "use_cache": null, which transformers 5.x rejects
# (strict dataclass validation: expected bool, got NoneType). Fixed locally by
# replacing the symlinked config.json in the snapshot with a corrected copy.
#
# Then hands back to then_30.sh, which validates all three 10-sample runs and
# starts the top-up to 30.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ROOT=/home/joedowling/Projects/qeval
name=qqwen-32b-rl
N_SAMPLES=10
PORT=8099
SLOTS=$N_SAMPLES
export QEVAL_MAX_CONCURRENT=1
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1
export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
PY=/home/joedowling/venvs/qeval/bin/python

echo "=== $name: waiting for conversion ==="
until [ -f "gguf/$name-q8_0.gguf" ] && ! pgrep -f "convert_hf_to_gguf" > /dev/null; do sleep 60; done
ls -la "gguf/$name-q8_0.gguf"

echo "=== $name: serving ==="
setsid bash scripts/serve.sh "$name" "$PORT" "$SLOTS" > "logs/serve-$name.log" 2>&1 &
tries=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10; tries=$((tries+1))
  [ $tries -gt 90 ] && { echo "SERVE_FAILED $name"; exit 1; }
done

echo "=== $name: generating ($N_SAMPLES samples) ==="
rm -f "results/$name.jsonl"          # the harness appends; never inherit a stale run
( cd q-evaluation-harness && $PY -m src.cli generate q-humaneval "openai/$name" \
    --backend litellm -n "$N_SAMPLES" -o "$ROOT/results/$name.jsonl" ) \
    > "logs/gen-$name.log" 2>&1
gen_rc=$?
pkill -f "$name-q8_0.gguf"
sleep 5
[ $gen_rc -ne 0 ] && { echo "GENERATE_FAILED $name"; exit 1; }

echo "=== $name: grading ==="
( cd q-evaluation-harness && $PY -m src.cli execute "$ROOT/results/$name.jsonl" \
    q-humaneval -o "$ROOT/results/$name-scored.json" ) > "logs/exec-$name.log" 2>&1 || {
  echo "EXECUTE_FAILED $name"; exit 1; }

$PY - "results/$name-scored.json" "$name" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metrics", d)
print("RESULT", sys.argv[2], json.dumps({k: v for k, v in m.items()
      if str(k).lower().startswith("pass")}), flush=True)
PY

echo "QQWEN_DONE — handing back to the 30-sample chain"
exec bash scripts/then_30.sh

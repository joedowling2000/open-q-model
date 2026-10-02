#!/bin/bash
# Finish round 4's scoring, then run the amendment 11 screen. One model on the
# GPU at a time.
#
# Round 4's generation completed (4920 samples), but its scoring crashed: I
# edited eval_checkpoint.sh while it was executing, and bash reads scripts
# incrementally. Rule from now on: never edit a script a running job is
# executing; write a new file instead.
set -uo pipefail
cd /home/joedowling/Projects/qeval
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1
NAME=qwen3.5-27b-r4
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "held-out for $NAME"
/home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf \
  --host 127.0.0.1 --port 8102 --alias $NAME --reasoning off -ngl 999 -t 6 -tb 6 \
  -c $((2048 * 30)) -np 30 --cont-batching --no-webui > logs/serve-$NAME-heldout.log 2>&1 &
S=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8102/health > /dev/null && break; sleep 10; done
taskset -c 0-4,10-14 $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8102/v1 --model $NAME \
  --n 10 --out results/heldout-$NAME.json > logs/heldout-$NAME.log 2>&1 || log "held-out failed"
grep HELDOUT_RESULT logs/heldout-$NAME.log
kill $S; wait $S 2>/dev/null; sleep 10

log "grading $NAME (generations already complete)"
( cd q-evaluation-harness && $QPY -m src.cli execute "../results/$NAME.jsonl" q-humaneval \
    -o "../results/$NAME-scored.json" ) > logs/exec-$NAME.log 2>&1 || log "grading failed"
python3 -c "import json;d=json.load(open('results/$NAME-scored.json'));print('GATE_RESULT $NAME', {k:round(v,4) for k,v in d.items() if k.startswith('pass_at')})"
log "R4_SCORED"

bash scripts/train/screen_coders.sh

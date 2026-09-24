#!/bin/bash
# Step 2 (amendments 5 and 6), run as one chain so the GPU never idles overnight.
#
#   1. Qwen3.5-27B (base, open general model) rebuilds Python references and
#      input generators for Morgan Stanley's problems from the descriptions
#      alone; their test cases only check the new reference.
#   2. qqWen-72B-RL (the teacher) writes q for those problems and for the Q
#      study bank questions still awaiting a solution. Every attempt is verified
#      on 60 fresh inputs; held-out questions are refused.
#
# Waits for any scoring run to finish first: one model on the GPU at a time.
set -uo pipefail
cd /home/joedowling/Projects/qeval
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
log() { echo "[$(date +%H:%M:%S)] $*"; }

serve() {  # name port slots
  bash scripts/serve.sh "$1" "$2" "$3" > "logs/serve-$1.log" 2>&1 &
  SERVER=$!
  for i in $(seq 180); do curl -sf "http://127.0.0.1:$2/health" > /dev/null && return 0; sleep 10; done
  log "server $1 never came up"; return 1
}

while systemctl --user list-units 'run-qevalv5*' --no-legend | grep -q running; do sleep 120; done
log "GPU free"

log "stage A: Morgan Stanley references with qwen3.5-27b"
serve qwen3.5-27b 8101 8 || exit 1
$QPY scripts/synth/ms_reference.py --url http://127.0.0.1:8101/v1 --model qwen3.5-27b \
  --workers 8 --out corpus/ms_rebuilt.jsonl > logs/ms-reference.log 2>&1
log "stage A exited rc=$? ($(grep -c '"ok": true' corpus/ms_rebuilt.jsonl) references kept)"
kill $SERVER; wait $SERVER 2>/dev/null; sleep 10

log "stage B: teacher distillation with qqwen-72b-rl"
serve qqwen-72b-rl 8103 16 || exit 1
$QPY scripts/synth/distill.py --url http://127.0.0.1:8103/v1 --model qqwen-72b-rl \
  --in corpus/q_study_v5/questions.jsonl corpus/ms_rebuilt.jsonl --only-unsolved \
  --samples 8 --workers 2 --out corpus/distilled.jsonl > logs/distill.log 2>&1
log "stage B exited rc=$? ($(grep -c '"ok": true' corpus/distilled.jsonl) problems kept)"
kill $SERVER; wait $SERVER 2>/dev/null
log "STEP2_DONE"

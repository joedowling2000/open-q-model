#!/bin/bash
# Keep the GPU busy overnight: when the 30-sample benchmark finishes, move
# straight on to synthetic data generation instead of idling until morning.
#
# Deliberately serial — two GPU jobs at once slow each other and double the heat,
# and the benchmark's numbers are the thing we must not disturb.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ROOT=/home/joedowling/Projects/qeval
PORT=8100
MODEL=qwen3.5-27b
PY=/home/joedowling/venvs/jlens/bin/python

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the benchmark to finish"
while true; do
  grep -aq "THIRTY_SAMPLE_PASS_DONE" logs/run_qqwen.log && { log "benchmark finished"; break; }
  # Don't wait forever on a chain that has died: if the unit is gone and no
  # generation is running, the night is better spent generating data.
  if ! systemctl --user is-active --quiet run-qeval-qqwen-20260917-150643.service; then
    log "benchmark unit is no longer active; proceeding"
    break
  fi
  sleep 120
done

# Whatever happened above, make sure no server is holding the GPU.
pkill -f "q8_0.gguf" 2>/dev/null
sleep 10

log "serving $MODEL for generation"
setsid bash scripts/serve.sh "$MODEL" "$PORT" 10 > logs/serve-synth.log 2>&1 &
tries=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10; tries=$((tries+1))
  [ $tries -gt 120 ] && { log "server never came up; aborting"; exit 1; }
done
log "server up"

# Batches rather than one long run: each batch appends to corpus/synthetic.jsonl,
# so an interruption keeps everything generated so far.
export SYNTH_BASE_URL="http://127.0.0.1:$PORT/v1" SYNTH_MODEL="$MODEL"
for batch in $(seq 1 12); do
  log "batch $batch"
  timeout 5400 $PY scripts/synth/pipeline.py \
    --n-problems 25 --samples 4 --cases 30 \
    --out corpus/synthetic.jsonl >> logs/synth-batches.log 2>&1
  kept=$(wc -l < corpus/synthetic.jsonl 2>/dev/null || echo 0)
  log "batch $batch done; corpus now $kept verified problems"
done

pkill -f "q8_0.gguf" 2>/dev/null
log "OVERNIGHT_DONE"

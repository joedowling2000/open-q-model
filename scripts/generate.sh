#!/bin/bash
# Generate the q dataset with two models resident at once.
#
#   problem author : Qwen3.5-27B  on :8100  (general reasoning, writes the
#                    description, Python reference and input generator)
#   q author       : qqWen-32B-RL on :8101  (best measured q model, 38.4%)
#
# 29 GB + 35 GB of weights fits in 121 GB, so both stay loaded: swapping models
# between every problem would cost more than sharing bandwidth does.
#
# See GENERATION.md for what the prompts specify and why.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ROOT=/home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
BATCHES="${BATCHES:-24}"
N_PROBLEMS="${N_PROBLEMS:-25}"
SAMPLES="${SAMPLES:-6}"

log() { echo "[$(date +%H:%M:%S)] $*"; }

start_server() {                       # name port slots logfile
  setsid bash scripts/serve.sh "$1" "$2" "$3" > "$4" 2>&1 &
  local tries=0
  until curl -sf "http://127.0.0.1:$2/health" > /dev/null; do
    sleep 10; tries=$((tries+1))
    [ $tries -gt 120 ] && { log "server $1 never came up"; return 1; }
  done
  log "$1 serving on :$2"
}

start_server qwen3.5-27b  8100 6 logs/serve-problems.log || exit 1
start_server qqwen-32b-rl 8101 8 logs/serve-qauthor.log || exit 1

export SYNTH_BASE_URL="http://127.0.0.1:8100/v1" SYNTH_MODEL=qwen3.5-27b
export SYNTH_Q_URL="http://127.0.0.1:8101/v1"    SYNTH_Q_MODEL=qqwen-32b-rl

for batch in $(seq 1 "$BATCHES"); do
  log "batch $batch of $BATCHES"
  timeout 7200 $PY scripts/synth/pipeline.py \
    --n-problems "$N_PROBLEMS" --samples "$SAMPLES" --cases 30 --parallel 4 \
    --out corpus/qdataset.jsonl >> logs/generate.log 2>&1
  kept=$(wc -l < corpus/qdataset.jsonl 2>/dev/null || echo 0)
  log "batch $batch done; dataset now $kept verified problems"
done

pkill -f "q8_0.gguf"
log "GENERATION_DONE"

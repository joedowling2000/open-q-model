#!/bin/bash
# Amendment 8 (i): ~3000 new problems from the open base model Qwen3.5-27B.
# Resumable: gen_problems.py counts what is already kept and continues.
set -uo pipefail
cd /home/joedowling/Projects/qeval
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
log() { echo "[$(date +%H:%M:%S)] $*"; }

bash scripts/serve.sh qwen3.5-27b 8101 8 > logs/serve-qwen3.5-27b-gen.log 2>&1 &
SERVER=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8101/health > /dev/null && break; sleep 10; done
log "generating problems"
$QPY scripts/synth/gen_problems.py --url http://127.0.0.1:8101/v1 --model qwen3.5-27b \
  --target "${TARGET:-3000}" --workers 8 --out corpus/generated_problems.jsonl \
  >> logs/gen-problems.log 2>&1
rc=$?
kill $SERVER; wait $SERVER 2>/dev/null
log "GEN_EXIT rc=$rc — $(tail -1 logs/gen-problems.log)"

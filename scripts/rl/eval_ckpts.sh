#!/bin/bash
# Score clean held-out for given RL checkpoints, to choose the final one
# (amendment 12: selection on held-out only). Usage: eval_ckpts.sh 11 12
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
log() { echo "[$(date +%H:%M:%S)] $*"; }
for R in "$@"; do
  A=$(printf "out/rl/r%03d" $R)
  [ -f $A/adapter.gguf ] || $PY scripts/train/convert_lora.py $A --base out/merged/qwen3.5-27b-r4 \
      --outfile $A/adapter.gguf --outtype f16 > logs/rl-convert.log 2>&1
  /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf \
    --lora $A/adapter.gguf --host 127.0.0.1 --port 8108 --alias policy --reasoning off -ngl 999 -t 6 -tb 6 \
    -c $((2048 * 32)) -np 32 --cont-batching --no-webui > logs/rl-eval-serve.log 2>&1 &
  S=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:8108/health > /dev/null && break; sleep 10; done
  taskset -c 0-4,10-14 $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8108/v1 --model policy --n 10 \
    --out results/heldout-rl-r$(printf %03d $R).json > logs/rl-heldout-r$R.log 2>&1 || log "r$R held-out failed"
  kill $S; wait $S 2>/dev/null; sleep 10
  log "r$R $(grep HELDOUT_RESULT logs/rl-heldout-r$R.log)"
done
log "CKPT_EVAL_DONE"

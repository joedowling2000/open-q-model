#!/bin/bash
# RL on round 4 (amendments 4, 9, 10, 12). Each round:
#   serve policy (round 4 Q8 GGUF + RL adapter) -> rollouts + rewards -> stop
#   server -> GRPO update of the adapter (bf16 merged round 4 + LoRA).
# After the pilot rounds, the amendment 9 check: is the reward rising, and is
# clean held-out not below round 4? Env: START, END, PROMPTS, GROUP, PILOT_END.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
START=${START:-1}; END=${END:-5}; PROMPTS=${PROMPTS:-128}; GROUP=${GROUP:-8}; PILOT_END=${PILOT_END:-5}
E="taskset -c 0-4,10-14"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
adapter() { printf "out/rl/r%03d" "$1"; }

serve() {  # adapter-dir-or-none port
  local flag=()
  if [ "$1" != "none" ]; then
    $PY scripts/train/convert_lora.py "$1" --base out/merged/qwen3.5-27b-r4 --outfile "$1/adapter.gguf" \
      --outtype f16 > logs/rl-convert.log 2>&1 || die "adapter conversion failed for $1"
    flag=(--lora "$1/adapter.gguf")
  fi
  $LC/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf "${flag[@]}" --host 127.0.0.1 --port $2 \
    --alias policy --reasoning off -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 --cont-batching --no-webui \
    > logs/rl-serve.log 2>&1 &
  SERVER=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:$2/health > /dev/null && return 0; sleep 10; done
  die "policy server never came up"
}

for R in $(seq $START $END); do
  PREV=$(adapter $((R - 1))); [ -d "$PREV" ] || PREV=none     # first round of a fresh run
  log "round $R: rollouts ($PROMPTS prompts x $GROUP) with policy $PREV"
  serve "$PREV" 8107
  $E $QPY scripts/rl/rollout.py --url http://127.0.0.1:8107/v1 --model policy --round $R \
    --prompts $PROMPTS --group $GROUP >> logs/rl-rollout.log 2>&1 || die "ROLLOUT_FAILED round $R"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
  grep ROLLOUT_SUMMARY logs/rl-rollout.log | tail -1
  log "round $R: GRPO update"
  PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/rl/grpo_update.py --round $R --adapter-in $PREV --adapter-out $(adapter $R) \
    >> logs/rl-update.log 2>&1 || die "UPDATE_FAILED round $R"
  [ -d "$(adapter $R)" ] || cp -r "$PREV" "$(adapter $R)"    # no-signal round: carry the adapter forward
  grep GRPO_DONE logs/rl-update.log | tail -1
  if [ "$R" -eq "$PILOT_END" ] || { [ "${EVAL_EVERY:-0}" -gt 0 ] && [ $((R % ${EVAL_EVERY:-1})) -eq 0 ]; }; then
    log "pilot check (amendment 9): held-out with policy $(adapter $R)"
    serve "$(adapter $R)" 8107
    $E $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8107/v1 --model policy --n 10 \
      --out results/heldout-rl-r$(printf %03d $R).json > logs/rl-heldout-r$R.log 2>&1 || die "HELDOUT_FAILED"
    kill $SERVER; wait $SERVER 2>/dev/null
    grep HELDOUT_RESULT logs/rl-heldout-r$R.log
    log "PILOT_DONE"
  fi
done
log "RL_CHAIN_DONE rounds $START-$END"

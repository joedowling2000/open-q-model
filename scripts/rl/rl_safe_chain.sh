#!/bin/bash
# Safer RL (amendment 13, after incident 7). Starts from the best pre-collapse
# adapter (chosen on clean held-out). KL penalty to round 4 (beta 0.05),
# lr 5e-6, grad clip 0.5, fresh Adam state. After every round's rollouts,
# collapse_check.py decides whether to train on them; on STOP the chain ends
# and the last good adapter stands. Held-out every 5 rounds.
# Env: START_ADAPTER (required), END (default 15), PROMPTS, GROUP.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
RUN=rl2; END=${END:-15}; PROMPTS=${PROMPTS:-128}; GROUP=${GROUP:-8}
E="taskset -c 0-4,10-14"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
adapter() { printf "out/$RUN/r%03d" "$1"; }
serve() {
  [ -f "$1/adapter.gguf" ] || $PY scripts/train/convert_lora.py "$1" --base out/merged/qwen3.5-27b-r4 \
    --outfile "$1/adapter.gguf" --outtype f16 > logs/rl2-convert.log 2>&1 || die "adapter conversion failed for $1"
  $LC/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf --lora "$1/adapter.gguf" --host 127.0.0.1 \
    --port 8107 --alias policy --reasoning off -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 --cont-batching \
    --no-webui > logs/rl2-serve.log 2>&1 &
  SERVER=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:8107/health > /dev/null && return 0; sleep 10; done
  die "policy server never came up"
}
mkdir -p out/$RUN corpus/$RUN
for R in $(seq 1 $END); do
  PREV=$(adapter $((R - 1))); [ -d "$PREV" ] || PREV=$START_ADAPTER
  [ -d "$(adapter $R)" ] && continue                          # resumable
  log "round $R: rollouts with policy $PREV"
  serve "$PREV"
  $E $QPY scripts/rl/rollout.py --url http://127.0.0.1:8107/v1 --model policy --round $R --run $RUN \
    --prompts $PROMPTS --group $GROUP >> logs/rl2-rollout.log 2>&1 || die "ROLLOUT_FAILED round $R"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
  grep ROLLOUT_SUMMARY logs/rl2-rollout.log | tail -1
  python3 scripts/rl/collapse_check.py --run $RUN --round $R | tee -a logs/rl2-checks.log \
    || { log "STOPPED by collapse check; last good adapter: $PREV"; echo "$PREV" > out/$RUN/LAST_GOOD; exit 0; }
  log "round $R: GRPO update (KL 0.05, lr 5e-6, clip 0.5)"
  FRESH=$([ "$PREV" = "$START_ADAPTER" ] && echo --fresh-optimizer)
  PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/rl/grpo_update.py --run $RUN --round $R --adapter-in $PREV --adapter-out $(adapter $R) \
    --lr 5e-6 --kl 0.05 --clip 0.5 $FRESH >> logs/rl2-update.log 2>&1 || die "UPDATE_FAILED round $R"
  [ -d "$(adapter $R)" ] || cp -r "$PREV" "$(adapter $R)"
  echo "$(adapter $R)" > out/$RUN/LAST_GOOD
  grep GRPO_DONE logs/rl2-update.log | tail -1
  if [ $((R % 5)) -eq 0 ]; then
    log "held-out check: $(adapter $R)"
    serve "$(adapter $R)"
    $E $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8107/v1 --model policy --n 10 \
      --out results/heldout-rl2-r$(printf %03d $R).json > logs/rl2-heldout-r$R.log 2>&1 || die "HELDOUT_FAILED"
    kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
    grep HELDOUT_RESULT logs/rl2-heldout-r$R.log
  fi
done
log "RL2_DONE"

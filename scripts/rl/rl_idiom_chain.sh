#!/bin/bash
# Idiom RL (amendment 16). From the final model's adapter (out/rl/r011): a full
# pass with a while/do loop scores 1 - PENALTY instead of 1.0. Otherwise the
# safe-run settings (amendment 13): KL to round 4 (beta 0.05), lr 5e-6, clip
# 0.5, fresh Adam state, collapse check before each update (here on the full-
# pass rate, limit 0.05). 64 prompts x 16 attempts: the same 1024 rollouts per
# round as before, but more chances per group of a loop-free pass to reinforce.
# Held-out and style held-out (with loop rates) every 5 rounds.
# Env: START, END, PENALTY.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
RUN=rl3; START_ADAPTER=out/rl/r011; START=${START:-1}; END=${END:-15}; PENALTY=${PENALTY:-0.5}
E="taskset -c 0-4,10-14"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; [ -n "${SERVER:-}" ] && kill $SERVER 2>/dev/null; exit 1; }
adapter() { printf "out/$RUN/r%03d" "$1"; }
serve() {
  [ -f "$1/adapter.gguf" ] || $PY scripts/train/convert_lora.py "$1" --base out/merged/qwen3.5-27b-r4 \
    --outfile "$1/adapter.gguf" --outtype f16 > logs/rl3-convert.log 2>&1 || die "adapter conversion failed for $1"
  $LC/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf --lora "$1/adapter.gguf" --host 127.0.0.1 \
    --port 8107 --alias policy --reasoning off -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 --cont-batching \
    --no-webui > logs/rl3-serve.log 2>&1 &
  SERVER=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:8107/health > /dev/null && return 0; sleep 10; done
  die "policy server never came up"
}
evaluate() {  # adapter tag
  serve "$1"
  $E $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8107/v1 --model policy --n 10 \
    --out results/heldout-$2.json > logs/rl3-heldout-$2.log 2>&1 || die "HELDOUT_FAILED $1"
  $E $QPY scripts/rl/eval_style.py --url http://127.0.0.1:8107/v1 --model policy \
    --out results/style-$2.json > logs/rl3-style-$2.log 2>&1 || die "STYLE_FAILED $1"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
  grep -h "HELDOUT_RESULT\|STYLE_RESULT" logs/rl3-heldout-$2.log logs/rl3-style-$2.log | cut -c1-160
}
mkdir -p out/$RUN corpus/$RUN
# Baseline: the style loop rate of the starting adapter (its style samples were not kept).
[ -f results/style-rl-r011-loops.json ] || { log "baseline: style held-out with loop rate, $START_ADAPTER"
  serve "$START_ADAPTER"
  $E $QPY scripts/rl/eval_style.py --url http://127.0.0.1:8107/v1 --model policy \
    --out results/style-rl-r011-loops.json > logs/rl3-style-r011.log 2>&1 || die "STYLE_FAILED baseline"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10; grep STYLE_RESULT logs/rl3-style-r011.log; }
for R in $(seq $START $END); do
  PREV=$(adapter $((R - 1))); [ -d "$PREV" ] || PREV=$START_ADAPTER
  if [ -d "$(adapter $R)" ]; then                             # resumable
    T=rl3-r$(printf %03d $R)                                  # incident 8: redo a check that failed
    [ $((R % 5)) -eq 0 ] && [ ! -f results/style-$T.json ] && { log "held-out + style check: $(adapter $R)"; evaluate "$(adapter $R)" "$T"; }
    continue
  fi
  log "round $R: rollouts with policy $PREV (loop penalty $PENALTY)"
  serve "$PREV"
  $E $QPY scripts/rl/rollout.py --url http://127.0.0.1:8107/v1 --model policy --round $R --run $RUN \
    --prompts 64 --group 16 --idiom-penalty $PENALTY --history corpus/$RUN/history.json \
    >> logs/rl3-rollout.log 2>&1 || die "ROLLOUT_FAILED round $R"
  kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
  grep ROLLOUT_SUMMARY logs/rl3-rollout.log | tail -1
  python3 scripts/rl/collapse_check.py --run $RUN --round $R --metric pass --max-drop 0.05 | tee -a logs/rl3-checks.log \
    || { log "STOPPED by collapse check; last good adapter: $PREV"; echo "$PREV" > out/$RUN/LAST_GOOD; break; }
  log "round $R: GRPO update (KL 0.05, lr 5e-6, clip 0.5)"
  FRESH=$([ "$PREV" = "$START_ADAPTER" ] && echo --fresh-optimizer)
  PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY scripts/rl/grpo_update.py --run $RUN --round $R --adapter-in $PREV --adapter-out $(adapter $R) \
    --lr 5e-6 --kl 0.05 --clip 0.5 $FRESH >> logs/rl3-update.log 2>&1 || die "UPDATE_FAILED round $R"
  [ -d "$(adapter $R)" ] || command cp -r "$PREV" "$(adapter $R)"
  echo "$(adapter $R)" > out/$RUN/LAST_GOOD
  grep GRPO_DONE logs/rl3-update.log | tail -1
  [ $((R % 5)) -eq 0 ] && { log "held-out + style check: $(adapter $R)"; evaluate "$(adapter $R)" "rl3-r$(printf %03d $R)"; }
done
LAST=$(cat out/$RUN/LAST_GOOD 2>/dev/null || echo $START_ADAPTER)
T=$(echo "$LAST" | sed 's|out/||; s|/|-|g')
[ "$LAST" != "$START_ADAPTER" ] && [ ! -f results/style-$T.json ] && { log "held-out + style check: $LAST"; evaluate "$LAST" "$T"; }
touch out/$RUN/DONE                                           # idiom_select.sh runs only after this
log "RL3_DONE rounds $START-$END; last good $LAST"

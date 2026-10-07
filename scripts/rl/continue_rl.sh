#!/bin/bash
# Amendment 14: after safe-run rounds 1-15, (i) benchmark-style problems,
# (ii) rounds 16-30 with the KL rule. Waits for the running chain to finish.
set -uo pipefail
cd /home/joedowling/Projects/qeval
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
log() { echo "[$(date +%H:%M:%S)] $*"; }
while systemctl --user list-units 'run-qrl2-*' --no-legend | grep -q .; do sleep 300; done
LAST=$(cat out/rl2/LAST_GOOD)
STOPS=$(grep -c "STOP$" logs/rl2-checks.log || true)
MAXDEG=$(grep -o "degenerate [0-9.]*%" logs/rl2-checks.log | grep -o "[0-9.]*" | sort -n | tail -1)
KL=$(python3 -c "print('0.02' if $STOPS == 0 and ${MAXDEG:-0} <= 0.1 else '0.05')")
log "rounds 1-15 done; last good $LAST; collapse stops $STOPS, max degenerate ${MAXDEG}% -> KL $KL"

log "(i) benchmark-style problems"
bash scripts/serve.sh qwen3.5-27b 8101 8 > logs/serve-qwen3.5-27b-hstyle.log 2>&1 &
S=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8101/health > /dev/null && break; sleep 10; done
taskset -c 0-4,10-14 $QPY -W ignore scripts/synth/gen_problems.py --url http://127.0.0.1:8101/v1 \
  --model qwen3.5-27b --style humaneval --target 1000 --workers 8 --out corpus/hstyle_problems.jsonl \
  >> logs/gen-hstyle.log 2>&1 || log "generation exited non-zero"
kill $S; wait $S 2>/dev/null; sleep 10
log "kept $(grep -c '"ok": true' corpus/hstyle_problems.jsonl); judged same as a benchmark task: $(grep -c 'judged the same task' corpus/hstyle_problems.jsonl)"
taskset -c 0-4,10-14 $QPY -W ignore scripts/rl/rl_pool.py >> logs/rl-pool.log 2>&1 || { log "pool rebuild failed"; exit 1; }
tail -12 logs/rl-pool.log | grep kept

log "(ii) rounds 16-30 from $LAST, KL $KL"
START_ADAPTER=$LAST START=16 END=30 KL=$KL bash scripts/rl/rl_safe_chain_v2.sh
log "CONTINUE_DONE"

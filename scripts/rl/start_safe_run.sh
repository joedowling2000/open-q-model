#!/bin/bash
# Wait for the r011/r012 held-out scores, pick the best of r005/r010/r011/r012
# on clean held-out (amendment 13), then start the safe RL run from it.
set -uo pipefail
cd /home/joedowling/Projects/qeval
while systemctl --user list-units 'run-qrlckpt*' --no-legend | grep -q .; do sleep 120; done
BEST=$(python3 - <<'PY'
import json
best = max(((json.load(open(f"results/heldout-rl-r{r:03d}.json"))["summary"]["pass_at_1_clean"], r)
            for r in (5, 10, 11, 12)))
print(f"out/rl/r{best[1]:03d} {best[0]:.4f}")
PY
)
echo "[$(date +%H:%M:%S)] start adapter (best clean held-out): $BEST"
START_ADAPTER=${BEST%% *} bash scripts/rl/rl_safe_chain.sh

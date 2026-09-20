#!/bin/bash
# Independent safety net for the gate A evaluation.
#
# after_generation.sh was edited while it was running (bash reads scripts
# incrementally, so its remaining instructions may be garbage). It is
# supervising the training run, so it cannot be restarted without killing that.
# This watches for the adapter instead, gives the original chain a fair chance
# to run the evaluation itself, and only steps in if it does not.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ADAPTER=out/lora-cpt/final
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the training adapter to appear"
until [ -f "$ADAPTER/adapter_model.safetensors" ]; do sleep 300; done
log "adapter present"

# Let the original chain start its own evaluation if it survived the edit.
sleep 600
if grep -aq "GATE_A_RESULT\|serving base + adapter" logs/eval-cpt.log 2>/dev/null; then
  log "the chain is running the evaluation; standing down"
  exit 0
fi

log "chain did not start the evaluation; running it here"
bash scripts/train/eval_checkpoint.sh "$ADAPTER" qwen3.5-27b-cpt 30 >> logs/eval-cpt.log 2>&1
log "gate A evaluation exited rc=$?"

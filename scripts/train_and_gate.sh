#!/bin/bash
# Continued pretraining, then gate A — as a clean chain.
#
# Replaces after_generation.sh, which was edited while running (bash reads a
# script incrementally, so its remaining instructions could not be trusted) and
# whose download stalled at 50/56 GB with a dead socket.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for the base weights"
until grep -aq "BASE_READY" logs/fetch-base.log; do
  if grep -aq "BASE_FAILED" logs/fetch-base.log; then log "download failed"; exit 1; fi
  sleep 60
done
log "base weights ready"

# 0.4 epochs, not 1. Measured 225 s/step on this box (Qwen3.5's DeltaNet layers
# fall back to a torch implementation — flash-linear-attention does not install
# here), so a full epoch is 30 hours before evaluation even starts. 194 steps
# with a complete cosine decay is a properly annealed checkpoint in 12 hours;
# the pre-registration fixes the corpus and the gates, not the epoch count.
log "LoRA continued pretraining on Qwen3.5-27B (8.11M tokens, 0.4 epoch)"
$PY scripts/train/train_lora.py \
  --model Qwen/Qwen3.5-27B --data data/cpt --out out/lora-cpt \
  --rank 32 --alpha 64 --lr 1e-4 --micro-batch 1 --accum 8 --epochs 0.4 \
  --save-every 100 \
  > logs/train-cpt.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then log "CPT_FAILED rc=$rc — see logs/train-cpt.log"; exit 1; fi
log "CPT_DONE"

log "gate A: scoring the checkpoint on Q-HumanEval"
bash scripts/train/eval_checkpoint.sh out/lora-cpt/final qwen3.5-27b-cpt 30 \
  >> logs/eval-cpt.log 2>&1
log "gate A exited rc=$? — see logs/eval-cpt.log"

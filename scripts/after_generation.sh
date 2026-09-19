#!/bin/bash
# When dataset generation finishes, put the GPU straight onto continued
# pretraining rather than leaving it idle until someone looks.
#
# Phase 3 of PLAN.md: LoRA on Qwen3.5-27B over the 8.11M-token human-written
# corpus (data/cpt, decontaminated — 6 files dropped for benchmark overlap).
#
# Evaluation is deliberately NOT chained on: scoring the result means merging
# the adapter, converting to GGUF and running the harness, and an unattended
# first attempt at that is more likely to waste hours than save them.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "waiting for generation to finish"
until grep -aq "GENERATION_DONE" logs/generate-run.log; do
  if ! systemctl --user is-active --quiet run-qgen-20260919-210151.service; then
    log "generation unit is gone; proceeding with whatever was produced"
    break
  fi
  sleep 300
done

kept=$(wc -l < corpus/qdataset.jsonl 2>/dev/null || echo 0)
log "dataset has $kept verified problems"

pkill -f "q8_0.gguf" 2>/dev/null       # free the GPU from the generation servers
sleep 20

log "starting LoRA continued pretraining on Qwen3.5-27B"
$PY scripts/train/train_lora.py \
  --model Qwen/Qwen3.5-27B --data data/cpt --out out/lora-cpt \
  --rank 32 --alpha 64 --lr 1e-4 --micro-batch 1 --accum 8 --epochs 1 \
  > logs/train-cpt.log 2>&1
rc=$?
log "training exited rc=$rc"
[ $rc -eq 0 ] && log "CPT_DONE" || log "CPT_FAILED — see logs/train-cpt.log"

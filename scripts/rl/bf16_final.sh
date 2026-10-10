#!/bin/bash
# The protocol's one bf16 calibration run, on the final model (reported
# separately from the Q8_0 headline). Runs after final_select.sh.
set -uo pipefail
cd /home/joedowling/Projects/qeval
log() { echo "[$(date +%H:%M:%S)] $*"; }
while systemctl --user list-units 'run-qfinal*' --no-legend | grep -q .; do sleep 600; done
grep -q FINAL_DONE logs/final-select.log || { log "final selection did not finish; not running bf16"; exit 1; }
log "converting the final model to bf16 GGUF"
/home/joedowling/venvs/jlens/bin/python /home/joedowling/Projects/serving/llama.cpp/convert_hf_to_gguf.py \
  out/merged/qwen3.5-27b-final --outtype bf16 --outfile gguf/qwen3.5-27b-final-bf16.gguf \
  > logs/convert-final-bf16.log 2>&1 || { log "convert failed"; exit 1; }
BASE_GGUF=$PWD/gguf/qwen3.5-27b-final-bf16.gguf PARITY_DATA=data/sft_r4 \
  bash scripts/train/eval_checkpoint.sh none qwen3.5-27b-final-bf16 30 >> logs/eval-final-bf16.log 2>&1
grep GATE_A_RESULT logs/eval-final-bf16.log | tail -1
log "BF16_DONE"

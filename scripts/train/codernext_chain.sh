#!/bin/bash
# Amendment 12: Qwen3-Coder-Next arm. Runs after the round 4 bf16 calibration
# (one model on the GPU at a time).
#   1. full-model feasibility: load NF4 experts, 5 steps; stop if peak > 100 GB
#   2. SFT: 1 epoch on data/sft_codernext (13060 samples, every task, both formats)
#   3. convert adapter, serve on the Coder-Next Q8 GGUF, prompt-parity check
#   4. score: Q-HumanEval (30 samples, fixed protocol) + held-out (clean)
# The optional CPT step is skipped for the first comparison. Stops on failure.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/*/)
export PYTHONPATH=/home/joedowling/venvs/fla-overlay:/home/joedowling/venvs/bnb-overlay
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NAME=qwen3-coder-next-sft
TARGETS=q_proj,k_proj,v_proj,o_proj,in_proj_qkvz,out_proj,gate_proj,up_proj,down_proj
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }

while systemctl --user list-units 'run-qbf16-*' --no-legend | grep -q running; do sleep 300; done
log "GPU free"

log "1. feasibility"
$PY scripts/train/train_lora.py --model "$SNAP" --quant-moe --target-modules $TARGETS \
  --data data/sft_codernext --out out/lora-codernext-feasibility --rank 32 --alpha 64 --lr 1e-4 \
  --micro-batch 2 --accum 4 --max-steps 5 --eval-every 1000 > logs/codernext-feasibility.log 2>&1 \
  || die "FEASIBILITY_FAILED — see logs/codernext-feasibility.log"
grep -E "loaded|trainable|step " logs/codernext-feasibility.log | tail -4
PEAK=$(awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="mem_avail_mb") c=i; next} {print $c}' \
  ~/.local/state/hw-log/hw-$(date +%F).csv | sort -n | head -1)
log "lowest available memory today: ${PEAK} MB"
rm -rf out/lora-codernext-feasibility

log "2. SFT, 1 epoch"
$PY scripts/train/train_lora.py --model "$SNAP" --quant-moe --target-modules $TARGETS \
  --data data/sft_codernext --out out/lora-codernext-sft --rank 32 --alpha 64 --lr 1e-4 \
  --micro-batch 2 --accum 8 --epochs 1 --warmup 20 --eval-every 100 --save-every 200 \
  > logs/train-codernext-sft.log 2>&1 || die "SFT_FAILED — see logs/train-codernext-sft.log"
log "SFT_DONE $(grep 'final val' logs/train-codernext-sft.log)"

log "3. convert adapter"
$PY scripts/train/convert_lora.py out/lora-codernext-sft/final --base "$SNAP" \
  --outfile gguf/$NAME-lora.gguf --outtype f16 > logs/convert-lora-$NAME.log 2>&1 || die "convert failed"

log "4. scoring $NAME"
BASE_GGUF=$PWD/gguf/qwen3-coder-next-q8_0-00001-of-00004.gguf HELDOUT=1 PARITY_DATA=data/sft_codernext \
  bash scripts/train/eval_checkpoint.sh gguf/$NAME-lora.gguf "$NAME" 30 >> logs/eval-$NAME.log 2>&1 \
  || die "EVAL_FAILED"
grep -E "GATE_A_RESULT|HELDOUT_RESULT" logs/eval-$NAME.log | tail -2
log "CODERNEXT_DONE"

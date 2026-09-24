#!/bin/bash
# Amendment 6, first step: SFT from the gate A checkpoint on the Q study bank
# alone, then gate B (Q-HumanEval, 30 samples) and the amendment 7 held-out set.
#
# The gate A LoRA is merged into the base in memory and a new, broader LoRA is
# trained on top (every projection in all 64 layers, not just the 16
# full-attention layers the CPT adapter touches). At scoring time both adapters
# are applied, in that order, to the unchanged base GGUF.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
NAME=qwen3.5-27b-sft-v5
log() { echo "[$(date +%H:%M:%S)] $*"; }

# fla-core provides the chunked DeltaNet kernels (25x the torch fallback at
# 2048 tokens; forward and gradients agree to bf16 precision). Kept in an
# overlay directory so the shared environment is untouched.
export PYTHONPATH=/home/joedowling/venvs/fla-overlay

# The first run was OOM-killed at step 203 of 482 (2026-09-24 02:10). On the
# GB10, GPU memory is system memory. With micro-batch 4, the caching allocator
# grew after a long batch and left ~10 GB free, and a later spike took the rest.
# Micro-batch 2 x accum 8 keeps the same 16-sample step with a lower activation
# peak, and expandable segments stop the allocator holding fragmented blocks.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log "SFT on data/sft_v5 (3870 samples, 2 epochs)"
$PY scripts/train/train_lora.py \
  --model Qwen/Qwen3.5-27B --data data/sft_v5 --out out/lora-sft-v5 \
  --merge-adapter out/lora-cpt/final --targets all \
  --rank 32 --alpha 64 --lr 1e-4 --micro-batch 2 --accum 8 --epochs 2 \
  --warmup 20 --eval-every 50 --save-every 100 \
  > logs/train-sft-v5.log 2>&1 || { log "SFT_FAILED — see logs/train-sft-v5.log"; exit 1; }
log "SFT_DONE"

log "gate B + held-out: scoring $NAME"
unset PYTHONPATH
HELDOUT=1 PARITY_DATA=data/sft_v5 bash scripts/train/eval_checkpoint.sh \
  gguf/qwen3.5-27b-cpt-lora.gguf,out/lora-sft-v5/final "$NAME" 30 \
  >> logs/eval-sft-v5.log 2>&1
log "EVAL exited rc=$? — see logs/eval-sft-v5.log"

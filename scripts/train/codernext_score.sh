#!/bin/bash
# Score the Coder-Next SFT adapter. Qwen's official Coder-Next GGUF uses older
# llama.cpp tensor names (fused ssm_in / ssm_out), while our converter emits
# attn_qkv / attn_gate for the adapter, so llama.cpp refused it ("LoRA tensor
# blk.0.attn_gate.weight does not exist in base model"). The base GGUF is now
# built from the HF weights with the same llama.cpp converter as the adapter.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
SNAP=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-Next/snapshots/*/)
NAME=qwen3-coder-next-sft
log() { echo "[$(date +%H:%M:%S)] $*"; }
log "building Coder-Next Q8_0 GGUF from HF weights (our converter)"
taskset -c 0-4,10-14 $PY /home/joedowling/Projects/serving/llama.cpp/convert_hf_to_gguf.py "$SNAP" \
  --outtype q8_0 --outfile gguf/qwen3-coder-next-q8_0.gguf > logs/convert-coder-next-base.log 2>&1 \
  || { log "BASE_CONVERT_FAILED"; exit 1; }
ls -la gguf/qwen3-coder-next-q8_0.gguf
log "scoring $NAME"
BASE_GGUF=$PWD/gguf/qwen3-coder-next-q8_0.gguf HELDOUT=1 PARITY_DATA=data/sft_codernext \
  bash scripts/train/eval_checkpoint.sh gguf/$NAME-lora.gguf "$NAME" 30 >> logs/eval-$NAME.log 2>&1 \
  || { log "EVAL_FAILED"; exit 1; }
grep -E "GATE_A_RESULT|HELDOUT_RESULT" logs/eval-$NAME.log | tail -2
log "CODERNEXT_SCORED"

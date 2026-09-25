#!/bin/bash
# Round 2, as one chain after the teacher finishes:
#   1. vector rewrite (amendment 3) of every loop solution, distilled and bank,
#      by the round 1 model (base + gate A + SFT v5 adapters); verified on 40 inputs
#   2. round 2 records and SFT data (Q study bank + distilled + rewrites)
#   3. SFT from the gate A checkpoint, same recipe as round 1
#   4. Q-HumanEval (gate B re-check) and held-out, full and clean (incident 4)
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
NAME=qwen3.5-27b-sft-r2
log() { echo "[$(date +%H:%M:%S)] $*"; }

while systemctl --user list-units 'run-qstep2*' --no-legend | grep -q running; do sleep 120; done
log "teacher finished; $(grep -c '"ok": true' corpus/distilled.jsonl) distilled problems"

log "1. vector rewrite with the round 1 model"
/home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
  -m gguf/qwen3.5-27b-q8_0.gguf \
  --lora gguf/qwen3.5-27b-cpt-lora.gguf,gguf/qwen3.5-27b-sft-v5-lora.gguf \
  --host 127.0.0.1 --port 8104 --alias qwen3.5-27b-sft-v5 --reasoning off \
  -ngl 999 -t 6 -tb 6 -c $((2048 * 8)) -np 8 --cont-batching --no-webui \
  > logs/serve-rewriter.log 2>&1 &
SERVER=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8104/health > /dev/null && break; sleep 10; done
$QPY scripts/synth/vector_rewrite.py --url http://127.0.0.1:8104/v1 --model qwen3.5-27b-sft-v5 \
  --in corpus/distilled.jsonl corpus/q_study_v5/questions.jsonl \
  --workers 8 --out corpus/vector_rewrites.jsonl > logs/vector-rewrite.log 2>&1
log "rewrite exited rc=$? — $(tail -1 logs/vector-rewrite.log)"
kill $SERVER; wait $SERVER 2>/dev/null; sleep 10

log "2. round 2 records and SFT data"
python3 scripts/train/bank_to_records.py > logs/round2-records.log 2>&1
python3 scripts/train/round2_records.py >> logs/round2-records.log 2>&1 || { log "records failed"; exit 1; }
$PY scripts/train/assemble_sft.py --in corpus/round2_records.jsonl --rejects /nonexistent \
  --out data/sft_r2 > logs/assemble-r2.log 2>&1 || { log "assembly failed"; exit 1; }
grep -A8 '"kept_by_task"' logs/assemble-r2.log

log "3. SFT round 2"
PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/train/train_lora.py \
  --model Qwen/Qwen3.5-27B --data data/sft_r2 --out out/lora-sft-r2 \
  --merge-adapter out/lora-cpt/final --targets all \
  --rank 32 --alpha 64 --lr 1e-4 --micro-batch 2 --accum 8 --epochs 2 \
  --warmup 20 --eval-every 100 --save-every 200 \
  > logs/train-sft-r2.log 2>&1 || { log "SFT_FAILED — see logs/train-sft-r2.log"; exit 1; }
log "SFT_DONE $(grep 'final val' logs/train-sft-r2.log)"

log "4. scoring $NAME"
HELDOUT=1 PARITY_DATA=data/sft_r2 bash scripts/train/eval_checkpoint.sh \
  gguf/qwen3.5-27b-cpt-lora.gguf,out/lora-sft-r2/final "$NAME" 30 >> logs/eval-sft-r2.log 2>&1
log "EVAL exited rc=$?"
grep -E "GATE_A_RESULT|HELDOUT_RESULT" logs/eval-sft-r2.log | tail -2
log "ROUND2_DONE"

#!/bin/bash
# Round 3: amendment 8's targeted distillation on the ~3000 generated problems.
#   1. the student (round 2 model) attempts every generated problem
#   2. the teacher (qqWen-72B) attempts only the student's failures, capped at 24 h
#   3. vector rewrite of new loop solutions, by the student (amendment 3)
#   4. records, release check, SFT from the gate A checkpoint (same recipe)
#   5. Q-HumanEval and held-out (full and clean)
# Every stage is resumable; the chain stops on the first failure.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
export QHOME=$HOME/.kx QLIC=$HOME/.kx
STUDENT=qwen3.5-27b-sft-r2
NAME=qwen3.5-27b-sft-r3
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }

serve_student() {
  /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
    -m gguf/qwen3.5-27b-q8_0.gguf \
    --lora gguf/qwen3.5-27b-cpt-lora.gguf,gguf/qwen3.5-27b-sft-r2-lora-1.gguf \
    --host 127.0.0.1 --port 8105 --alias $STUDENT --reasoning off \
    -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 --cont-batching --no-webui \
    > logs/serve-student-r2.log 2>&1 &
  SERVER=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:8105/health > /dev/null && return 0; sleep 10; done
  die "student server never came up"
}

# Student: one batch of 4 (its failures go to the teacher, which gets 4+4),
# 32 slots so more requests batch together. Measured on 29 Sep: 16 slots and
# 4+4 attempts ran at ~24 tok/s, and 67% of attempts went to problems it failed.
# Run the chain pinned to the efficiency cores (taskset -c 0-4,10-14): the
# performance-core clusters were tripping the thermal guard every ~9 s.
log "1. student attempts the generated problems"
serve_student
$QPY scripts/synth/distill.py --url http://127.0.0.1:8105/v1 --model $STUDENT \
  --in corpus/generated_problems.jsonl --samples 4 --extra 0 --workers 8 \
  --out corpus/student_r2.jsonl >> logs/student-r2.log 2>&1 || die "STUDENT_FAILED"
kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
log "student: $(tail -1 logs/student-r2.log)"

log "2. teacher on the student's failures (24 h cap, resumable)"
python3 - <<'PY'
import json
fails = {json.loads(l)["id"] for l in open("corpus/student_r2.jsonl")
         if json.loads(l).get("reason") == "no attempt verified"}
with open("corpus/teacher_targets_r3.jsonl", "w") as fh:
    for l in open("corpus/generated_problems.jsonl"):
        r = json.loads(l)
        if r.get("id") in fails:
            fh.write(l)
print(len(fails), "teacher targets")
PY
bash scripts/serve.sh qqwen-72b-rl 8103 16 > logs/serve-qqwen-72b-rl-r3.log 2>&1 &
SERVER=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8103/health > /dev/null && break; sleep 10; done
timeout 24h $QPY scripts/synth/distill.py --url http://127.0.0.1:8103/v1 --model qqwen-72b-rl \
  --in corpus/teacher_targets_r3.jsonl --samples 4 --extra 4 --workers 4 \
  --out corpus/teacher_r3.jsonl >> logs/teacher-r3.log 2>&1
rc=$?
kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
[ $rc -eq 0 ] || [ $rc -eq 124 ] || die "TEACHER_FAILED rc=$rc"
log "teacher: rc=$rc (124 = hit the 24 h cap), $(grep -c '"ok": true' corpus/teacher_r3.jsonl) kept"

log "3. vector rewrite of new loop solutions"
serve_student
$QPY scripts/synth/vector_rewrite.py --url http://127.0.0.1:8105/v1 --model $STUDENT \
  --in corpus/student_r2.jsonl corpus/teacher_r3.jsonl --workers 8 \
  --out corpus/vector_rewrites.jsonl >> logs/vector-rewrite.log 2>&1 || die "REWRITE_FAILED"
kill $SERVER; wait $SERVER 2>/dev/null; sleep 10
log "rewrite: $(tail -1 logs/vector-rewrite.log)"

log "4. records, release check, SFT round 3"
SOLVED="corpus/distilled.jsonl corpus/student_r2.jsonl corpus/teacher_r3.jsonl"
python3 scripts/train/bank_to_records.py > logs/round3-records.log 2>&1 || die "bank records failed"
rm -f corpus/disqualified.json
python3 scripts/train/round2_records.py --solved $SOLVED --out corpus/round3_records.jsonl \
  >> logs/round3-records.log 2>&1 || die "records failed"
$QPY scripts/train/release_check.py corpus/round3_records.jsonl | tee -a logs/round3-records.log \
  || die "release check failed"
python3 scripts/train/round2_records.py --solved $SOLVED --out corpus/round3_records.jsonl \
  >> logs/round3-records.log 2>&1 || die "records failed"
$PY scripts/train/assemble_sft.py --in corpus/round3_records.jsonl --rejects /nonexistent \
  --out data/sft_r3 > logs/assemble-r3.log 2>&1 || die "assembly failed"
PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/train/train_lora.py \
  --model Qwen/Qwen3.5-27B --data data/sft_r3 --out out/lora-sft-r3 \
  --merge-adapter out/lora-cpt/final --targets all \
  --rank 32 --alpha 64 --lr 1e-4 --micro-batch 2 --accum 8 --epochs 2 \
  --warmup 20 --eval-every 200 --save-every 400 \
  > logs/train-sft-r3.log 2>&1 || die "SFT_FAILED — see logs/train-sft-r3.log"
log "SFT_DONE $(grep 'final val' logs/train-sft-r3.log)"

log "5. scoring $NAME"
HELDOUT=1 PARITY_DATA=data/sft_r3 bash scripts/train/eval_checkpoint.sh \
  gguf/qwen3.5-27b-cpt-lora.gguf,out/lora-sft-r3/final "$NAME" 30 >> logs/eval-sft-r3.log 2>&1 \
  || die "EVAL_FAILED"
grep -E "GATE_A_RESULT|HELDOUT_RESULT" logs/eval-sft-r3.log | tail -2
log "ROUND3_DONE"

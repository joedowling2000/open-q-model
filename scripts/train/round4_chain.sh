#!/bin/bash
# Round 4 (amendment 10): benchmark-format alignment and MBPP, on top of merged round 3.
#   0. confirm the merged round 3 model reproduces round 3 (39.8% +/- 3), else stop
#   1. MBPP: same-task judge against Q-HumanEval, then references (Qwen3.5-27B)
#   2. merged round 3 attempts the MBPP problems (no run-time adapters)
#   3. records + release check + harness-format rendering (each rename re-verified)
#   4. SFT on top of merged round 3: 1 epoch, prose + harness prompts
#   5. merge, convert, score (Q-HumanEval + held-out)
# Do NOT pin the whole chain: llama.cpp's decode for this model uses CPU threads,
# and on the efficiency cores it fell from ~1.65 to 0.83 tok/s per slot (1 Oct).
# Only the CPU-heavy verification (q + Python) is pinned to the efficiency cores,
# which is what was tripping the thermal guard (29 Sep). Stops on the first
# failure. FROM=n resumes.
E="taskset -c 0-4,10-14"
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
FROM=${FROM:-0}
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
up() { for i in $(seq 180); do curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0; sleep 10; done; die "server on $1 never came up"; }

if [ "$FROM" -le 0 ]; then
while systemctl --user list-units 'run-qevalmerged2*' --no-legend | grep -q running; do sleep 120; done
P1=$(python3 -c "import json;print(json.load(open('results/qwen3.5-27b-r3-merged-scored.json'))['pass_at_1'])") \
  || die "merged round 3 has no score"
log "merged round 3 pass@1 = $P1 (adapters at run time: 0.398)"
python3 -c "import sys;sys.exit(0 if abs($P1-0.398)<=0.03 else 1)" || die "MERGE_MISMATCH: not reproducing round 3; stopping"
fi

if [ "$FROM" -le 1 ]; then
log "1. MBPP judge + references (qwen3.5-27b)"
bash scripts/serve.sh qwen3.5-27b 8101 8 > logs/serve-qwen3.5-27b-mbpp.log 2>&1 & S=$!; up 8101
$E $QPY -W ignore scripts/synth/mbpp_reference.py --url http://127.0.0.1:8101/v1 --model qwen3.5-27b \
  --workers 8 --out corpus/mbpp_rebuilt.jsonl >> logs/mbpp-reference.log 2>&1 || die "MBPP_FAILED"
kill $S; wait $S 2>/dev/null; sleep 10
log "mbpp: $(grep -c '"ok": true' corpus/mbpp_rebuilt.jsonl) kept; $(grep -c 'judged the same task' corpus/mbpp_rebuilt.jsonl) judged same as a benchmark task"
fi

if [ "$FROM" -le 2 ]; then
log "2. merged round 3 attempts MBPP"
$LC/build/bin/llama-server -m gguf/qwen3.5-27b-r3-q8_0.gguf --host 127.0.0.1 --port 8106 \
  --alias qwen3.5-27b-r3 --reasoning off -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 \
  --cont-batching --no-webui > logs/serve-r3-merged-student.log 2>&1 & S=$!; up 8106
$E $QPY scripts/synth/distill.py --url http://127.0.0.1:8106/v1 --model qwen3.5-27b-r3 \
  --in corpus/mbpp_rebuilt.jsonl --samples 4 --extra 4 --workers 8 \
  --out corpus/student_mbpp.jsonl >> logs/student-mbpp.log 2>&1 || die "STUDENT_FAILED"
kill $S; wait $S 2>/dev/null; sleep 10
log "student on MBPP: $(grep -c '"ok": true' corpus/student_mbpp.jsonl) solved"
fi

if [ "$FROM" -le 3 ]; then
log "3. records, release check, harness rendering"
SOLVED="corpus/distilled.jsonl corpus/student_r2.jsonl corpus/teacher_r3.jsonl corpus/student_mbpp.jsonl"
python3 scripts/train/bank_to_records.py > logs/round4-records.log 2>&1 || die "bank records failed"
rm -f corpus/disqualified.json
python3 scripts/train/round2_records.py --solved $SOLVED --out corpus/round4_base_records.jsonl >> logs/round4-records.log 2>&1 || die "records failed"
$E $QPY scripts/train/release_check.py corpus/round4_base_records.jsonl | tee -a logs/round4-records.log || die "release check failed"
python3 scripts/train/round2_records.py --solved $SOLVED --out corpus/round4_base_records.jsonl >> logs/round4-records.log 2>&1 || die "records failed"
$E $QPY scripts/train/round4_records.py | tee -a logs/round4-records.log || die "rendering failed"
$PY scripts/train/assemble_sft.py --in corpus/round4_records.jsonl --rejects /nonexistent \
  --out data/sft_r4 > logs/assemble-r4.log 2>&1 || die "assembly failed"
grep -E '"kept_by_task"|dropped_contaminated' -A6 logs/assemble-r4.log | head -9
fi

if [ "$FROM" -le 4 ]; then
log "4. SFT round 4 on top of merged round 3"
PYTHONPATH=/home/joedowling/venvs/fla-overlay PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
$PY scripts/train/train_lora.py \
  --model out/merged/qwen3.5-27b-r3 --data data/sft_r4 --out out/lora-sft-r4 \
  --targets all --rank 32 --alpha 64 --lr 5e-5 --micro-batch 2 --accum 8 --epochs 1 \
  --warmup 20 --eval-every 100 --save-every 200 \
  > logs/train-sft-r4.log 2>&1 || die "SFT_FAILED — see logs/train-sft-r4.log"
log "SFT_DONE $(grep 'final val' logs/train-sft-r4.log)"
fi

if [ "$FROM" -le 5 ]; then
log "5. merge, convert, score"
$PY scripts/train/merge_into_checkpoint.py --base out/merged/qwen3.5-27b-r3 \
  --out out/merged/qwen3.5-27b-r4 out/lora-sft-r4/final > logs/merge-r4.log 2>&1 || die "merge failed"
$PY $LC/convert_hf_to_gguf.py out/merged/qwen3.5-27b-r4 --outtype q8_0 \
  --outfile gguf/qwen3.5-27b-r4-q8_0.gguf > logs/convert-r4.log 2>&1 || die "convert failed"
BASE_GGUF=$PWD/gguf/qwen3.5-27b-r4-q8_0.gguf HELDOUT=1 PARITY_DATA=data/sft_r4 \
  bash scripts/train/eval_checkpoint.sh none qwen3.5-27b-r4 30 >> logs/eval-r4.log 2>&1 || die "EVAL_FAILED"
grep -E "GATE_A_RESULT|HELDOUT_RESULT" logs/eval-r4.log | tail -2
log "ROUND4_DONE"
fi

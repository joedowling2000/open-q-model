#!/bin/bash
# Score a LoRA checkpoint on Q-HumanEval — gate A of the pre-registration.
#
#   eval_checkpoint.sh <adapter-dir> <name> [samples]
#
# No merging. Merging a LoRA into Qwen3.5 and re-saving rewrites the checkpoint
# layout (the original has 488 tensors, transformers exposes 321 fused ones),
# and llama.cpp then refuses the result with "missing tensor blk.24.attn_norm".
# llama.cpp applies adapters directly, so the base GGUF we already benchmarked
# is reused unchanged — which also keeps the comparison honest, since the base
# and the trained model then differ by exactly the adapter.
set -uo pipefail
ADAPTER="$1"; NAME="$2"; SAMPLES="${3:-30}"
ROOT=/home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
PORT=8102
BASE_GGUF="$ROOT/gguf/qwen3.5-27b-q8_0.gguf"
LORA_GGUF="$ROOT/gguf/$NAME-lora.gguf"
cd "$ROOT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/*/ 2>/dev/null | head -1)
[ -z "$SNAP" ] && { log "base HF snapshot missing; needed for adapter conversion"; exit 1; }

log "converting adapter to GGUF"
$PY /home/joedowling/Projects/serving/llama.cpp/convert_lora_to_gguf.py \
  "$ADAPTER" --base "$SNAP" --outfile "$LORA_GGUF" --outtype f16 \
  > "logs/convert-lora-$NAME.log" 2>&1 || { log "adapter conversion failed"; exit 1; }

log "serving base + adapter"
setsid /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
  -m "$BASE_GGUF" --lora "$LORA_GGUF" \
  --host 127.0.0.1 --port $PORT --alias "$NAME" --reasoning off \
  -ngl 999 -t 6 -tb 6 -c $((2048 * SAMPLES)) -np "$SAMPLES" --cont-batching --no-webui \
  > "logs/serve-$NAME.log" 2>&1 &
tries=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10; tries=$((tries+1))
  [ $tries -gt 120 ] && { log "server never came up"; exit 1; }
done

export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export QEVAL_MAX_CONCURRENT=1
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1
QPY=/home/joedowling/venvs/qeval/bin/python

log "generating $SAMPLES samples/problem"
rm -f "results/$NAME.jsonl"
( cd q-evaluation-harness && $QPY -m src.cli generate q-humaneval "openai/$NAME" \
    --backend litellm -n "$SAMPLES" -o "$ROOT/results/$NAME.jsonl" ) \
    > "logs/gen-$NAME.log" 2>&1
gen_rc=$?
pkill -f "$NAME-lora.gguf"
sleep 5
[ $gen_rc -ne 0 ] && { log "generation failed"; exit 1; }

log "grading"
( cd q-evaluation-harness && $QPY -m src.cli execute "$ROOT/results/$NAME.jsonl" \
    q-humaneval -o "$ROOT/results/$NAME-scored.json" ) > "logs/exec-$NAME.log" 2>&1 || {
  log "grading failed"; exit 1; }

$QPY - "results/$NAME-scored.json" "$NAME" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = d.get("metrics", d)
passes = {k: round(v, 4) for k, v in m.items() if str(k).lower().startswith("pass")}
print("GATE_A_RESULT", sys.argv[2], json.dumps(passes), flush=True)
print("baseline Qwen3.5-27B pass@1 = 0.1024; qqWen-32B-RL = 0.3839", flush=True)
PY
log "EVAL_DONE $NAME"

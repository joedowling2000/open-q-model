#!/bin/bash
# Score a LoRA checkpoint on Q-HumanEval — gate A of the pre-registration.
#
#   eval_checkpoint.sh <adapter-dir>[,<adapter-dir|file.gguf>...] <name> [samples]
#
# Several adapters stack in the order given: SFT trains a new LoRA on top of the
# gate A adapter merged into the base, so it is scored as
# `gguf/qwen3.5-27b-cpt-lora.gguf,out/lora-sft-v5/final`. They must go to
# llama-server comma-separated in ONE --lora: repeating the flag is accepted,
# warned about in the log, and only the last adapter is applied.
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
cd "$ROOT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

SNAP=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/*/ 2>/dev/null | head -1)
[ -z "$SNAP" ] && { log "base HF snapshot missing; needed for adapter conversion"; exit 1; }

LORAS=()
IFS=',' read -ra PARTS <<< "$ADAPTER"
for i in "${!PARTS[@]}"; do
  part="${PARTS[$i]}"
  if [[ "$part" == *.gguf ]]; then
    LORAS+=("$ROOT/$part")
    continue
  fi
  out="$ROOT/gguf/$NAME-lora$([ ${#PARTS[@]} -gt 1 ] && echo "-$i").gguf"
  log "converting adapter $part to GGUF"
  $PY scripts/train/convert_lora.py \
    "$part" --base "$SNAP" --outfile "$out" --outtype f16 \
    > "logs/convert-lora-$NAME-$i.log" 2>&1 || { log "adapter conversion failed"; exit 1; }
  LORAS+=("$out")
done
LORA_ARG=$(IFS=','; echo "${LORAS[*]}")
LORA_GGUF="${LORAS[-1]}"

log "serving base + adapter"
setsid /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
  -m "$BASE_GGUF" --lora "$LORA_ARG" \
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

# SFT checkpoints: refuse to score if the served prompt differs from training's.
if [ -n "${PARITY_DATA:-}" ]; then
  $PY scripts/train/check_prompt_parity.py --port $PORT --model "$NAME" --data "$PARITY_DATA" \
    || { log "prompt parity check failed; not scoring"; pkill -f -- "--alias $NAME "; exit 1; }
fi

log "generating $SAMPLES samples/problem"
rm -f "results/$NAME.jsonl"
( cd q-evaluation-harness && $QPY -m src.cli generate q-humaneval "openai/$NAME" \
    --backend litellm -n "$SAMPLES" -o "$ROOT/results/$NAME.jsonl" ) \
    > "logs/gen-$NAME.log" 2>&1
gen_rc=$?
# Amendment 7: the held-out set is scored on the same server, same adapters.
if [ "${HELDOUT:-0}" = "1" ] && [ $gen_rc -eq 0 ]; then
  log "scoring the held-out set"
  $QPY scripts/train/eval_heldout.py --url "http://127.0.0.1:$PORT/v1" --model "$NAME" \
    --n 10 --out "results/heldout-$NAME.json" > "logs/heldout-$NAME.log" 2>&1 \
    || log "held-out scoring failed; see logs/heldout-$NAME.log"
  grep HELDOUT_RESULT "logs/heldout-$NAME.log"
fi
pkill -f -- "--alias $NAME "
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

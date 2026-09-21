#!/bin/bash
# Finish gate A for a checkpoint whose generation run lost problems.
#
#   resume_eval.sh <name> [samples]
#
# The first CPT run (eval_checkpoint.sh, 2026-09-21) sent each problem as one
# n=30 request under litellm's 600 s default timeout. With the adapter applied
# at runtime the server decodes ~1 tok/s per slot, so 30 x 512 tokens overran
# it, and the harness wrote a timed-out problem as 30 empty completions, which
# grade as failures. 14 of the first 76 problems were lost that way.
#
# This keeps every problem that came back with real completions, regenerates
# the empty ones plus the ones never reached, on the same server settings,
# and refuses to grade if any problem is still empty.
set -uo pipefail
NAME="$1"; SAMPLES="${2:-30}"
ROOT=/home/joedowling/Projects/qeval
PORT=8102
BASE_GGUF="$ROOT/gguf/qwen3.5-27b-q8_0.gguf"
LORA_GGUF="$ROOT/gguf/$NAME-lora.gguf"
PART1="$ROOT/results/$NAME.part1.jsonl"
PART2="$ROOT/results/$NAME.part2.jsonl"
OUT="$ROOT/results/$NAME.jsonl"
QPY=/home/joedowling/venvs/qeval/bin/python
cd "$ROOT"

log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$PART1" ] || mv "$OUT" "$PART1"

# Problems to (re)generate: every one without 30 non-empty completions.
MISSING=$($QPY - "$PART1" "$SAMPLES" <<'PY'
import json, sys
from collections import defaultdict
good = defaultdict(int)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["completion"].strip():
        good[r["task_id"]] += 1
done = {t for t, n in good.items() if n == int(sys.argv[2])}
print(",".join(str(t) for t in range(164) if t not in done))
PY
)
log "regenerating $(echo "$MISSING" | tr ',' '\n' | wc -l) problems: $MISSING"

log "serving base + adapter (same settings as the first run)"
setsid /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
  -m "$BASE_GGUF" --lora "$LORA_GGUF" \
  --host 127.0.0.1 --port $PORT --alias "$NAME" --reasoning off \
  -ngl 999 -t 6 -tb 6 -c $((2048 * SAMPLES)) -np "$SAMPLES" --cont-batching --no-webui \
  >> "logs/serve-$NAME.log" 2>&1 &
tries=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 10; tries=$((tries+1))
  [ $tries -gt 120 ] && { log "server never came up"; exit 1; }
done

export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
# Two problems in flight so the slots finishing early are not left idle while
# a problem's slowest samples finish; with 60 requests on 30 slots a request
# can wait, hence a timeout far above the ~600 s a request takes alone.
export QEVAL_MAX_CONCURRENT=2
export QEVAL_REQUEST_TIMEOUT=7200
export QEVAL_ONLY_TASKS="$MISSING"
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1

log "generating $SAMPLES samples/problem"
rm -f "$PART2"
( cd q-evaluation-harness && $QPY -m src.cli generate q-humaneval "openai/$NAME" \
    --backend litellm -n "$SAMPLES" -o "$PART2" ) >> "logs/gen-$NAME.log" 2>&1
gen_rc=$?
pkill -f "$NAME-lora.gguf"
sleep 5
[ $gen_rc -ne 0 ] && { log "generation failed"; exit 1; }

log "merging"
$QPY - "$PART1" "$PART2" "$OUT" "$SAMPLES" <<'PY' || { log "merge refused"; exit 1; }
import json, sys
from collections import defaultdict
p1, p2, out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
new = [json.loads(l) for l in open(p2)]
redone = {r["task_id"] for r in new}
old = [json.loads(l) for l in open(p1)]
rows = [r for r in old if r["task_id"] not in redone] + new
by = defaultdict(list)
for r in rows:
    by[r["task_id"]].append(r)
empty = sorted(t for t, v in by.items() if any(not r["completion"].strip() for r in v))
short = sorted(t for t, v in by.items() if len(v) != n)
if len(by) != 164 or empty or short:
    sys.exit(f"incomplete: {len(by)} problems, empty={empty}, wrong count={short}")
rows.sort(key=lambda r: (r["task_id"], r["sample_index"]))
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"merged {len(rows)} completions: {len(old) - sum(r['task_id'] in redone for r in old)} kept, {len(new)} new")
PY

log "grading"
( cd q-evaluation-harness && $QPY -m src.cli execute "$OUT" \
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

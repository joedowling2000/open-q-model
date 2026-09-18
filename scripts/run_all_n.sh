#!/bin/bash
# Score the three models on Q-HumanEval, one at a time, as their downloads land.
# N_SAMPLES controls samples per problem; outputs carry an -n<N> suffix beyond 10.
#
# Per model: wait for the download marker -> convert to Q8_0 GGUF -> serve with
# llama.cpp -> generate 10 samples/problem -> grade with q -> stop the server.
# Serial by design: one model at a time keeps memory and heat predictable.
set -uo pipefail
cd /home/joedowling/Projects/qeval
ROOT=/home/joedowling/Projects/qeval

N_SAMPLES="${N_SAMPLES:-10}"
# Suffix keeps each pass's artefacts separate; the first 10-sample pass wrote
# unsuffixed names, so keep those as-is for continuity.
SUFFIX=""; [ "$N_SAMPLES" != "10" ] && SUFFIX="-n$N_SAMPLES"
PORT=8099
# llama.cpp rejects a request for n completions when n > slots
# ("n_cmpl cannot be greater than the number of slots"), so the slot count has
# to be at least the sample count.
# Problems in flight x samples each. Measured on this box (Qwen3.5-27B Q8_0):
#   10 slots (1 problem in flight)  -> 3.9 t/s per slot, 159 s per problem
#   40 slots (4 problems in flight) -> 0.74 t/s per slot, 211 s per problem
# Bigger batches LOSE here: 273 GB/s is already saturated at batch 10, and the
# extra KV cache traffic and scheduling overhead cost more than they buy.
CONCURRENCY="${QEVAL_MAX_CONCURRENT:-1}"
SLOTS=$(( N_SAMPLES * CONCURRENCY ))
[ "$SLOTS" -gt 60 ] && SLOTS=60 && CONCURRENCY=$(( SLOTS / N_SAMPLES ))
export QEVAL_MAX_CONCURRENT="$CONCURRENCY"
export QHOME=$HOME/.kx QLIC=$HOME/.kx PATH=$HOME/.kx/bin:$PATH PYKX_THREADING=1
export OPENAI_API_KEY=local
export OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"

run_one() {
  local repo="$1" name="$2"
  echo "=== $name: waiting for download ==="
  until grep -aq "^OK $repo\$" logs/downloads.log; do sleep 60; done

  echo "=== $name: converting ==="
  bash scripts/prepare_model.sh "$repo" "$name" > "logs/convert-$name.log" 2>&1 || {
    echo "CONVERT_FAILED $name"; return 1; }

  echo "=== $name: serving ==="
  setsid bash scripts/serve.sh "$name" "$PORT" "$SLOTS" > "logs/serve-$name.log" 2>&1 &
  local tries=0
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
    sleep 10; tries=$((tries+1))
    [ $tries -gt 90 ] && { echo "SERVE_FAILED $name"; return 1; }
  done

  # The harness resolves dataset paths relative to the working directory
  # ("./datasets/q_humaneval.jsonl"), so generation and grading must run from
  # inside the repo. Outputs stay absolute.
  # The harness APPENDS to its solutions file. A restarted run therefore mixes
  # the previous attempt's samples into this one (1800 rows instead of 1640,
  # some problems with 20 samples), which quietly corrupts pass@k. Start clean.
  rm -f "$ROOT/results/$name$SUFFIX.jsonl"

  echo "=== $name: generating ($N_SAMPLES samples) ==="
  ( cd q-evaluation-harness && /home/joedowling/venvs/qeval/bin/qeval generate \
      q-humaneval "openai/$name" --backend litellm -n "$N_SAMPLES" \
      -o "$ROOT/results/$name$SUFFIX.jsonl" ) > "logs/gen-$name$SUFFIX.log" 2>&1
  local gen_rc=$?

  # Stop the server before grading: q execution wants the CPU, and holding 30 GB
  # of weights for nothing helps no one.
  pkill -f "llama-server .*$name-q8_0.gguf"
  sleep 5
  [ $gen_rc -ne 0 ] && { echo "GENERATE_FAILED $name"; return 1; }

  echo "=== $name: grading ==="
  ( cd q-evaluation-harness && /home/joedowling/venvs/qeval/bin/qeval execute \
      "$ROOT/results/$name$SUFFIX.jsonl" q-humaneval \
      -o "$ROOT/results/$name$SUFFIX-scored.json" ) > "logs/exec-$name$SUFFIX.log" 2>&1 || {
    echo "EXECUTE_FAILED $name"; return 1; }

  python3 - "results/$name$SUFFIX-scored.json" "$name" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
m = d.get("metrics", d)
print("RESULT", sys.argv[2], json.dumps({k: v for k, v in m.items()
      if str(k).lower().startswith("pass")}), flush=True)
PY
}

run_one "Qwen/Qwen3.5-27B"                      qwen3.5-27b
run_one "google/gemma-4-31B-it"                 gemma-4-31b-it
run_one "morganstanley/qqWen-32B-RL-Reasoning"  qqwen-32b-rl
echo ALL_EVALS_DONE

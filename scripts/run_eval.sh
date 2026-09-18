#!/bin/bash
# Run Q-HumanEval against the locally served model.
#   scripts/run_eval.sh <short-name> [num-samples] [port]
set -euo pipefail
NAME="$1"; N="${2:-10}"; PORT="${3:-8080}"
cd /home/joedowling/Projects/qeval/q-evaluation-harness
export OPENAI_API_KEY=local OPENAI_API_BASE="http://127.0.0.1:$PORT/v1" OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export PYKX_THREADING=1 QARGS="${QARGS:-}"
/home/joedowling/venvs/qeval/bin/qeval run q-humaneval "openai/$NAME" \
  --backend litellm --num-samples "$N" \
  --output-dir /home/joedowling/Projects/qeval/results/$NAME 2>&1

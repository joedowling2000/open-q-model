#!/bin/bash
# Serve a GGUF over an OpenAI-compatible API for the harness.
#   scripts/serve.sh <short-name> [port] [parallel-slots]
# Parallel slots matter: the harness asks for many samples per problem, and
# batching them is what makes a 27B+ model tractable at 273 GB/s.
set -euo pipefail
NAME="$1"; PORT="${2:-8080}"; NP="${3:-8}"
# --reasoning off matters for comparability: Qwen3.5's chat template turns
# thinking on by default, and a reasoning model would spend the 512-token budget
# on <think> and emit no code. The leaderboard's one-shot protocol is explicitly
# "no extended thinking", so every model here runs with it off.
exec /home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server \
  -m /home/joedowling/Projects/qeval/gguf/$NAME-q8_0.gguf \
  --host 127.0.0.1 --port "$PORT" --alias "$NAME" \
  --reasoning off \
  -ngl 999 -t "${THREADS:-6}" -tb "${THREADS:-6}" \
  -c $((2048 * NP)) -np "$NP" --cont-batching --no-webui
# -t 6 instead of the default 20: with every layer on the GPU the CPU threads
# only sample and tokenise, but llama.cpp spin-waits on them, heating the very
# CPU clusters (ts0p/ts1p) that trip thermal-guard. Measured 2026-09-18: 342
# thermal freezes in two hours, 14% of wall time frozen, on top of the duty
# cycle's 17%.

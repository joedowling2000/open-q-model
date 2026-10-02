#!/bin/bash
# Amendment 11 screen: fetch Q8_0 GGUFs of the two coder bases, verified by size.
#   Qwen3-Coder-30B-A3B-Instruct: unsloth's Q8_0 (Qwen publishes no GGUF)
#   Qwen3-Coder-Next: Qwen's own Q8_0, 4 split files (llama.cpp loads the set)
# Runs outside runlog: thermal-guard freezes stall downloads (see fetch_teacher.sh).
set -euo pipefail
DIR=/home/joedowling/Projects/qeval/gguf
log() { echo "[$(date +%H:%M:%S)] $*"; }

fetch() {  # repo  remote-path  local-name
  local repo=$1 path=$2 out=$DIR/$3
  local want
  want=$(curl -sfL "https://huggingface.co/api/models/$repo?blobs=true" \
    | python3 -c "import json,sys;print(next(x['size'] for x in json.load(sys.stdin)['siblings'] if x['rfilename']=='$path'))")
  for try in $(seq 1 30); do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && { log "$3 complete"; return 0; }
    log "$3 attempt $try ($have of $want)"
    curl -fL -C - --http1.1 --retry 5 --retry-delay 30 --speed-limit 1000000 --speed-time 120 \
      -o "$out" "https://huggingface.co/$repo/resolve/main/$path" || sleep 30
  done
  log "$3 incomplete"; echo FETCH_FAILED; exit 1
}

fetch unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf \
  qwen3-coder-30b-a3b-q8_0.gguf
for i in 1 2 3 4; do
  fetch Qwen/Qwen3-Coder-Next-GGUF "Qwen3-Coder-Next-Q8_0/Qwen3-Coder-Next-Q8_0-0000$i-of-00004.gguf" \
    "qwen3-coder-next-q8_0-0000$i-of-00004.gguf"
done
echo FETCH_DONE

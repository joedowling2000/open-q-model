#!/bin/bash
# Stage the open-q-27b release for Hugging Face. Hard links, so no extra disk.
# Usage: assemble.sh [SOURCE_NAME]   (default qwen3.5-27b-final; the idiom
# model would be qwen3.5-27b-idiom, once amendment 16 has decided)
set -euo pipefail
cd /home/joedowling/Projects/qeval
SRC=${1:-qwen3.5-27b-final}
STAGE=release/stage/open-q-27b
rm -rf "$STAGE"; mkdir -p "$STAGE"
for f in out/merged/$SRC/*; do ln "$f" "$STAGE/$(basename "$f")"; done
ln gguf/$SRC-q8_0.gguf "$STAGE/open-q-27b-Q8_0.gguf"
[ -f gguf/$SRC-bf16.gguf ] && ln gguf/$SRC-bf16.gguf "$STAGE/open-q-27b-BF16.gguf"
command cp -f release/open-q-27b/README.md "$STAGE/README.md"
( cd "$STAGE" && sha256sum *.gguf *.safetensors > SHA256SUMS )
du -shL "$STAGE"; ls "$STAGE"

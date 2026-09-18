#!/bin/bash
# Convert a downloaded HF model to a Q8_0 GGUF for llama.cpp.
#   scripts/prepare_model.sh <hf-repo-id> <short-name>
# Q8_0 is near-lossless and ~1 byte/param, so a 32B fits in ~35 GB and runs
# far faster than bf16 on this box's 273 GB/s of bandwidth. Every model in the
# comparison uses the same quantisation, so the comparison stays fair.
set -euo pipefail
REPO="$1"; NAME="$2"
LC=/home/joedowling/Projects/serving/llama.cpp
OUT=/home/joedowling/Projects/qeval/gguf/$NAME-q8_0.gguf
[ -f "$OUT" ] && { echo "exists: $OUT"; exit 0; }
SNAP=$(/home/joedowling/venvs/jlens/bin/python - "$REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json","*.txt","*.safetensors","*.model","*.jinja"]))
PY
)
echo "converting $SNAP -> $OUT"
/home/joedowling/venvs/jlens/bin/python $LC/convert_hf_to_gguf.py "$SNAP" --outtype q8_0 --outfile "$OUT"
ls -lh "$OUT"

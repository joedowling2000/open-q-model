#!/bin/bash
# Turn a LoRA adapter into something the evaluation harness can score.
#
#   merge_and_convert.sh <base-hf-repo> <adapter-dir> <short-name>
#
# LoRA weights cannot be served by llama.cpp directly, so: merge into the base,
# convert to Q8_0 GGUF (the same quantisation every benchmarked model used), and
# delete the merged bf16 copy, which is 54 GB of nothing once the GGUF exists.
set -euo pipefail
BASE="$1"; ADAPTER="$2"; NAME="$3"
ROOT=/home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
MERGED="$ROOT/out/merged-$NAME"
GGUF="$ROOT/gguf/$NAME-q8_0.gguf"

[ -f "$GGUF" ] && { echo "exists: $GGUF"; exit 0; }

free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt 120 ]; then
  echo "only ${free_gb}G free; merge needs ~54G plus ~30G for the GGUF"; exit 1
fi

echo "merging $ADAPTER into $BASE"
$PY - "$BASE" "$ADAPTER" "$MERGED" <<'PY'
import sys, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base, adapter, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter)
model = model.merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
PY

echo "converting to Q8_0"
$PY /home/joedowling/Projects/serving/llama.cpp/convert_hf_to_gguf.py \
  "$MERGED" --outtype q8_0 --outfile "$GGUF"

rm -rf "$MERGED"
ls -lh "$GGUF"
echo "MERGE_CONVERT_DONE $NAME"

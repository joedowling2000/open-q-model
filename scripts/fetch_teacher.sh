#!/bin/bash
# Fetch the amendment 5 teacher, qqWen-72B-RL, as a Q8_0 GGUF.
#
# The upstream repo is 291 GB of fp32 safetensors and the disk has ~150 GB free,
# so the conversion route (download, convert, quantise) cannot run here. The
# mradermacher Q8_0 GGUF is the same weights at the quantisation every model in
# this project is scored at. It ships in two parts that llama.cpp needs joined;
# joining part by part and deleting each as it lands keeps the peak at ~116 GB
# rather than 155.
set -euo pipefail
DIR=/home/joedowling/Projects/qeval/gguf
URL=https://huggingface.co/mradermacher/qqWen-72B-RL-GGUF/resolve/main
OUT=$DIR/qqwen-72b-rl-q8_0.gguf
log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$OUT" ] && { log "already present"; echo TEACHER_READY; exit 0; }
for p in 1 2; do
  f=$DIR/qqWen-72B-RL.Q8_0.gguf.part${p}of2
  for try in 1 2 3 4 5 6 7 8; do
    log "part $p, attempt $try"
    curl -fL -C - --retry 5 --retry-delay 30 --speed-limit 1000000 --speed-time 120 \
      -o "$f" "$URL/qqWen-72B-RL.Q8_0.gguf.part${p}of2" && break
    sleep 60
  done
done
log "joining"
cat $DIR/qqWen-72B-RL.Q8_0.gguf.part1of2 > "$OUT.partial"
rm $DIR/qqWen-72B-RL.Q8_0.gguf.part1of2
cat $DIR/qqWen-72B-RL.Q8_0.gguf.part2of2 >> "$OUT.partial"
rm $DIR/qqWen-72B-RL.Q8_0.gguf.part2of2
mv "$OUT.partial" "$OUT"
ls -la "$OUT"
echo TEACHER_READY

#!/bin/bash
# Fetch the amendment 5 teacher, qqWen-72B-RL, as a Q8_0 GGUF.
#
# The upstream repo is 291 GB of fp32 safetensors and the disk has ~150 GB free,
# so the conversion route (download, convert, quantise) cannot run here. The
# mradermacher Q8_0 GGUF is the same weights at the quantisation every model in
# this project is scored at. It ships in two parts that llama.cpp needs joined;
# joining part by part and deleting each as it lands keeps the peak at ~116 GB
# rather than 155.
#
# Run it outside runlog. thermal-guard freezes every run-* unit when the SoC is
# hot; a frozen curl stalls its HTTP/2 stream until the server cancels it. The
# first fetch lost part 1 to eight such cancellations, and an earlier version of
# this script then joined the partial file anyway. Sizes are now checked.
set -euo pipefail
DIR=/home/joedowling/Projects/qeval/gguf
URL=https://huggingface.co/mradermacher/qqWen-72B-RL-GGUF/resolve/main
OUT=$DIR/qqwen-72b-rl-q8_0.gguf
log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$OUT" ] && { log "already present"; echo TEACHER_READY; exit 0; }
declare -A SIZE=([1]=38657073856 [2]=38599966464)   # overwritten below from the API
for p in 1 2; do
  SIZE[$p]=$(curl -sfL "https://huggingface.co/api/models/mradermacher/qqWen-72B-RL-GGUF?blobs=true" \
    | python3 -c "import json,sys;print(next(x['size'] for x in json.load(sys.stdin)['siblings'] if x['rfilename'].endswith('Q8_0.gguf.part${p}of2')))")
done
for p in 1 2; do
  f=$DIR/qqWen-72B-RL.Q8_0.gguf.part${p}of2
  for try in $(seq 1 30); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" = "${SIZE[$p]}" ] && break
    log "part $p, attempt $try ($have of ${SIZE[$p]} bytes)"
    curl -fL -C - --http1.1 --retry 5 --retry-delay 30 --speed-limit 1000000 --speed-time 120 \
      -o "$f" "$URL/qqWen-72B-RL.Q8_0.gguf.part${p}of2" || sleep 30
  done
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$have" = "${SIZE[$p]}" ] || { log "part $p incomplete ($have of ${SIZE[$p]})"; echo TEACHER_FAILED; exit 1; }
done
log "joining"
cat $DIR/qqWen-72B-RL.Q8_0.gguf.part1of2 > "$OUT.partial"
rm $DIR/qqWen-72B-RL.Q8_0.gguf.part1of2
cat $DIR/qqWen-72B-RL.Q8_0.gguf.part2of2 >> "$OUT.partial"
rm $DIR/qqWen-72B-RL.Q8_0.gguf.part2of2
want=$(( ${SIZE[1]} + ${SIZE[2]} ))
[ "$(stat -c %s "$OUT.partial")" = "$want" ] || { log "joined size wrong"; echo TEACHER_FAILED; exit 1; }
mv "$OUT.partial" "$OUT"
ls -la "$OUT"
echo TEACHER_READY

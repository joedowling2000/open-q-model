#!/bin/bash
# Measure decode throughput for one server configuration, so speed changes are
# decided by measurement rather than intuition (raising slots to 40 *looked*
# obviously right and was 30% slower).
#
#   scripts/bench.sh <label> [extra server flags...]
#
# Uses llama.cpp's native /completion with ignore_eos and a fixed n_predict, so
# every configuration generates exactly the same number of tokens: a chat
# request that stops early measures the model's brevity, not the hardware.
set -uo pipefail
LABEL="$1"; shift
BIN=/home/joedowling/Projects/serving/llama.cpp/build/bin/llama-server
ROOT=/home/joedowling/Projects/qeval
PORT=8123
NPAR=10
NPRED=192

"$BIN" -m "$ROOT/gguf/qwen3.5-27b-q8_0.gguf" \
  --host 127.0.0.1 --port $PORT --alias bench --reasoning off \
  -ngl 999 -c 20480 -np $NPAR --cont-batching --no-webui "$@" \
  > "$ROOT/logs/bench-$LABEL.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

tries=0
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  sleep 5; tries=$((tries+1))
  [ $tries -gt 90 ] && { echo "BENCH $LABEL: server never came up"; exit 1; }
done

python3 - "$PORT" "$LABEL" "$NPAR" "$NPRED" <<'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

port, label, npar, npred = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
prompt = ("Complete this q function, then explain how it works.\n\n"
          "// Count the vowels in a string\ncount_vowels:{[x]\n  / function body\n  }")

def one(_):
    body = json.dumps({"prompt": prompt, "n_predict": npred,
                       "ignore_eos": True, "temperature": 0.8}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))

t0 = time.time()
with ThreadPoolExecutor(max_workers=npar) as ex:
    outs = list(ex.map(one, range(npar)))
secs = time.time() - t0
tok = sum(o.get("tokens_predicted", npred) for o in outs)
accept = [o.get("timings", {}).get("draft_acceptance_rate") for o in outs]
accept = [a for a in accept if a is not None]
extra = f", draft acceptance {sum(accept)/len(accept):.2f}" if accept else ""
print(f"BENCH {label}: {tok} tokens / {secs:.1f}s = {tok/secs:.1f} tok/s aggregate "
      f"({tok/secs/npar:.2f} per slot){extra}", flush=True)
PY

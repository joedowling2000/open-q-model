#!/bin/bash
# Amendment 11: one-shot screen of the coder bases, then the fixed decision rule.
# Waits for round 4 (GPU) and for the downloads. Same protocol as every score,
# except 10 samples (a screen), served with 30 slots for the throughput figure.
set -uo pipefail
cd /home/joedowling/Projects/qeval
log() { echo "[$(date +%H:%M:%S)] $*"; }
while systemctl --user list-units 'run-qround4*' --no-legend | grep -q running; do sleep 300; done
until grep -qE "FETCH_DONE|FETCH_FAILED" logs/fetch-coders.log 2>/dev/null; do sleep 300; done
grep -q FETCH_DONE logs/fetch-coders.log || { log "downloads failed"; exit 1; }

for spec in "qwen3-coder-30b-a3b:gguf/qwen3-coder-30b-a3b-q8_0.gguf" \
            "qwen3-coder-next:gguf/qwen3-coder-next-q8_0-00001-of-00004.gguf"; do
  NAME=${spec%%:*}; GG=$PWD/${spec#*:}
  log "screening $NAME"
  BASE_GGUF=$GG SLOTS=30 bash scripts/train/eval_checkpoint.sh none "$NAME" 10 >> logs/screen-$NAME.log 2>&1 \
    || log "$NAME scoring failed"
  python3 - "$NAME" <<'PY' | tee -a logs/screen-summary.log
import json, re, sys, datetime as dt
name = sys.argv[1]
d = json.load(open(f"results/{name}-scored.json"))
log = open(f"logs/screen-{name}.log").read()
t = re.findall(r"\[(\d\d:\d\d:\d\d)\] (generating|grading)", log)
start = dt.datetime.strptime([a for a, b in t if b == "generating"][-1], "%H:%M:%S")
end = dt.datetime.strptime([a for a, b in t if b == "grading"][-1], "%H:%M:%S")
secs = (end - start).seconds or 1
toks = sum(int(x) for x in re.findall(r"n_decoded = +(\d+)", open(f"logs/serve-{name}.log").read()))
print(json.dumps({"model": name, "pass_at_1": round(d["pass_at_1"], 4),
                  "throughput_tok_s": round(toks / secs, 1), "generated_tokens": toks, "seconds": secs}))
PY
done

log "decision (amendment 11): pass@1 >= 0.15 and throughput >= 3x the 27B's (~30 tok/s at 30 slots)"
python3 - <<'PY' | tee -a logs/screen-summary.log
import json
rows = [json.loads(l) for l in open("logs/screen-summary.log") if l.startswith("{")]
for r in rows:
    r["qualifies"] = r["pass_at_1"] >= 0.15 and r["throughput_tok_s"] >= 90
    print(r)
q = sorted([r for r in rows if r["qualifies"]], key=lambda r: -r["pass_at_1"])
print("SCREEN_RESULT", q[0]["model"] if q else "none qualifies: RL starts from round 4 on the 27B")
PY

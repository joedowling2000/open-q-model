#!/bin/bash
# Amendment 15: final selection on both held-out sets, then the single Q-HumanEval
# score of the chosen model, and the same-task judge's positive control.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
QPY=/home/joedowling/venvs/qeval/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
E="taskset -c 0-4,10-14"
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
tag() { echo "$1" | sed 's|out/||; s|/|-|g'; }
serve() {
  [ -f "$1/adapter.gguf" ] || $PY scripts/train/convert_lora.py "$1" --base out/merged/qwen3.5-27b-r4 \
    --outfile "$1/adapter.gguf" --outtype f16 > logs/final-convert.log 2>&1 || die "convert failed $1"
  $LC/build/bin/llama-server -m gguf/qwen3.5-27b-r4-q8_0.gguf --lora "$1/adapter.gguf" --host 127.0.0.1 \
    --port 8109 --alias policy --reasoning off -ngl 999 -t 6 -tb 6 -c $((2048 * 32)) -np 32 --cont-batching \
    --no-webui > logs/final-serve.log 2>&1 &
  S=$!
  for i in $(seq 180); do curl -sf http://127.0.0.1:8109/health > /dev/null && return 0; sleep 10; done
  die "server never came up for $1"
}
score() {  # adapter: style held-out always, clean held-out if missing
  local t=$(tag $1); serve "$1"
  $E $QPY scripts/rl/eval_style.py --url http://127.0.0.1:8109/v1 --model policy \
    --out results/style-$t.json > logs/final-style-$t.log 2>&1 || log "style eval failed $1"
  if [ ! -f "results/heldout-$t.json" ]; then
    $E $QPY scripts/train/eval_heldout.py --url http://127.0.0.1:8109/v1 --model policy --n 10 \
      --out results/heldout-$t.json > logs/final-heldout-$t.log 2>&1 || log "held-out failed $1"
  fi
  kill $S; wait $S 2>/dev/null; sleep 10
  log "scored $1: $(grep -h STYLE_RESULT logs/final-style-$t.log | cut -c1-120)"
}

while systemctl --user list-units 'run-qrlcont*' --no-legend | grep -q .; do sleep 600; done
LAST=$(cat out/rl2/LAST_GOOD)
log "RL finished; last good $LAST"

# Round 26 drew from the pool before the style set was frozen: drop any it drew.
python3 - <<'PY'
import json
s = json.load(open("corpus/heldout_style.json"))
drawn = {json.loads(l)["id"] for l in open("corpus/rl2/rollouts_r026.jsonl")} if __import__("os").path.exists("corpus/rl2/rollouts_r026.jsonl") else set()
later = set()
for r in range(27, 31):
    p = f"corpus/rl2/rollouts_r{r:03d}.jsonl"
    if __import__("os").path.exists(p):
        later |= {json.loads(l)["id"] for l in open(p)}
bad = sorted(set(s["ids"]) & (drawn | later))
s["dropped_drawn_in_rl"] = bad
s["ids"] = sorted(set(s["ids"]) - set(bad)); s["n"] = len(s["ids"])
json.dump(s, open("corpus/heldout_style.json", "w"), indent=1)
print(f"style held-out: {len(bad)} drawn in round 26+ and dropped; {s['n']} remain")
PY

# Results files are named by tag: out/rl/r011 -> rl-r011 (matches existing heldout-rl-r011.json).
CANDS="out/rl/r011 out/rl2/r005 out/rl2/r008 out/rl2/r020 out/rl2/r025 $LAST"
for A in $(echo $CANDS | tr ' ' '\n' | awk '!seen[$0]++'); do score $A; done

choose() {  # print "adapter combined clean style" sorted by combined, best first
  python3 - "$@" <<'PY'
import json, sys
clean_ids = json.load(open("corpus/heldout_v5_clean.json"))["ids"]
rows = []
for a in sys.argv[1:]:
    t = a.replace("out/", "").replace("/", "-")
    try:
        h = {r["id"]: r["correct"] / r["n"] for r in json.load(open(f"results/heldout-{t}.json"))["rows"]}
        st = json.load(open(f"results/style-{t}.json"))["summary"]["pass_at_1"]
    except FileNotFoundError:
        continue
    c = sum(h[i] for i in clean_ids) / len(clean_ids)
    rows.append(((c + st) / 2, c, st, a))
for comb, c, st, a in sorted(rows, reverse=True):
    print(f"{a} {comb:.4f} {c:.4f} {st:.4f}")
PY
}
choose $(echo $CANDS) | tee results/final_candidates.txt
TOP3=$(head -3 results/final_candidates.txt | awk '{print $1}')
$PY scripts/rl/average_adapters.py --out out/rl_avg/top3 $TOP3 || die "averaging failed"
score out/rl_avg/top3
choose $(echo $CANDS) out/rl_avg/top3 | tee results/final_candidates.txt
BEST=$(head -1 results/final_candidates.txt | awk '{print $1}')
log "FINAL CHOICE (amendment 15): $BEST"
echo "$BEST" > results/final_choice.txt

log "merging $BEST into round 4 and scoring Q-HumanEval once"
$PY scripts/train/merge_into_checkpoint.py --base out/merged/qwen3.5-27b-r4 --out out/merged/qwen3.5-27b-final \
  $BEST > logs/merge-final.log 2>&1 || die "merge failed"
$PY $LC/convert_hf_to_gguf.py out/merged/qwen3.5-27b-final --outtype q8_0 \
  --outfile gguf/qwen3.5-27b-final-q8_0.gguf > logs/convert-final.log 2>&1 || die "convert failed"
BASE_GGUF=$PWD/gguf/qwen3.5-27b-final-q8_0.gguf PARITY_DATA=data/sft_r4 \
  bash scripts/train/eval_checkpoint.sh none qwen3.5-27b-final 30 >> logs/eval-final.log 2>&1 || die "EVAL_FAILED"
grep GATE_A_RESULT logs/eval-final.log | tail -1

log "same-task judge positive control (amendment 14 i)"
bash scripts/serve.sh qwen3.5-27b 8101 8 > logs/serve-judge-control.log 2>&1 &
S=$!
for i in $(seq 180); do curl -sf http://127.0.0.1:8101/health > /dev/null && break; sleep 10; done
$QPY -W ignore - <<'PY' | tee results/judge_positive_control.txt
import json, random, sys
sys.path.insert(0, "scripts/synth")
import gen_problems as g
from mbpp_reference import overview
he = [json.loads(l) for l in open("q-evaluation-harness/datasets/q_humaneval.jsonl")]
rng = random.Random(0); pos = rng.sample(he, 15)
hits = sum(g.same_task_as_benchmark(overview(h["prompt"]), "http://127.0.0.1:8101/v1", "qwen3.5-27b") for h in pos)
kept = [json.loads(l)["description"] for l in open("corpus/hstyle_problems.jsonl") if json.loads(l)["ok"]]
neg = rng.sample(kept, 15)
fp = sum(g.same_task_as_benchmark(d, "http://127.0.0.1:8101/v1", "qwen3.5-27b") for d in neg)
print(f"JUDGE_CONTROL: benchmark problems flagged as same task {hits}/15 (want 15); kept generated problems flagged {fp}/15")
PY
kill $S; wait $S 2>/dev/null
log "FINAL_DONE"

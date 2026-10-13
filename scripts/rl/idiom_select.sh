#!/bin/bash
# Amendment 16 selection, then the single Q-HumanEval score of the chosen
# idiom checkpoint. Waits for rl_idiom_chain.sh.
#   combined = (clean held-out + style held-out) / 2, as amendment 15
#   loop     = while/do share of all held-out and style samples (control 6's test)
# Eligible: combined >= r011's combined - 3.0 pts. Chosen: the eligible one with
# the lowest loop rate, if that is at least 20 pts below r011's. Otherwise
# IDIOM_NONE and nothing is scored.
set -uo pipefail
cd /home/joedowling/Projects/qeval
PY=/home/joedowling/venvs/jlens/bin/python
LC=/home/joedowling/Projects/serving/llama.cpp
export QHOME=$HOME/.kx QLIC=$HOME/.kx
log() { echo "[$(date +%H:%M:%S)] $*"; }
die() { log "$*"; exit 1; }
while systemctl --user list-units 'run-qrl3*' --no-legend | grep -q .; do sleep 600; done
LAST=$(cat out/rl3/LAST_GOOD 2>/dev/null || echo none)
CHOICE=$(python3 - out/rl3/r005 out/rl3/r010 out/rl3/r015 "$LAST" <<'PY' | tee results/idiom_candidates.txt | tail -1
import json, re, sys
L = re.compile(r"\b(while|do)\[")
clean = json.load(open("corpus/heldout_v5_clean.json"))["ids"]
def stats(tag, style_file):
    h = json.load(open(f"results/heldout-{tag}.json"))["rows"]; st = json.load(open(style_file))
    by = {r["id"]: r["correct"] / r["n"] for r in h}
    c = sum(by[i] for i in clean) / len(clean)
    s = [x for r in h for x in r["samples"]] + [x for r in st["rows"] for x in r["samples"]]
    return (c + st["summary"]["pass_at_1"]) / 2, c, st["summary"]["pass_at_1"], sum(bool(L.search(x)) for x in s) / len(s)
b = stats("rl-r011", "results/style-rl-r011-loops.json")
print(f"out/rl/r011 (baseline) combined {b[0]:.4f} clean {b[1]:.4f} style {b[2]:.4f} loop {b[3]:.3f}")
best = None
for a in dict.fromkeys(sys.argv[1:]):
    t = a.replace("out/", "").replace("/", "-")
    try:
        m = stats(t, f"results/style-{t}.json")
    except (FileNotFoundError, KeyError):
        continue
    ok = m[0] >= b[0] - 0.03
    print(f"{a} combined {m[0]:.4f} clean {m[1]:.4f} style {m[2]:.4f} loop {m[3]:.3f} {'eligible' if ok else 'below bar'}")
    if ok and (best is None or m[3] < best[1]):
        best = (a, m[3])
print(best[0] if best and best[1] <= b[3] - 0.20 else "IDIOM_NONE")
PY
)
cat results/idiom_candidates.txt
[ "$CHOICE" = "IDIOM_NONE" ] && { log "IDIOM_NONE: no checkpoint met the amendment 16 rule; the final model stands"; exit 0; }
log "IDIOM CHOICE (amendment 16): $CHOICE; merging into round 4 and scoring Q-HumanEval once"
$PY scripts/train/merge_into_checkpoint.py --base out/merged/qwen3.5-27b-r4 --out out/merged/qwen3.5-27b-idiom \
  $CHOICE > logs/merge-idiom.log 2>&1 || die "merge failed"
$PY $LC/convert_hf_to_gguf.py out/merged/qwen3.5-27b-idiom --outtype q8_0 \
  --outfile gguf/qwen3.5-27b-idiom-q8_0.gguf > logs/convert-idiom.log 2>&1 || die "convert failed"
BASE_GGUF=$PWD/gguf/qwen3.5-27b-idiom-q8_0.gguf PARITY_DATA=data/sft_r4 \
  bash scripts/train/eval_checkpoint.sh none qwen3.5-27b-idiom 30 >> logs/eval-idiom.log 2>&1 || die "EVAL_FAILED"
grep GATE_A_RESULT logs/eval-idiom.log | tail -1
python3 - <<'PY' | tee results/idiom_vs_final.txt
import json, random, re, statistics
L = re.compile(r"\b(while|do)\[")
per = lambda p: {r["task_id"]: r["num_correct"] / r["num_samples"] for r in json.load(open(p))["per_problem_results"]}
a, b = per("results/qwen3.5-27b-idiom-scored.json"), per("results/qwen3.5-27b-final-scored.json")
k = sorted(a); rng = random.Random(0); m = lambda v: sum(v) / len(v)
ci = sorted(m([a[rng.choice(k)] for _ in k]) for _ in range(10000))
d = sorted(m([a[t] - b[t] for t in (rng.choice(k) for _ in k)]) for _ in range(10000))
for name in ("idiom", "final"):
    c = [json.loads(l)["completion"] for l in open(f"results/qwen3.5-27b-{name}.jsonl")]
    print(f"{name}: loop rate {100 * sum(bool(L.search(x)) for x in c) / len(c):.1f}%, median length {statistics.median(map(len, c)):.0f} chars")
pa, diff = 100 * m(list(a.values())), 100 * m([a[t] - b[t] for t in k])
print(f"idiom pass@1 {pa:.2f} [{100*ci[250]:.1f}, {100*ci[9750]:.1f}]; paired vs final {diff:+.2f} [{100*d[250]:+.1f}, {100*d[9750]:+.1f}]")
PY
log "IDIOM_DONE"

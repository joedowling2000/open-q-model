# Building a better open q/kdb+ model — plan and status

**Location:** `~/Projects/qeval` (local, ~95 GB). Nothing here is in a sandbox.
**Started:** 2026-09-16 · **Hardware:** DGX Spark (GB10, 121 GB unified, ~273 GB/s)

The question: **Morgan Stanley's qqWen is the only serious open q model, built on a
2024 base from 1.6M tokens. Can we do better?**

---

## Status at a glance

| Phase | State |
|---|---|
| 0. Measure the starting line | ✅ **done, 30 samples, validated** |
| 1. Assemble a corpus | ✅ measured: ~17M tokens usable |
| 2. Synthetic data pipeline | 🔄 rebuilding as distillation from qqWen — see [GENERATION.md](GENERATION.md) and amendment 2 |
| 3. Continued pretraining | ⬜ not started |
| 4. SFT | ⬜ not started |
| 5. RL with execution rewards | ⬜ not started |
| 6. Agent-mode evaluation | ⬜ not started (needs a new harness backend) |

---

## What we measured (Q-HumanEval, current grader, Q8_0, one-shot, no tools)

| Model | Pass@1 | Pass@5 | Pass@10 |
|---|---|---|---|
| **qqWen-32B-RL** (q-specialised, Qwen-2.5 base) | **38.4%** | 54.2% | 59.4% |
| Qwen3.5-27B (general, 2026) | 10.2% | 18.6% | 23.6% |
| Gemma-4-31B-it (general, 2026) | 10.1% | 15.0% | 18.0% |

30 samples per problem, validated. Each model measured twice independently
(10 then a fresh 20); the passes agree within a point.

Reference, same grader: Claude Opus 4.8 one-shot **53.2%**; Opus 5 in agent mode
**97.6%** (not comparable — iterates against a live interpreter).

**These appear to be the first published open-weight scores on Q-HumanEval under
the post-June-2026 grader.**

**What it settles:** two generations of general-model progress bought ~nothing for
q. Specialisation is worth 3.7x and is the whole story. The target to beat is
38.4%, and the gap to frontier one-shot is 15 points, not 87.

---

## What the data actually looks like

| Source | Tokens | Licence |
|---|---|---|
| GitHub q source (harvested, filtered) | 5.2M | MIT/Apache/BSD/ISC/CC0 only |
| Kx documentation | 1.35M | CC-BY-4.0 |
| MS `sft-python-q-problems` (678 problems, tests) | 10.4M | Apache-2.0 — *excluded from training by the pre-registration; seeds only* |
| MS `q_pretrained_dataset` (their corpus) | 1.56M | Apache-2.0 |
| rd200/kdb_q | 0.07M | Apache-2.0 |
| **Usable total** (before overlap) | **~17M** | vs qqWen's 1.6M |
| **Permitted for training under the pre-registration** | **~8M** | human-written only, plus self-generated |

Harvest path: 1,398 public q repos → **331 permissively licensed** (71% carry *no
licence*) → 4,550 unique files → 4,099 after filtering out impostors (a JSON maths
dataset and Chinese QuickMacro scripts both use `.q`/`.Q`).

**Conclusion: public q is small and mostly unlicensed.** Harvesting gives ~4x
qqWen's corpus and then stops. Beyond that, data has to be *generated*.

---

## Three findings that shape the plan

1. **"Verified" data is only as good as its tests.** MS's LeetCode 1013 solution
   returns false whenever the array sum is non-zero — wrong — and passes all five
   of its test cases because each uses a zero-sum array. Rejection sampling
   against thin tests bakes in special-casing.
2. **Blind fuzzing cannot audit that.** LeetCode problems carry cross-argument
   constraints (`0 <= index < n`); random inputs violate them, so most apparent
   disagreements are invalid inputs. Problems must ship their own input generator.
3. **Verification is only as strong as the input distribution.** Our own pipeline
   first *accepted* the known-wrong solution: random arrays almost never partition
   into three equal parts, so "always false" agreed with the reference on all 30
   cases. Fixed with a discriminating-power check — reject a problem whose
   reference answers barely vary, because such inputs cannot separate right from
   wrong.

---

## The plan

### Phase 3 — continued pretraining (next, needs GPU)
LoRA on **Qwen3.5-27B** over the ~8M-token human-written corpus (harvested q, Kx
docs, MS's harvested corpus, rd200).
Iterate at 4–9B, report at 27B. Hand-written LoRA rather than `peft` if the
educational angle matters (see the gate note below).

**Gate:** perplexity on held-out q improves, and Q-HumanEval pass@1 beats the
10.4% base. If continued pretraining alone gets nowhere near qqWen's 37.9%, the
gain is in SFT/RL, not data volume — proceed anyway but reweight effort.

### Phase 4 — SFT
**Self-generated data only**, per the pre-registration: problems and q solutions
written by the base model, kept only where they agree with their Python reference
across >=30 discriminating inputs.

Morgan Stanley's *q solutions* are excluded — they are another pipeline's model
output, and training on them would make this distillation rather than
self-improvement. Their problem *descriptions* may be used as generation seeds,
disclosed. Their `q_pretrained_dataset` stays in Phase 3 because it is
human-written code, not model output.

**Gate B:** pass@1 above the base, CI on the difference excluding zero.

### Phase 5 — RL with execution rewards
GRPO as qqWen used, with KDB-X as the reward oracle (free, offline, commercial
use permitted; 16 GB RAM, 4 secondary threads). Feasible at 4–9B; slow at 27B.

**Gate:** beats 38.4% pass@1. That is the headline claim.

### Phase 6 — agent mode
The harness only supports Claude Code and Codex CLIs. An OpenAI-compatible agent
backend would let open models be scored in agent mode, where frontier models gain
+34 points. Worth upstreaming to KX.

---

## Pre-registration

The claim, thresholds and data-lineage rule are fixed in
[PREREGISTRATION.md](PREREGISTRATION.md), written 2026-09-18 before any training
run. Headline: **can a general model teach itself q from its own
execution-verified output?** Baseline 10.4%, target 37.9%, primary metric pass@1
at 30 samples. qqWen's output is forbidden in training — using it would turn the
claim from self-improvement into distillation.

## Rules we hold ourselves to

1. **Q-HumanEval is evaluation only.** Its 164 problems never enter training, and
   a text-overlap check runs before any number is reported. The pipeline already
   enforces this (it dropped a fixture for overlapping a benchmark problem).
2. **Same grader, quantisation and sample count** across compared models.
3. **Licences stay clean:** MIT/Apache/BSD/ISC/CC0/CC-BY only; no GPL-3 q corpora;
   **no distillation from frontier APIs** whose terms forbid training competitors.
   Generate with open weights we run ourselves.
4. **Measure, don't assume.** Three "obvious" speedups were tried and all lost:
   40 slots (-30%), n-gram speculation (-11%), a 0.8B draft model (-48%).

---

## How to run it

```bash
# evaluation
scripts/prepare_model.sh <hf-repo> <name>     # HF -> Q8_0 GGUF
scripts/serve.sh <name> [port] [slots]        # llama.cpp, OpenAI-compatible
scripts/run_all.sh                            # all three models, 10 samples
scripts/then_30.sh                            # validate, top up to 30, merge
scripts/validate.py 10 <names...>             # trust check on a finished run

# corpus
scripts/corpus/enumerate_repos.py             # GitHub survey with licences
scripts/corpus/harvest.py                     # clone permissive repos, extract q
scripts/corpus/filter.py                      # drop non-q impostors
scripts/corpus/measure.py                     # tokens, licences, concentration

# synthetic data
scripts/synth/pipeline.py --fixtures          # CPU: prove verification stages
scripts/synth/pipeline.py --n-problems 50     # GPU: full generation
scripts/synth/fuzz_verify.py --sample 100     # differential audit of MS data
```

**Serving note:** llama.cpp, not vLLM (which fights aarch64). 10 slots, Q8_0,
`--reasoning off`, 6 CPU threads. ~39 tok/s aggregate for a 27B — bandwidth-bound,
and no arrangement of slots or speculation beat it.

**Thermals:** launch long runs through `runlog` so `thermal-guard` can freeze them.
Guard at 74/64 °C. Spikes to ~88 °C come from CPU core clusters, not the GPU, and
the September crashes happened near 94 °C.

---

## Open questions

- **Does the educational gate apply?** By the `~/Projects/ideas` rule this is
  engineering, not maths-by-hand or LLM internals. Angles that would qualify:
  implement LoRA and GRPO by hand rather than importing them; study what q's
  symbol-dense syntax does to tokenisation (measurable, and directly relevant).
- **Is 37.9% beatable with ~17M tokens?** Unknown. qqWen reached it from 1.6M plus
  SFT and RL, so the pipeline matters more than the corpus.
- **How much does quantisation cost?** Everything here is Q8_0. A bf16 check on
  one model would calibrate against leaderboard numbers.
- **Was the MS SFT data ever audited for correctness?** Our attempt found one
  likely genuine disagreement (LeetCode 1989) but could not clear the rest,
  because valid inputs need per-problem generators.

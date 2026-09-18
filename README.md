# Q evaluation — where do open weights actually stand on q/kdb+?

> **See [PLAN.md](PLAN.md)** for the project plan, status and decisions.
> This file covers the evaluation harness setup and its results.

Step 1 of the "could we build a better open q model?" question: **measure the
starting line**. Nobody has published Q-HumanEval scores for 2026 open-weight
models under the current grader, so before training anything we score three:

| Model | Why it's here | Size (bf16) |
|---|---|---|
| `morganstanley/qqWen-32B-RL-Reasoning` | the incumbent specialist (Qwen-2.5 base, Aug 2025) | 131 GB on disk (stored F32) |
| `Qwen/Qwen3.5-27B` | a modern general base, two generations newer | 56 GB |
| `google/gemma-4-31B-it` | modern general base, different family, Apache 2.0 | 63 GB |

**The question this answers:** does a 2026 general model already match a 2025
q-specialised one? If yes, redoing the specialisation on a modern base is clearly
worth it. If no, qqWen's pipeline is carrying real weight and the gain has to come
from data volume instead.

## What KX has published

- [q-evaluation-harness](https://github.com/KxSystems/q-evaluation-harness) (MIT) —
  Q-HumanEval, 164 hand-written problems, the benchmark used here.
- [kx-skills](https://github.com/KxSystems/kx-skills) (Apache 2.0) — Claude Code
  plugins that teach an agent q. KX has **not** released a model.

Reference points from their leaderboard:

| Mode | Model | Pass@1 |
|---|---|---|
| Agent (new grader) | Claude Opus 5, clean room | 97.6% |
| One-shot (new grader) | Claude Opus 4.8 | 53.2% |
| One-shot (**old grader**) | qqWen 72B | 45.1% |
| One-shot (**old grader**) | Qwen3-Coder-30B-A3B | 8.3% |
| One-shot (**old grader**) | Llama-3.3-70B | 10.1% |

⚠️ **Grader change, June 2026** ([PR #7](https://github.com/KxSystems/q-evaluation-harness/pull/7)):
scores moved +3 to +13 points and **old numbers are not comparable to new ones**.
Everything we run uses the current grader, so qqWen must be re-scored here rather
than compared against its 45.1%.

## Setup

```bash
~/venvs/qeval          # harness deps (pykx, litellm, torch, transformers)
~/Projects/qeval/gguf  # Q8_0 GGUFs for llama.cpp
~/Projects/qeval/results
```

**Blocked on a licence key.** PyKX executes the generated q, and needs a kdb+
licence. Get the free **KDB-X Community Edition** key (free for personal *and*
commercial use, 16 GB RAM, one instance, 4 secondary threads, 16 connections):

1. Sign up at <https://developer.kx.com/products/kdb-x/install>
2. The welcome email carries a base64 `kc.lic`
3. Save it and point PyKX at it, e.g. `export QLIC=$HOME/.kx`

Generation works without it; only grading needs q.

## Running

```bash
scripts/prepare_model.sh <hf-repo> <name>   # HF -> Q8_0 GGUF
scripts/serve.sh <name> [port] [slots]      # llama.cpp OpenAI-compatible server
scripts/run_eval.sh <name> [samples] [port] # Q-HumanEval via the litellm backend
```

Serving through llama.cpp rather than the harness's own vLLM path is deliberate:
vLLM on this aarch64 box is a fight, and llama.cpp is already built here with
`qwen35`, `gemma4` and `qwen2` support. Every model is quantised the same way
(**Q8_0**, near-lossless) so the comparison stays internally fair — but note it
is *not* strictly comparable to a bf16 leaderboard submission.

**Sample count.** The leaderboard standard is 50 samples per problem (temp 0.8,
seed 1234). A first pass at 10 is enough to rank three models; re-run the
survivors at 50 before quoting a number anywhere public.

**Thermals.** Long generation runs are exactly what crashed this box three times
in September. Launch evals through `runlog` so `thermal-guard` can freeze them.

## Patches to the cloned harness

Four bugs, all hit on first run. The first three are fixed in
`q-evaluation-harness/` here and are worth upstreaming; the fourth is recorded
but untestable until the licence lands.

1. **`src/models/factory.py`** imported vLLM at module scope, so the API and
   HuggingFace backends were unusable anywhere vLLM will not install (this box
   is aarch64). Now imported inside the `vllm` branch.
2. **`src/utils/hardware_profiler.py`** — same top-level vLLM import, same fix.
3. **`src/evaluation/q_python_executor.py`** ran `del os.environ['QHOME']`
   unconditionally, raising `KeyError` at import and taking down the whole CLI
   — including `generate`, which never touches q. Now `os.environ.pop(..., None)`.
4. **Not fixed — aarch64 q path.** The same file maps any non-x86_64 Linux to
   `l32`: `arch = "l64" if platform.machine() in ("x86_64", "amd64") else "l32"`.
   On ARM64 the kdb+ directory is `l64arm`, so grading will look in the wrong
   place here. Fix once q is installed and the path can be confirmed.

### Three traps in driving it (ours, not the harness's)

1. **Run from inside the repo.** Datasets resolve relative to the working
   directory (`./datasets/q_humaneval.jsonl`), so `generate` and `execute` must
   be invoked with the harness as cwd. Anywhere else fails instantly.
2. **Turn reasoning off.** Qwen3.5's chat template enables thinking by default.
   With `DEFAULT_MAX_TOKENS = 512` the model spends the budget on `<think>` and
   emits no code, scoring ~0 for reasons that have nothing to do with q. The
   one-shot protocol is "no extended thinking", so `serve.sh` passes
   `--reasoning off`.
3. **Slots ≥ samples.** llama.cpp refuses `n` completions when `n > -np`
   ("n_cmpl cannot be greater than the number of slots"). Every request 400s,
   the harness records empty solutions, and the run *looks* like it worked —
   at an impossible 4.6 problems/second with an idle GPU. Watch GPU power as
   the tell.

Smoke test, 2026-09-16: `gemma-3-4b-it` at Q8_0, one sample per problem, all
164 generated in 6m24s through llama.cpp with 4 slots. Output was Python-shaped
pseudo-q, which matches its 3.02% on the historical board and confirms the
plumbing rather than the model.

## Rules we hold ourselves to

1. **Q-HumanEval is evaluation only.** If this project ever trains a model, those
   164 problems never enter training data, and a text-overlap check against them
   runs before any number is reported.
2. **Same grader, same quantisation, same sample count** across compared models,
   or the comparison is void.
3. **Licences stay clean**: MIT/Apache/CC-BY sources only, no GPL-3 q corpora, no
   distillation from frontier APIs whose terms forbid training competitors.

## Results — 10 samples per problem, current grader, Q8_0, no tool use

| Model | Base | Pass@1 | Pass@5 | Pass@10 | distinct/1640 |
|---|---|---|---|---|---|
| **qqWen-32B-RL-Reasoning** | Qwen-2.5-32B, q-specialised | **37.9%** | 53.9% | 58.5% | 1605 |
| Qwen3.5-27B | general, 2026 | 10.4% | 19.2% | 25.6% | 1640 |
| Gemma-4-31B-it | general, 2026 | 10.2% | 15.1% | 17.7% | 975 |

All three validated: 100% non-empty, zero infrastructure errors.

Reference points from KX's leaderboard (same grader): Claude Opus 4.8 one-shot
**53.2%**; Claude Opus 5 in agent mode **97.6%**. Not directly comparable to the
rows above — agent mode iterates against a live interpreter.

**What this settles.** Two generations of general-model progress bought ~nothing
for q: a 2026 27B and a 2026 31B both sit at ~10%, while a 2025 q-specialised
32B scores 37.9%. Specialisation is worth ~3.7x, and it is the whole story.

That is the case for the project: qqWen's pipeline applied to a *modern* base has
room to beat 37.9%, and the gap to frontier one-shot (53.2%) is 15 points rather
than the 87 points that agent mode implies.

Caveats: 10 samples, not the leaderboard's 50 (a 30-sample pass is running);
Q8_0 quantisation throughout, so these are not bf16 numbers; and Gemma's low
distinct count (975) shows it repeats itself at temperature 0.8, which depresses
its pass@10 more than its pass@1.

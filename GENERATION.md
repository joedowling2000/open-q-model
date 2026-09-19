# How to make an open-weight model generate a q training set

What you have to specify, why each part is there, and what breaks without it.
Written after a failed attempt: Qwen3.5-27B generating its own q produced
**3 usable problems in 3.5 hours**, because a model that scores 10% on q writes
things like `solve:{[strings] \`json!(\`:flip 1=; flip enlist "a"; strings)}` —
noise shaped like q. Two lessons drove the redesign:

1. **The teacher must already be good at the target language.** Verification
   filters bad output; it cannot create good output. A 10% model gives a ~10%
   yield at best, and in practice worse, because generated problems are harder
   than benchmark problems.
2. **Separate the two skills.** Writing a good *problem* (clear statement,
   correct reference implementation, valid input generator) is general reasoning.
   Writing good *q* is domain skill. They can come from different models.

---

## The pipeline in one line

**A strong general model writes the problem; the best available q model writes
the solution; a real q interpreter decides what survives.**

```
problem author  ->  q author (n attempts)  ->  execution verification  ->  dataset
 (general model)     (best q model)            (KDB-X + Python oracle)
```

---

## 1. Choosing the models

| Role | Requirement | Our choice | Why |
|---|---|---|---|
| Problem author | strong general reasoning, writes valid Python | Qwen3.5-27B | measured, local, Apache-2.0 |
| q author | **best measured q ability** | qqWen-32B-RL (38.4% pass@1) | 3.8x better at q than any general model we measured |
| Oracle | executes both languages | KDB-X + CPython | free, offline, commercial use permitted |

**Licence matters as much as capability.** Everything above is Apache-2.0, so
the generated dataset can be published. Frontier APIs generally forbid using
outputs to train competing models, which would make the dataset unreleasable —
capability you cannot use is not capability.

---

## 2. What the problem-author prompt must specify

The prompt asks for **three artefacts**, not one. Asking only for "a problem"
produces something unverifiable.

```
```description   one paragraph, precise, with explicit constraints
```python        def solve(...) — the reference implementation
```generator     def gen_input(rng) — returns VALID kwargs for solve()
```

**Why the generator is non-negotiable.** Without it you must invent inputs, and
invented inputs violate the problem's constraints — `n = -10`, an index of 18
into a 2-element array, edges naming node 5 of a 3-node graph. Python then
raises, q returns something, and you record a "disagreement" that is really your
own bad input. With a generator, valid inputs exist by construction.

**Rules the prompt must state explicitly:**

- `solve()` is deterministic, returns a JSON-serialisable value, runs in well
  under a second.
- `gen_input(rng)` never produces inputs that violate the stated constraints,
  and uses `rng` (a seeded `random.Random`) so runs are reproducible.
- Standard library only — no pandas, no numpy. Imports you do not install are
  the most common reason a generated reference fails to run.
- Output is **exactly** the three fenced blocks, nothing else. Models pad with
  commentary; the parser must find fences, and unfenced replies are counted and
  discarded.

**Topic control.** Left alone, a model writes the same five problems. Cycle an
explicit topic list, and make it cover what q is actually *for*:

> vector arithmetic · string manipulation · sorting and ranking · time-series
> windows · table joins · grouping and aggregation · set operations · running
> totals and deltas · filtering with predicates · type conversion · dictionaries
> and keyed lookup · matrix and nested lists

**Difficulty control.** State the target difficulty in the prompt and vary it.
Only-easy problems teach nothing; only-hard problems yield nothing. A ladder
(easy / medium / hard in fixed proportions) keeps both ends populated.

---

## 3. What the q-author prompt must specify

```
Write a q/kdb+ function that solves this problem.
<description>
Reply with exactly one fenced q block defining `solve`, and nothing else.
solve:{[arg1;arg2] ... }
Use idiomatic q. Do not include tests or commentary.
```

- **Fix the signature.** Argument names and order come from the Python
  reference, so the verifier can call `solve[a;b]` positionally.
- **Demand idiomatic q.** Morgan Stanley's released dataset is full of `while`
  loops and index arithmetic — Python written in q. If you do not ask for vector
  style you will train that accent into the model. Measure it: report the rate
  of `while`/`do` in accepted solutions.
- **Sample n times per problem** (4–8). Solutions are cheap; problems are
  expensive. One correct attempt out of eight still yields a training record.
- **No tests in the output.** The model's own tests are worthless here; the
  verifier supplies them.

---

## 4. Verification — the part that makes the data trustworthy

Execution is what separates a dataset from a pile of plausible text.

**Differential testing.** Run the Python reference and the q candidate over the
same generated inputs and compare results. Not "does the q run" — *does it agree
with the reference*.

**Compare structurally, not textually.** q's `show` prints `1b` where Python
prints `True`, and `1 2 3` where Python prints `[1, 2, 3]`. Have q serialise its
answer with `.j.j` (JSON) and compare parsed values with a type-tolerant
equality (int/float within 1e-6, lists elementwise).

**Require discriminating inputs.** This is the subtle one. Our first pipeline
*accepted a known-wrong solution*: for "can this array be split into three equal
parts", random arrays almost never can, so a solution that always answers "no"
agreed with the reference on all 30 cases. **Verification is only as strong as
the input distribution.** So: reject any problem whose reference gives the same
answer in more than 90% of cases — its inputs cannot separate right from wrong.
This is also why five fixed test cases (as in the MS dataset) is not enough.

**Skip what you cannot represent.** If an input contains a value with no
faithful q literal (`None` being the common case), skip the problem rather than
guess a mapping — a wrong guess makes a *correct* solution look wrong.

**Watch the q/Python type edges:**

| Python | q | Trap |
|---|---|---|
| `[0]` | `enlist 0` | `(0)` is the **atom** 0, not a list |
| `"a"` | `enlist "a"` | a 1-char string is a **char atom** |
| `[]` | `()` | fine |
| `{"a": 1}` | `` `a!1 `` | symbol keys vs string keys |
| `None` | — | no faithful mapping: skip |

---

## 5. Hygiene the dataset needs before anyone trains on it

- **Decontaminate against the benchmark**, both directions: problem statements
  *and* solutions, against Q-HumanEval's 164 problems and their references.
  Report how many were dropped.
- **Deduplicate** by normalised solution text; a model asked for 5,000 problems
  will repeat itself.
- **Hold out 10–15%** as an internal test set, never trained on, to detect a
  model that has learned the generator's style rather than q.
- **Record provenance per record**: which model wrote the problem, which wrote
  the solution, how many inputs it was verified against, which attempt number
  succeeded.
- **Keep the rejects.** They are the diagnostic when yield collapses, and they
  are how you tell "the model is bad at q" from "my verifier is too strict".

---

## 6. Operational lessons, paid for in GPU time

- **One bad record must not kill a batch.** A `TypeError` on a single generated
  input cost ~20 problems of work before it was caught.
- **Repair once before discarding.** A generator that references an undefined
  name is a weak-model artefact; feeding the error back for one retry recovers
  it cheaply.
- **Batch and append.** Long runs get interrupted; each batch should commit its
  survivors so nothing is lost.
- **Watch GPU power, not the progress bar.** A run where every request fails
  looks *fast* and produces a plausible zero.
- **Yield is the health metric.** Track kept/attempted per batch. If it collapses,
  stop and read the rejects rather than buying more hours.

---

## 7. What to expect

With a teacher that scores ~38% on q, sampling 4–8 solutions per problem, a
yield of roughly **30–60% of problems** is plausible — against the ~3% we got
from a 10% teacher. The expensive stage is generation, not verification: budget
on tokens produced, and remember that verification runs on CPU and can overlap
with the next batch's generation.

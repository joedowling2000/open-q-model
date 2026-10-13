---
license: apache-2.0
base_model: Qwen/Qwen3.5-27B
language:
- en
pipeline_tag: text-generation
library_name: transformers
tags:
- q
- kdb+
- code
- gguf
---

# open-q-27b

open-q-27b is Qwen3.5-27B fine-tuned to write q, the language of kdb+. On KX's
Q-HumanEval benchmark it passes 64.8% of problems on the first attempt. That
compares with 10.2% for the base model, 38.4% for Morgan Stanley's
qqWen-32B-RL, and 57.0% for Claude Opus 4.8. All four were measured on the same
machine with the same harness and grader.

I trained it at home on a single NVIDIA DGX Spark over about four weeks in
September and October 2026. Code, data manifests, logs and the full
pre-registration (including the things that went wrong) are at
https://github.com/joedowling2000/open-q-model.

## Results

Q-HumanEval, 164 problems, current grader (after KX's PR #7). Each model wrote
30 samples per problem: one-shot, no tools, no thinking, 512 tokens maximum.
Our model and the open models ran at temperature 0.8. Opus ran at 1.0, because
its API won't accept anything else. Ranges are 95% bootstrap intervals over
problems.

| Model | pass@1 | pass@10 |
|---|---|---|
| Qwen3.5-27B (base) | 10.2% | 23.6% |
| qqWen-32B-RL | 38.4% | 59.4% |
| Claude Opus 4.8 | 57.0% [51.0, 63.0] | 83.3% |
| **open-q-27b, Q8_0** | **64.8% [58.9, 70.8]** | **83.4%** |
| open-q-27b, bf16 | 65.1% [59.1, 71.1] | 83.3% |

Compared problem by problem, open-q-27b is ahead of Opus 4.8 by 7.9 points
(range +0.8 to +14.8) and ahead of qqWen-32B-RL by 26.4 (+20.2 to +32.7). Q8_0
scores the same as bf16 within noise, so the 8-bit file is the one most people
should use.

Two things the headline doesn't show:

- **Given ten tries, it and Opus are level** (both about 83%). The difference is
  in getting it right the first time.
- **It writes loops.** 78.5% of its answers use `while` or `do` where
  experienced q programmers would use a vector expression. The code is correct
  more often than not, but it reads like Python in q syntax. This came from
  reinforcement learning that rewarded passing tests and nothing else. A
  follow-up run that penalises loops is in progress.

On a separate set of 60 harder, business-style questions never used in
training, it scores 24.5% (round 4 before RL: 18.2%). Expect it to struggle with
longer, multi-step problems.

## Using it

Two GGUF files are included for llama.cpp: `open-q-27b-Q8_0.gguf` (29 GB) and
`open-q-27b-BF16.gguf` (55 GB). The safetensors weights are bf16 and load with
transformers.

```
llama-server -m open-q-27b-Q8_0.gguf --reasoning off -ngl 999 -c 8192
```

Turn thinking off. The model was trained and tested without it.

It does best with the prompt layout the benchmark uses: a q-doc comment with
`@overview`, `@param` and `@return` lines, then a function stub with `x`, `y`
and `z` as arguments.

```
You are an expert q/kdb+ programmer.
Write idiomatic q function to complete the following task:
// @overview
// Return the n largest values in a list, largest first.
//
// @param x {long[]} The list.
// @param y {long}   How many to return.
//
// @return {long[]}
top_n:{[x;y]
    / function body
    }
```

Plain-English requests work too, but less reliably. Settings used in testing:
temperature 0.8, top_p 0.95, at most 512 new tokens.

With transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("josephdowling10/open-q-27b")
model = AutoModelForCausalLM.from_pretrained("josephdowling10/open-q-27b",
                                             dtype="bfloat16", device_map="auto")
msgs = [{"role": "user", "content": prompt}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                 return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The checkpoint keeps Qwen3.5's vision encoder unchanged. I haven't tested
images and nothing here was trained on them.

## How it was trained

All training used LoRA adapters, merged into the weights after each stage.

1. **Continued pretraining** on about 8M tokens of human-written q: permissively
   licensed GitHub code, KX documentation, and two open q corpora.
2. **Supervised fine-tuning in four rounds** on problems with verified
   solutions. Every solution had to match a Python reference on 60 or more
   freshly generated inputs before it was used. Solutions came from three
   sources:
   - q written by two q developers I work with;
   - qqWen-72B-RL (distilled, then verified the same way);
   - the model's own verified answers.

   Where possible, loop solutions were rewritten into vector form and
   re-checked. The last round also trained on the benchmark's prompt layout
   (its public template, none of its problems).
3. **Reinforcement learning (GRPO).** The model attempted problems in groups of
   eight, scored by running the code. Partial credit was given for passing some
   inputs. The first run collapsed at round 14. The released checkpoint is
   round 11, chosen on held-out data before the benchmark was run.

The benchmark was scored once, on the chosen model. Every checkpoint choice used
held-out sets that never touched training. Training problems were checked
against all 164 benchmark problems for overlap, and look-alikes were removed.
The thresholds, decision rules and every change to them are written down, with
dates, in the repository's PREREGISTRATION.md.

## Data and licences

The weights are released under Apache-2.0, the same as Qwen3.5-27B.

| Source | Use | Licence |
|---|---|---|
| GitHub q repositories | pretraining | MIT, Apache-2.0, BSD, ISC or CC0 only |
| KX documentation | pretraining | CC-BY-4.0 |
| morganstanley/q_pretrained_dataset | pretraining | Apache-2.0 |
| rd200/kdb_q | pretraining | Apache-2.0 |
| Q study bank (colleagues' problems and solutions) | fine-tuning | used with permission |
| morganstanley/sft-python-q-problems | problem descriptions only, as seeds | Apache-2.0 |
| MBPP | problem descriptions only, as seeds | CC-BY-4.0 |
| qqWen-72B-RL outputs | fine-tuning, after verification | Apache-2.0 |
| Problems written by Qwen3.5-27B | fine-tuning and RL | Apache-2.0 |

Morgan Stanley's own q solutions and Python references were never trained on.
No output from Claude, GPT or any other closed model went into training. Opus
answers were used for the comparison above only.

## Limitations

- Loop-heavy style, as above. Check that the code is idiomatic before you
  rely on it.
- Tested only on short, single-function tasks. Nothing here measures queries
  against real tables, IPC or performance.
- It will produce wrong code with confidence. Run what it writes.
- Q-HumanEval has 164 problems, so differences of a few points are within noise.

## Thanks

To the two q developers whose solutions anchor the fine-tuning data, to KX for
the evaluation harness, to Morgan Stanley for releasing qqWen and its data, and
to the Qwen team for the base model.

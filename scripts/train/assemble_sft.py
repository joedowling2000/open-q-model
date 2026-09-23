#!/usr/bin/env python3
"""Turn verified problems into supervised fine-tuning data, several ways over.

A verified problem is more than one training example. The pipeline produces a
description, a Python reference, a q solution and an input generator, all of
which agree with each other by execution — and the reject log holds q that was
*proved wrong*, with the exact input that exposed it. That is enough to write
six tasks, and the expensive part (verification) is already paid for:

  desc2q      description -> q                 the benchmark task itself
  py2q        Python -> q                      translation, the qqWen skill
  q2py        q -> Python                      the same mapping, reversed; forces
                                               the model to *read* q, not emit it
  q2desc      q -> plain English                comprehension
  fix         broken q + failing input -> q    real bugs, not corrupted code
  vectorise   imperative q -> vector q         only where a rewrite verified

`fix` is the one worth arguing for. Synthetic bug injection teaches a model to
undo whatever corruption you applied. These bugs are genuine: models wrote them
while attempting the problem, and execution proved them wrong. They are also
q-shaped rather than Python-shaped — the first reject in the file is
`solve:{[a;b] a*b+1}`, which is wrong because q evaluates right to left, so it
computes `a*(b+1)`. No corruption script invents that.

Loss is masked to the response. Training on the prompt tokens as well teaches
the model to generate problem statements, which nothing asks it to do.

    python scripts/train/assemble_sft.py --in corpus/qdataset.audited.jsonl \
        --out data/sft
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/home/joedowling/Projects/qeval")
TOKENIZER = ("/home/joedowling/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/"
             "snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654")
IGNORE = -100
SHINGLE = 60


def qfence(code: str) -> str:
    return f"```q\n{code.strip()}\n```"


def pyfence(code: str) -> str:
    return f"```python\n{code.strip()}\n```"


# ------------------------------------------------------------------- tasks

def desc2q(rec: dict) -> tuple[str, str]:
    return (f"Write a q (kdb+) function `solve` for this problem.\n\n"
            f"{rec['description']}\n\n"
            f"Reply with one fenced q block.",
            qfence(rec["q_solution"]))


def py2q(rec: dict) -> tuple[str, str]:
    return (f"Translate this Python function into idiomatic q (kdb+), keeping "
            f"the same behaviour and the name `solve`.\n\n"
            f"{pyfence(rec['python_solution'])}",
            qfence(rec["q_solution"]))


def q2py(rec: dict) -> tuple[str, str]:
    return (f"Translate this q (kdb+) function into equivalent Python, keeping "
            f"the same behaviour and the name `solve`.\n\n"
            f"{qfence(rec['q_solution'])}",
            pyfence(rec["python_solution"]))


def q2desc(rec: dict) -> tuple[str, str]:
    return (f"Explain what this q (kdb+) function computes, in plain English. "
            f"Describe the behaviour and the accepted inputs.\n\n"
            f"{qfence(rec['q_solution'])}",
            rec["description"].strip())


def vectorise(rec: dict) -> tuple[str, str] | None:
    imperative = rec.get("q_solution_imperative")
    if not imperative:
        return None
    return (f"Rewrite this q (kdb+) function in idiomatic vector style with "
            f"identical behaviour — no `while`, no `do`, no index counters.\n\n"
            f"{qfence(imperative)}",
            qfence(rec["q_solution"]))


def fix(rec: dict, broken: dict) -> tuple[str, str]:
    inp = json.dumps(broken["input"], default=str)
    exp = json.dumps(broken["expected"], default=str)
    got = json.dumps(broken["got"], default=str)
    return (f"This q (kdb+) function is wrong.\n\n"
            f"{qfence(broken['q_solution'])}\n\n"
            f"It should solve: {rec['description']}\n\n"
            f"On the input `{inp[:600]}` it returns `{got[:200]}` but the "
            f"correct answer is `{exp[:200]}`.\n\n"
            f"Give the corrected function. Reply with one fenced q block.",
            qfence(rec["q_solution"]))


# ------------------------------------------------------------- assembly

def match_rejects(accepted: list[dict], rejects: list[dict]) -> dict[str, list[dict]]:
    """Rejects store a 200-char description; match them back by that prefix."""
    index = {" ".join(a["description"].split())[:180]: a["id"] for a in accepted}
    out: dict[str, list[dict]] = {}
    for r in rejects:
        if r.get("reason") != "q_disagreed":
            continue                     # "ran badly" has no expected/got pair
        if len(r.get("q_solution", "")) >= 600:
            continue                     # truncated in the log; would teach a stub
        if not all(k in r for k in ("input", "expected", "got")):
            continue
        key = " ".join(r["description"].split())[:180]
        if key in index:
            out.setdefault(index[key], []).append(r)
    return out


def benchmark_shingles() -> set[str]:
    path = ROOT / "q-evaluation-harness/datasets/q_humaneval.jsonl"
    shingles: set[str] = set()
    if not path.exists():
        print("WARNING: benchmark not found; decontamination disabled")
        return shingles
    for line in path.open():
        row = json.loads(line)
        for field in ("prompt", "tests", "canonical_solution"):
            text = re.sub(r"\s+", " ", str(row.get(field, ""))).strip()
            for i in range(0, max(len(text) - SHINGLE, 0), SHINGLE // 2):
                shingles.add(text[i:i + SHINGLE])
    return shingles


def contaminated(text: str, shingles: set[str]) -> bool:
    flat = re.sub(r"\s+", " ", text).strip()
    for i in range(0, max(len(flat) - SHINGLE, 0), SHINGLE // 2):
        if flat[i:i + SHINGLE] in shingles:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="corpus/qdataset.audited.jsonl")
    ap.add_argument("--rejects", default="corpus/qdataset_rejects.jsonl")
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--max-fix-per-problem", type=int, default=3)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    accepted = [json.loads(l) for l in (ROOT / args.src).open()]
    rejects = ([json.loads(l) for l in (ROOT / args.rejects).open()]
               if (ROOT / args.rejects).exists() else [])
    by_problem = match_rejects(accepted, rejects)
    rng = random.Random(args.seed)

    samples: list[dict] = []
    for rec in accepted:
        # A record may restrict its tasks (bank_to_records.py gives templated
        # family exercises desc2q and py2q only, to downweight them).
        allowed = set(rec.get("tasks") or ("desc2q", "py2q", "q2py", "q2desc"))
        for name, fn in (("desc2q", desc2q), ("py2q", py2q),
                         ("q2py", q2py), ("q2desc", q2desc)):
            if name not in allowed:
                continue
            samples.append({"task": name, "problem": rec["id"],
                            "prompt": fn(rec)[0], "response": fn(rec)[1]})
        got = vectorise(rec)
        if got:
            samples.append({"task": "vectorise", "problem": rec["id"],
                            "prompt": got[0], "response": got[1]})
        broken = by_problem.get(rec["id"], [])
        rng.shuffle(broken)
        for b in broken[:args.max_fix_per_problem]:
            p, r = fix(rec, b)
            samples.append({"task": "fix", "problem": rec["id"],
                            "prompt": p, "response": r})

    # Decontamination again at sample level. The corpus was checked, but `fix`
    # samples embed reject text that never went through that check.
    shingles = benchmark_shingles()
    before = len(samples)
    samples = [s for s in samples
               if not contaminated(s["prompt"] + " " + s["response"], shingles)]
    dropped = before - len(samples)

    # Split by PROBLEM, not by sample: the six views of one problem share a
    # solution, so a sample-level split leaks the answer into validation.
    ids = sorted({s["problem"] for s in samples})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = set(ids[:n_val])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    def encode(s: dict) -> tuple[list[int], list[int]] | None:
        # enable_thinking=False is not a style choice, it is train/serve parity.
        # Qwen3.5's template emits `<think>\n` when thinking is on and
        # `<think>\n\n</think>\n\n` when it is off, and every number in this
        # project — the 10.2% baseline included — was measured against a server
        # started with `--reasoning off`, which sets enable_thinking false in
        # the template's context. Training on the thinking-on prefix would
        # supervise the model on tokens the server never sends.
        #
        # return_dict=False: recent transformers returns a BatchEncoding here,
        # and a plain list is what the concatenation below needs.
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": s["prompt"]}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
            enable_thinking=False)
        response = tok(s["response"] + tok.eos_token,
                       add_special_tokens=False)["input_ids"]
        ids = prompt + response
        if len(ids) > args.seq_len:
            return None
        labels = [IGNORE] * len(prompt) + response
        pad = args.seq_len - len(ids)
        return ids + [tok.pad_token_id or 0] * pad, labels + [IGNORE] * pad

    train_ids, train_lab, val_ids_, val_lab = [], [], [], []
    too_long = Counter()
    kept = Counter()
    for s in samples:
        enc = encode(s)
        if enc is None:
            too_long[s["task"]] += 1
            continue
        ids_, lab = enc
        kept[s["task"]] += 1
        if s["problem"] in val_ids:
            val_ids_.append(ids_); val_lab.append(lab)
        else:
            train_ids.append(ids_); train_lab.append(lab)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train.npy", np.array(train_ids, dtype=np.int32))
    np.save(out / "train_labels.npy", np.array(train_lab, dtype=np.int32))
    np.save(out / "val.npy", np.array(val_ids_, dtype=np.int32))
    np.save(out / "val_labels.npy", np.array(val_lab, dtype=np.int32))
    with (out / "samples.jsonl").open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s) + "\n")

    supervised = int((np.array(train_lab) != IGNORE).sum()) if train_lab else 0
    summary = {
        "problems": len(accepted),
        "problems_with_rejects": len(by_problem),
        "samples_built": before,
        "dropped_contaminated": dropped,
        "dropped_too_long": dict(too_long),
        "kept_by_task": dict(kept),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids_),
        "val_problems": sorted(val_ids),
        "seq_len": args.seq_len,
        "supervised_tokens": supervised,
        "tokenizer": TOKENIZER,
        # Recorded so check_prompt_parity.py can assert the serving prefix
        # matches this one before any SFT checkpoint is scored.
        "generation_prefix": tok.apply_chat_template(
            [{"role": "user", "content": "PROBE"}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False),
        "source": args.src,
        "source_sha256": hashlib.sha256(
            (ROOT / args.src).read_bytes()).hexdigest()[:16],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("SFT_ASSEMBLE_DONE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

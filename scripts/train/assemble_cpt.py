#!/usr/bin/env python3
"""Assemble the continued-pretraining corpus, decontaminated and packed.

Inputs are the human-written sources the pre-registration permits:
  * corpus/raw + corpus/keep.txt — licence-filtered GitHub q, non-q filtered out
  * corpus/kx-docs               — Kx documentation (CC-BY-4.0)
  * corpus/hf/q_pretrained_dataset — Morgan Stanley's harvested corpus (Apache-2.0,
    human-written code, not model output)
  * corpus/hf/kdb_q              — small curated q snippets (Apache-2.0)

Decontamination runs in both directions against Q-HumanEval: any source file
sharing a long verbatim span with a benchmark prompt *or* its tests is dropped,
and the count is reported. Public q repos do contain HumanEval-style exercises,
so this is not theoretical.

Output: a packed, tokenised dataset on disk plus a stats file for the write-up.

    python scripts/train/assemble_cpt.py --out data/cpt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
TOKENIZER = ("/home/joedowling/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/"
             "snapshots/2fc06364715b967f1860aea9cf38778875588b17")
SHINGLE = 60           # characters; long enough that collisions are real reuse


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def benchmark_shingles() -> set[str]:
    """Verbatim spans from the benchmark, both prompts and tests."""
    path = ROOT / "q-evaluation-harness/datasets/q_humaneval.jsonl"
    shingles: set[str] = set()
    if not path.exists():
        print("WARNING: benchmark not found; decontamination disabled")
        return shingles
    for line in path.open():
        row = json.loads(line)
        for field in ("prompt", "tests", "canonical_solution", "entry_point"):
            text = normalise(str(row.get(field, "")))
            for i in range(0, max(len(text) - SHINGLE, 0), SHINGLE // 2):
                shingles.add(text[i:i + SHINGLE])
    return shingles


def contaminated(text: str, shingles: set[str]) -> bool:
    flat = normalise(text)
    for i in range(0, max(len(flat) - SHINGLE, 0), SHINGLE // 2):
        if flat[i:i + SHINGLE] in shingles:
            return True
    return False


def sources() -> list[tuple[str, str]]:
    """(source_label, text) for every permitted document."""
    out: list[tuple[str, str]] = []

    keep = ROOT / "corpus/keep.txt"
    if keep.exists():
        for p in keep.read_text().split():
            try:
                out.append(("github-q", Path(p).read_text(errors="ignore")))
            except OSError:
                continue

    docs = ROOT / "corpus/kx-docs/docs"
    if docs.exists():
        for p in sorted(docs.rglob("*.md")):
            out.append(("kx-docs", p.read_text(errors="ignore")))

    import glob
    import pandas as pd
    for f in sorted(glob.glob(str(ROOT / "corpus/hf/q_pretrained_dataset/**/*.parquet"),
                              recursive=True)):
        for text in pd.read_parquet(f)["text"].tolist():
            out.append(("ms-pretrain", str(text)))

    kdbq = ROOT / "corpus/hf/kdb_q/train.json"
    if kdbq.exists():
        data = json.loads(kdbq.read_text())
        for item in (data if isinstance(data, list) else data.get("data", [])):
            out.append(("rd200-kdb_q", item if isinstance(item, str) else json.dumps(item)))

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/cpt")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--val-fraction", type=float, default=0.02)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)

    shingles = benchmark_shingles()
    print(f"benchmark shingles: {len(shingles)}")

    docs = sources()
    print(f"documents in: {len(docs)}")

    seen: set[str] = set()
    kept: list[tuple[str, str]] = []
    stats = Counter()
    for label, text in docs:
        if not text or not text.strip():
            stats["empty"] += 1
            continue
        digest = hashlib.sha256(normalise(text).encode()).hexdigest()
        if digest in seen:
            stats["duplicate"] += 1
            continue
        if contaminated(text, shingles):
            stats[f"contaminated:{label}"] += 1
            continue
        seen.add(digest)
        kept.append((label, text))
        stats[f"kept:{label}"] += 1

    print("\nassembly:")
    for k, v in sorted(stats.items()):
        print(f"  {k:28} {v}")

    # Pack into fixed-length blocks: concatenate with a separator, then chunk.
    ids: list[int] = []
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    for _, text in kept:
        ids.extend(tok(text)["input_ids"] + [eos])
    n_blocks = len(ids) // args.seq_len
    blocks = [ids[i * args.seq_len:(i + 1) * args.seq_len] for i in range(n_blocks)]
    n_val = max(1, int(len(blocks) * args.val_fraction))

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    np.save(out / "train.npy", np.array(blocks[n_val:], dtype=np.int32))
    np.save(out / "val.npy", np.array(blocks[:n_val], dtype=np.int32))
    summary = {
        "documents_in": len(docs), "documents_kept": len(kept),
        "tokens": len(ids), "seq_len": args.seq_len,
        "blocks_train": len(blocks) - n_val, "blocks_val": n_val,
        "stats": dict(stats),
        "contaminated_total": sum(v for k, v in stats.items() if k.startswith("contaminated")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n{len(ids)/1e6:.2f}M tokens -> {len(blocks)} blocks of {args.seq_len} "
          f"({len(blocks)-n_val} train / {n_val} val)")
    print(f"dropped for benchmark overlap: {summary['contaminated_total']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

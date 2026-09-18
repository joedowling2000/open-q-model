#!/usr/bin/env python3
"""Measure the harvested corpus in the unit that matters: tokens.

qqWen's continued-pretraining corpus was "over five million characters and more
than 1.6 million tokens" (arXiv:2508.06813, Table 4: 15 repos plus Kx docs). The
question this answers is what multiple of that a licence-filtered public harvest
actually yields — the number the whole "can we build a better one" plan rests on.

Tokenised with the Qwen3.5 tokeniser, since that family is the likely base.

    python scripts/corpus/measure.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
QQWEN_TOKENS = 1_600_000          # their whole pretraining corpus
TOKENIZER = ("/home/joedowling/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/"
             "snapshots/2fc06364715b967f1860aea9cf38778875588b17")


def tokeniser():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def count(paths, tok, label: str) -> tuple[int, int, int]:
    n_files = n_bytes = n_tokens = 0
    batch: list[str] = []
    for p in paths:
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        if not text.strip():
            continue
        n_files += 1
        n_bytes += len(text.encode())
        batch.append(text)
        if len(batch) >= 200:
            n_tokens += sum(len(x) for x in tok(batch)["input_ids"])
            batch = []
    if batch:
        n_tokens += sum(len(x) for x in tok(batch)["input_ids"])
    print(f"{label:28} {n_files:>6} files  {n_bytes/1e6:>8.1f} MB  {n_tokens/1e6:>7.2f}M tokens")
    return n_files, n_bytes, n_tokens


def main() -> int:
    tok = tokeniser()
    print(f"{'source':28} {'files':>6}  {'size':>11}  {'tokens':>14}")
    print("-" * 66)

    q_files = sorted((ROOT / "corpus/raw").rglob("*.q"))
    qf, qb, qt = count(q_files, tok, "GitHub q source")

    docs = ROOT / "corpus/kx-docs/docs"
    doc_files = sorted(docs.rglob("*.md")) if docs.exists() else []
    df, db, dt = count(doc_files, tok, "Kx docs (CC-BY-4.0)")

    print("-" * 66)
    total = qt + dt
    print(f"{'TOTAL':28} {qf+df:>6} files  {(qb+db)/1e6:>8.1f} MB  {total/1e6:>7.2f}M tokens")
    print(f"\nqqWen's pretraining corpus: {QQWEN_TOKENS/1e6:.2f}M tokens")
    print(f"this harvest is {total/QQWEN_TOKENS:.1f}x that "
          f"({qt/QQWEN_TOKENS:.1f}x from code alone)")

    manifest = [json.loads(l) for l in (ROOT / "corpus/manifest.jsonl").open()]
    lic = Counter(m["license"] for m in manifest)
    print("\nlicences of harvested q files:")
    for k, v in lic.most_common():
        print(f"  {k or 'none':16} {v:>5} files")

    by_repo = Counter()
    for m in manifest:
        by_repo[m["repo"]] += m["bytes"]
    print("\nlargest contributors:")
    for repo, b in by_repo.most_common(10):
        print(f"  {repo:44} {b/1e6:>6.2f} MB")

    concentration = sum(b for _, b in by_repo.most_common(10)) / max(sum(by_repo.values()), 1)
    print(f"\ntop 10 repos are {concentration:.0%} of the code by bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

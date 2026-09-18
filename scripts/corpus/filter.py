#!/usr/bin/env python3
"""Separate real kdb+ q from everything else that happens to use a .q extension.

The raw harvest is 14.5M tokens, but the largest "contributors" are a JSON maths
dataset (FERMAT), Chinese QuickMacro automation scripts (.Q, INI format) and
similar. qqWen handled this by scoring every file 0-10 with Qwen-2.5-32B and
keeping >=4, then hand-checking. That needs the GPU, which is busy, so this is
the cheap first pass: structural signals that q has and the impostors do not.

Anything this keeps should still get an LLM pass before training on it; anything
it rejects is rejected for a stated, inspectable reason.

    python scripts/corpus/filter.py --report out/corpus_filter.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")

# Signals of genuine q. Deliberately structural rather than keyword-only: many
# of these tokens ("each", "til") appear in English prose too.
Q_PATTERNS = [
    re.compile(r"^\s*[.\w]+\s*:\s*\{", re.M),        # name:{[x] ...}  function definition
    re.compile(r"\\d\s+\.[a-zA-Z]"),                  # \d .ns  namespace directive
    re.compile(r"\bselect\b[^\n]{0,80}\bfrom\b", re.I),
    re.compile(r"\bupsert\b|\binsert\b"),
    re.compile(r"0N!"),                                # the q print/trace idiom
    re.compile(r"\.z\.[a-z]{1,3}\b"),                  # .z.p, .z.ts, .z.pg ...
    re.compile(r"`[a-zA-Z][\w.]*"),                    # `symbols
    re.compile(r"^\s*//", re.M),                       # q line comment
    re.compile(r"\beach\b|\bover\b|\bscan\b"),
    re.compile(r"\{\[[a-zA-Z][\w;]*\]"),               # explicit parameter list
]

JSON_START = re.compile(r"^\s*[\[{]\s*\"")
INI_START = re.compile(r"^\s*\[[A-Za-z ]+\]\s*$", re.M)
QUICKMACRO = re.compile(r"SyntaxVersion=|BeginHotkey=|MacroID=")
PY_OR_JS = re.compile(r"^\s*(import |from \w+ import |def \w+\(|function \w+\(|const \w+ =)", re.M)
HTML = re.compile(r"<(html|div|span|script)\b", re.I)
CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")


def classify(text: str) -> tuple[bool, str, int]:
    """Return (keep, reason, n_signals)."""
    if not text.strip():
        return False, "empty", 0
    if JSON_START.match(text):
        return False, "json", 0
    if QUICKMACRO.search(text[:4000]) or (INI_START.search(text[:2000]) and "=" in text[:2000]):
        return False, "ini/quickmacro", 0

    sample = text[:200_000]
    cjk = len(CJK.findall(sample))
    if cjk > len(sample) * 0.02:
        return False, "cjk-heavy", 0
    if HTML.search(sample[:4000]):
        return False, "html", 0
    if PY_OR_JS.search(sample[:4000]) and not re.search(r"^\s*[.\w]+\s*:\s*\{", sample[:4000], re.M):
        return False, "python/js", 0

    signals = sum(1 for p in Q_PATTERNS if p.search(sample))
    per_kb = signals / max(len(sample) / 1024, 1)
    # Threshold scales with size. A 46-byte test stub ("a5Loaded:1b;") is real q
    # and can only show one or two signals; a 50 KB file with two is data. A flat
    # >=3 rule rejected genuine TorQ configs and finos/kdb test modules.
    need = 1 if len(sample) < 4_000 else (2 if len(sample) < 20_000 else 3)
    if signals < need:
        return False, "too-few-q-signals", signals
    # A very large file with almost no q structure is data, not code.
    if len(sample) > 50_000 and per_kb < 0.02:
        return False, "signal-sparse", signals
    return True, "keep", signals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default="out/corpus_filter.json")
    ap.add_argument("--keep-list", default="corpus/keep.txt")
    args = ap.parse_args()

    manifest = [json.loads(l) for l in (ROOT / "corpus/manifest.jsonl").open()]
    reasons = Counter()
    kept_bytes = dropped_bytes = 0
    kept: list[str] = []
    dropped_by_repo = Counter()

    for m in manifest:
        path = ROOT / "corpus/raw" / m["sha256"][:2] / f"{m['sha256']}.q"
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            reasons["unreadable"] += 1
            continue
        keep, reason, _ = classify(text)
        reasons[reason] += 1
        if keep:
            kept.append(str(path))
            kept_bytes += m["bytes"]
        else:
            dropped_bytes += m["bytes"]
            dropped_by_repo[m["repo"]] += m["bytes"]

    Path(args.keep_list).write_text("\n".join(kept))
    report = {
        "files_in": len(manifest),
        "files_kept": len(kept),
        "bytes_kept": kept_bytes,
        "bytes_dropped": dropped_bytes,
        "reasons": dict(reasons),
        "largest_drops": dropped_by_repo.most_common(10),
    }
    Path(args.report).write_text(json.dumps(report, indent=1))

    print(f"kept {len(kept)}/{len(manifest)} files, "
          f"{kept_bytes/1e6:.1f} MB of {(kept_bytes+dropped_bytes)/1e6:.1f} MB")
    for reason, n in reasons.most_common():
        print(f"  {reason:20} {n:>5}")
    print("\nlargest drops by repo:")
    for repo, b in dropped_by_repo.most_common(6):
        print(f"  {repo:44} {b/1e6:>6.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

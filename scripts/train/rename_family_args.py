#!/usr/bin/env python3
"""Give family-exercise q solutions the problem's real argument names.

The bank's template writes every family solution as `solve:{[arg0;arg1] ...}`,
whatever the problem calls its inputs. After SFT on v5, the model answered
"running maximum of `xs`" with `solve:{[arg0]maxs arg0}`: correct, but it had
learned to ignore the names the problem gives. The rename is mechanical, and
each renamed solution must pass qcheck on 60 fresh inputs. One that fails, or
whose names would collide with a q reserved word, keeps its original text.

Writes corpus/q_study_v5/family_renamed.json ({id: new q}); bank_to_records.py
applies it.

    python scripts/train/rename_family_args.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
sys.path.insert(0, str(ROOT / "scripts/synth"))
import qcheck  # noqa: E402

BANK = ROOT / "corpus/q_study_v5"
SEEDS = range(900000, 900060)
# Names that would shadow q keywords or built-ins if used as parameters.
RESERVED = {"abs", "all", "and", "any", "asc", "avg", "count", "cols", "delete",
            "desc", "div", "do", "each", "exec", "exp", "first", "flip", "from",
            "group", "if", "in", "key", "last", "like", "log", "max", "min", "mod",
            "neg", "not", "or", "over", "prd", "prev", "rank", "reverse", "scan",
            "select", "sum", "sums", "til", "type", "update", "value", "where",
            "while", "within", "xor", "x", "y", "z", "v", "t", "d"}


def rename(code: str, names: list[str]) -> str | None:
    if any(n in RESERVED or not re.fullmatch(r"[A-Za-z]\w*", n) for n in names):
        return None
    # Refuse if a new name already appears as a token (it would be captured).
    if any(re.search(rf"(?<![\w`.]){re.escape(n)}(?![\w])", code) for n in names):
        return None
    out = code
    for i, n in enumerate(names):
        out = re.sub(rf"(?<![\w`.])arg{i}(?![\w])", n, out)
    return out if out != code else None


def main() -> int:
    recs = [json.loads(l) for f in ("questions.jsonl", "reserve_exercises.jsonl")
            for l in (BANK / f).open()]
    fam = [r for r in recs if r["question_type"] != "original" and r["solutions"]]

    def one(r):
        names = qcheck.arg_order(r)
        new = rename(r["solutions"][0]["code"], names)
        if new is None:
            return r["id"], None, "skipped"
        res = qcheck.check(new, r["python_src"], names, SEEDS)
        return r["id"], (new if res["ok"] else None), ("ok" if res["ok"] else "failed")

    with ThreadPoolExecutor(2) as pool:   # keep CPU heat low while a scoring run is live
        results = list(pool.map(one, fam))
    renamed = {i: new for i, new, s in results if new}
    (BANK / "family_renamed.json").write_text(json.dumps(renamed, indent=0))
    print(dict(Counter(s for _, _, s in results)))
    print(next(iter(renamed.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Merge two independent sample sets for the same model into one.

pass@k is estimated from n independent draws at a fixed temperature. The 10- and
20-sample passes draw from the same model with the same parameters and different
random seeds, so their union is a valid 30-sample set — no need to regenerate
the first ten, which is a day of GPU time on this box.

Refuses to merge if the two files disagree about the problem set, and renumbers
sample_index so each task's samples are 0..n-1.

    python scripts/merge_samples.py results/m.jsonl results/m-n20.jsonl results/m-n30.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> dict[int, list[dict]]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    with path.open() as fh:
        for line in fh:
            row = json.loads(line)
            by_task[row["task_id"]].append(row)
    return by_task


def main() -> int:
    a_path, b_path, out_path = (Path(p) for p in sys.argv[1:4])
    a, b = load(a_path), load(b_path)

    if set(a) != set(b):
        print(f"ERROR: task sets differ ({len(a)} vs {len(b)} tasks)", file=sys.stderr)
        return 1

    models = {r.get("model_name") for rows in (a, b) for row in rows.values() for r in row}
    if len(models) > 1:
        print(f"ERROR: mixed models in inputs: {models}", file=sys.stderr)
        return 1

    counts = set()
    with out_path.open("w") as out:
        for task_id in sorted(a):
            rows = a[task_id] + b[task_id]
            counts.add(len(rows))
            for i, row in enumerate(rows):
                row["sample_index"] = i
                out.write(json.dumps(row) + "\n")

    if len(counts) != 1:
        print(f"WARNING: uneven sample counts per task: {sorted(counts)}", file=sys.stderr)
    print(f"merged {a_path.name} + {b_path.name} -> {out_path.name} "
          f"({len(a)} tasks x {sorted(counts)} samples)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

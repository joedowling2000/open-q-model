#!/usr/bin/env python3
"""Trim a solutions file to exactly N samples per problem.

The harness appends to its output file rather than truncating it, so a restarted
run leaves the previous attempt's samples in place: 160 stale rows + 1640 new
ones = 1800, with a handful of problems carrying 20 samples and the rest 10.
pass@k across unevenly sampled problems is not the statistic it claims to be.

Keeps the LAST n rows per task (the completed run appends last) and reports what
it dropped.

    python scripts/trim_samples.py results/m.jsonl 10
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    path, n = Path(sys.argv[1]), int(sys.argv[2])
    by_task: dict[int, list[dict]] = defaultdict(list)
    for line in path.open():
        row = json.loads(line)
        by_task[row["task_id"]].append(row)

    before = sum(len(v) for v in by_task.values())
    short = {t: len(v) for t, v in by_task.items() if len(v) < n}
    if short:
        print(f"ERROR: {len(short)} tasks have fewer than {n} samples: "
              f"{dict(list(short.items())[:5])}", file=sys.stderr)
        return 1

    backup = path.with_suffix(path.suffix + ".untrimmed")
    if not backup.exists():
        path.replace(backup)
    else:
        print(f"note: {backup.name} already exists, keeping it")

    with path.open("w") as out:
        for task_id in sorted(by_task):
            for i, row in enumerate(by_task[task_id][-n:]):
                row["sample_index"] = i
                out.write(json.dumps(row) + "\n")

    after = len(by_task) * n
    print(f"trimmed {path.name}: {before} -> {after} rows "
          f"({len(by_task)} tasks x {n}); original kept as {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

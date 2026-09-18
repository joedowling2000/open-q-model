#!/usr/bin/env python3
"""Decide whether a Q-HumanEval run is trustworthy enough to build on.

The failure mode this exists for: a run that *looks* finished but is hollow.
When llama.cpp rejected n > slots, every request 400'd, the harness wrote 1640
empty completions, and the summary line reported a perfectly ordinary-looking
0%. Nothing raised. So check the things that would have caught it.

    python scripts/validate.py 10 qwen3.5-27b gemma-4-31b-it qqwen-32b-rl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N_PROBLEMS = 164
MIN_NONEMPTY = 0.95      # fraction of completions that must have content
MAX_ERRORED = 0.05       # fraction of gradings allowed to be infrastructure errors


def check(name: str, n_samples: int, suffix: str = "") -> tuple[bool, list[str]]:
    notes: list[str] = []
    ok = True

    sols = ROOT / "results" / f"{name}{suffix}.jsonl"
    scored = ROOT / "results" / f"{name}{suffix}-scored.json"
    gen_log = ROOT / "logs" / f"gen-{name}.log"

    if not sols.exists():
        return False, [f"missing {sols.name}"]
    rows = [json.loads(line) for line in sols.open()]

    expected = N_PROBLEMS * n_samples
    if len(rows) != expected:
        ok = False
        notes.append(f"{len(rows)} solutions, expected {expected}")

    nonempty = sum(1 for r in rows if (r.get("completion") or "").strip())
    frac = nonempty / max(len(rows), 1)
    notes.append(f"non-empty {frac:.1%}")
    if frac < MIN_NONEMPTY:
        ok = False

    # Distinct completions guard against a server returning one cached answer.
    distinct = len({(r.get("completion") or "")[:200] for r in rows})
    notes.append(f"{distinct} distinct")
    if distinct < N_PROBLEMS:
        ok = False
        notes.append("too few distinct completions")

    if gen_log.exists():
        failures = len(re.findall(r"Async generation failed", gen_log.read_text(errors="ignore")))
        if failures:
            ok = False
            notes.append(f"{failures} async generation failures")

    if not scored.exists():
        return False, notes + [f"missing {scored.name}"]
    data = json.loads(scored.read_text())
    metrics = data.get("metrics", data)
    passes = {k: v for k, v in metrics.items() if str(k).lower().startswith("pass")}
    if not passes:
        ok = False
        notes.append("no pass@k metrics")
    else:
        notes.append(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in sorted(passes.items())))

    results = data.get("results") or []
    if results:
        errored = sum(1 for r in results
                      if "error" in str(r.get("status", "")).lower())
        frac_err = errored / len(results)
        notes.append(f"infra-errored {frac_err:.1%}")
        if frac_err > MAX_ERRORED:
            ok = False

    return ok, notes


def main() -> int:
    n_samples = int(sys.argv[1])
    names = sys.argv[2:]
    suffix = "" if n_samples == 10 else f"-n{n_samples}"
    all_ok = True
    for name in names:
        ok, notes = check(name, n_samples, suffix)
        print(f"{'PASS' if ok else 'FAIL'} {name}: {'; '.join(notes)}", flush=True)
        all_ok &= ok
    print("VALIDATION_OK" if all_ok else "VALIDATION_FAILED", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Assert the server formats prompts exactly as training did. Fail loudly if not.

Qwen3.5's chat template emits `<think>\\n` when thinking is on and
`<think>\\n\\n</think>\\n\\n` when it is off. Every benchmark number in this
project was measured against `llama-server --reasoning off`, which sets
`enable_thinking` false in the template's context. Assembling SFT data with the
default (thinking on) supervises the model on a prefix the server never sends.

A mismatch here does not crash anything. It quietly costs accuracy, and the
result looks like "fine-tuning did not help" — a wrong conclusion about the
experiment rather than a visible bug. This project has already lost hours to
exactly that shape of failure (a serving flag producing a plausible zero), so
the check runs before any SFT checkpoint is scored.

    python scripts/train/check_prompt_parity.py --port 8102 --data data/sft
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path("/home/joedowling/Projects/qeval")
PROBE = "PROBE"


def served_prefix(port: int, model: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROBE}],
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/apply-template", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)["prompt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--model", default="qwen3.5-27b-cpt")
    ap.add_argument("--data", default="data/sft")
    args = ap.parse_args()

    summary = json.loads((ROOT / args.data / "summary.json").read_text())
    trained = summary.get("generation_prefix")
    if not trained:
        print("PARITY_FAIL no generation_prefix recorded in", args.data)
        return 2

    try:
        served = served_prefix(args.port, args.model)
    except Exception as exc:
        print(f"PARITY_FAIL could not reach the server: {type(exc).__name__}: {exc}")
        return 2

    # The server may prepend a system message the training side did not carry.
    # What has to match is the assistant turn opening — the tokens immediately
    # before the model's first generated token, which is where the think-block
    # difference lives.
    def tail(text: str) -> str:
        marker = "<|im_start|>assistant"
        return text[text.rindex(marker):] if marker in text else text

    if tail(served) == tail(trained):
        print("PARITY_OK", repr(tail(trained)))
        return 0

    print("PARITY_FAIL — the server does not format prompts the way training did.")
    print("  trained on:", repr(tail(trained)))
    print("  served    :", repr(tail(served)))
    print("  full served prompt:", repr(served))
    print("Fix the mismatch before scoring: either re-assemble the SFT data with"
          " the serving setting, or start the server to match training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())

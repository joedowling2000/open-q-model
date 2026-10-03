#!/usr/bin/env python3
"""Coder-Next's bf16 weights for QLoRA training (amendment 12). 159 GB, 40 shards.
Resumable; every file's size is checked against the Hub's listing at the end."""
import json, sys, urllib.request
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen3-Coder-Next"
path = Path(snapshot_download(REPO, max_workers=4))
with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true") as r:
    want = {s["rfilename"]: s.get("size") for s in json.load(r)["siblings"]}
bad = [f for f, n in want.items() if n and (path / f).stat().st_size != n]
print("FETCH_FAILED" if bad else "FETCH_DONE", path, bad[:3])
sys.exit(1 if bad else 0)

#!/bin/bash
# Fetch the base weights with timeouts and retries. The plain snapshot_download
# stalls: the server drops the connection and the client waits forever (seen
# twice now — gemma-3-4b in the benchmark phase, Qwen3.5-27B here at 50/56 GB).
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30 HF_XET_HIGH_PERFORMANCE=1
/home/joedowling/venvs/jlens/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for attempt in range(8):
    try:
        p = snapshot_download("Qwen/Qwen3.5-27B", max_workers=8,
                              allow_patterns=["*.json","*.txt","*.safetensors","*.model","*.jinja"])
        print("BASE_READY", p, flush=True); break
    except Exception as exc:
        print("retry", attempt, type(exc).__name__, exc, flush=True)
else:
    print("BASE_FAILED", flush=True)
PY

#!/bin/bash
# Download the three evaluation models. Xet high-performance mode is on: the
# plain link runs at ~10 MB/s, which is 7 hours for 250 GB. (HF_HUB_ENABLE_HF_TRANSFER
# is deprecated — the hub moved to Xet.)
export HF_XET_HIGH_PERFORMANCE=1
/home/joedowling/venvs/jlens/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ['Qwen/Qwen3.5-27B','google/gemma-4-31B-it','morganstanley/qqWen-32B-RL-Reasoning']:
    print('START', repo, flush=True)
    snapshot_download(repo, max_workers=8,
                      allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.jinja'])
    print('OK', repo, flush=True)
print('ALL_DOWNLOADS_DONE', flush=True)
PY

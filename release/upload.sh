#!/bin/bash
# Upload the staged release to Hugging Face. Run only after the go-ahead.
set -euo pipefail
cd /home/joedowling/Projects/qeval
REPO=${REPO:-josephdowling10/open-q-27b}
/home/joedowling/venvs/jlens/bin/python - "$REPO" <<'PY'
import sys
from huggingface_hub import HfApi
api, repo = HfApi(), sys.argv[1]
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path="release/stage/open-q-27b")
print("UPLOADED", f"https://huggingface.co/{repo}")
PY

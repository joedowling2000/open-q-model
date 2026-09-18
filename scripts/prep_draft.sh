#!/bin/bash
# Draft model for speculative decoding: same family/tokeniser as Qwen3.5-27B.
export HF_XET_HIGH_PERFORMANCE=1
bash /home/joedowling/Projects/qeval/scripts/prepare_model.sh Qwen/Qwen3.5-0.8B qwen3.5-0.8b

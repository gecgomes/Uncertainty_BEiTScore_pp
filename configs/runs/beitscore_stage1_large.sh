#!/usr/bin/env bash
# BEiTScore Stage 1 (large): binary correct/incorrect caption training on Stage1/stage1.pkl,
# initialised from the BEiT-3 VQA checkpoint.
set -euo pipefail
python scripts/train.py \
    --benchmark stage1 \
    --model_size large \
    --model_type indomain

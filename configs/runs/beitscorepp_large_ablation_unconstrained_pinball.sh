#!/usr/bin/env bash
# BEiTScore++ ablation: independent quantile heads trained with the pinball loss.
set -euo pipefail
python scripts/train.py \
    --benchmark quant \
    --model_size large \
    --init_checkpoint "$BEITSCORE_MODELS/beitscore/large/model.safetensors" \
    --no-chained_construction \
    --pinball

#!/usr/bin/env bash
# BEiTScore++ ablation: chained head and centered interval loss, without cluster batching.
set -euo pipefail
python scripts/train.py \
    --benchmark quant \
    --model_size large \
    --init_checkpoint "$BEITSCORE_MODELS/beitscore/large/model.safetensors" \
    --no-use_clusters

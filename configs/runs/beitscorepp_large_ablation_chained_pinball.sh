#!/usr/bin/env bash
# BEiTScore++ ablation: chained (monotone) head trained with the pinball loss.
set -euo pipefail
python scripts/train.py \
    --benchmark quant \
    --model_size large \
    --init_checkpoint "$BEITSCORE_MODELS/beitscore/large/model.safetensors" \
    --pinball

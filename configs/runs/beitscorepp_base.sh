#!/usr/bin/env bash
# BEiTScore++ (base), as in the paper: chained quantile head, centered interval loss,
# cluster batching (K = 63), bias offset -2.5, batch 16, patience 15, started from the
# released BEiTScore base checkpoint (scripts/download_release.py models).
set -euo pipefail
python scripts/train.py \
    --benchmark quant \
    --model_size base \
    --init_checkpoint "$BEITSCORE_MODELS/beitscore/base/model.safetensors"

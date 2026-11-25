#!/usr/bin/env bash
# BEiTScore Stage 2 (base): regression on the human-judgement mixture, from the Stage-1 run.
# Set STAGE1_RUN to the run number written by beitscore_stage1_base.sh.
set -euo pipefail
python scripts/train.py \
    --benchmark regression \
    --model_size base \
    --get_checkpoint_from_finetuning \
    --task_resume stage1 \
    --use_run "${STAGE1_RUN:?set STAGE1_RUN}"

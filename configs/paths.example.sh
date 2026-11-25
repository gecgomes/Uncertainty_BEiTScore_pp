# Copy to configs/paths.sh, edit, and `source configs/paths.sh` before running any script.
# Every script reads its locations from these variables (see beitscore/paths.py).

# Benchmarks and images, laid out as described in the README (Data section).
export BEITSCORE_DATA=/path/to/data

# BEiT-3 checkpoints, the sentencepiece model (beit3.spm) and the BEiTScore checkpoints.
export BEITSCORE_MODELS=/path/to/models

# Training runs (finetuning/<task>/run-N), score dumps, calibration files and figures.
export BEITSCORE_OUTPUT=/path/to/outputs

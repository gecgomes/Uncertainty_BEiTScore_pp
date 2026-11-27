# Where the data, the checkpoints and the outputs live. Source it before running any script:
#
#   source configs/paths.sh
#
# By default they go in data/, models/ and outputs/ at the root of the repository (git ignores
# these folders). To use other folders, for example on a larger disk, set the variables before
# sourcing this file (in ~/.bashrc, or in a job script); this file keeps any value already set:
#
#   export BEITSCORE_DATA=/big/disk/beitscore/data
#
# BEITSCORE_DATA    benchmarks, training data and their images
# BEITSCORE_MODELS  BEiT-3 checkpoints, the sentencepiece model (beit3.spm), BEiTScore and BEiTScore++ weights
# BEITSCORE_OUTPUT  training runs (finetuning/<task>/run-N), scores, metrics, calibration files

_beitscore_repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export BEITSCORE_DATA="${BEITSCORE_DATA:-$_beitscore_repo/data}"
export BEITSCORE_MODELS="${BEITSCORE_MODELS:-$_beitscore_repo/models}"
export BEITSCORE_OUTPUT="${BEITSCORE_OUTPUT:-$_beitscore_repo/outputs}"
mkdir -p "$BEITSCORE_DATA" "$BEITSCORE_MODELS" "$BEITSCORE_OUTPUT"
unset _beitscore_repo
echo "BEITSCORE_DATA=$BEITSCORE_DATA"
echo "BEITSCORE_MODELS=$BEITSCORE_MODELS"
echo "BEITSCORE_OUTPUT=$BEITSCORE_OUTPUT"

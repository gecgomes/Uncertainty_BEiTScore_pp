# Run configurations

One training command per model of the papers, run from the repository root after `source configs/paths.sh`:

| Script | Model |
|---|---|
| `beitscore_stage1_{base,large}.sh` | BEiTScore Stage 1: correct versus incorrect captions on the Stage-1 data |
| `beitscore_stage2_{base,large}.sh` | BEiTScore Stage 2: regression on human ratings, from a Stage-1 run (`STAGE1_RUN=<n>`) |
| `beitscorepp_{base,large}.sh` | BEiTScore++, from the released BEiTScore checkpoint |
| `beitscorepp_large_ablation_*.sh` | The BEiTScore++ ablations |

Each run is written to `$BEITSCORE_OUTPUT/finetuning/<task>/run-<n>/`.

# Baselines

Scores other reference-free captioning metrics on the benchmarks of this repository, including LongCap-Arena (`vela`), and builds a LaTeX table that compares them with BEiTScore and BEiTScore++.
Nothing of these metrics is redistributed here: each setup script clones the authors' code and downloads their checkpoints.

The metrics need incompatible library versions, so each family has its own folder, environment and runner:

| Folder | Metrics | Environment | Runner |
|---|---|---|---|
| `encoders/` | `beitscore-b`, `beitscore-l`, `beitscorepp-b`, `beitscorepp-l`, `clipscore-b`, `clipscore-l`, `pacscore++-b`, `pacscore++-l`, `specs` | the environment of this repository | `encoders/run_encoders.py` |
| `fleur/` | `fleur` | `fleur/.venv` (LLaVA 1.1.3: torch 2.0.1, transformers 4.31.0) | `fleur/run_fleur.py` |
| `expert/` | `expert` | `expert/.venv` (LLaVA 1.2.2: torch 2.1.2, transformers 4.37.2) | `expert/run_expert.py` |
| `vqascore/` | `vqascore` | `vqascore/.venv` (t2v_metrics 1.2: torch 2.1.2) | `vqascore/run_vqascore.py` |
| (this folder) | the table | any of them | `make_table.py` |

All of them read and write `$BEITSCORE_OUTPUT/baselines/` (`--results` changes it):

```
$BEITSCORE_OUTPUT/baselines/
  items/<benchmark>.json            the image-caption pairs of a benchmark and its human judgements
  scores/<model>/<benchmark>.json   one score per distinct image-caption pair
  metrics/<model>/<benchmark>.json  every metric, written by make_table.py
  table.tex
```

A runner reuses saved scores when they exist for the same benchmark items; `--recompute` scores again and overwrites them.
Everything is stored as JSON, so every environment can read the files.

Benchmarks: `vela`, `vicr`, `polaris`, `nebula`, `flickr8k-expert`, `flickr8k-cf`, `composite`, `genai`, `nocaps-foil`, `valse`, `sugarcrepe`, `longcapvlcp`, `winoground`.
Download them first (see [docs/benchmarks.md](../docs/benchmarks.md)): `python scripts/prepare_data/benchmarks.py download`.
`encoders/run_encoders.py` turns them into the items every runner scores, because its environment has the `beitscore` package, so it runs first.

All commands below run from the repository root, after `source configs/paths.sh`.

## 1. Setup (once)

```bash
# environment of this repository: PAC-S++ and SPECS code and checkpoints
bash baselines/encoders/setup.sh

# one virtual environment per LLM-based metric (Python 3.10, as the LLaVA instructions ask)
bash baselines/fleur/setup.sh
bash baselines/expert/setup.sh
bash baselines/vqascore/setup.sh
```

| Setup | Clones | Downloads |
|---|---|---|
| `encoders/setup.sh` | github.com/aimagelab/pacscore, github.com/mbzuai-nlp/SPECS into `encoders/repos/` | `PAC++_clip_ViT-B-32.pth`, `PAC++_clip_ViT-L-14.pth` (links of the PAC-S README), `spec.pt` (Hugging Face `Xiaohud/SPECS`, the link of the SPECS model zoo) |
| `fleur/setup.sh` | github.com/Yebin46/FLEUR with its LLaVA submodule | |
| `expert/setup.sh` | github.com/hjkim811/EXPERT with its LLaVA submodule | |
| `vqascore/setup.sh` | | t2v_metrics 1.2 from PyPI, OpenAI CLIP from GitHub |

Weights downloaded from Hugging Face on first use: LAION CLIP ViT-B/32 and ViT-L/14 (CLIPScore), LLaVA-1.5-13B (FLEUR, EXPERT), the EXPERT LoRA, CLIP-FlanT5-XXL (VQAScore).
BEiTScore and BEiTScore++ use the release downloaded by `python scripts/download_release.py models`, or the checkpoint given with `--beitscore_<size>_checkpoint` / `--beitscorepp_<size>_checkpoint`; for BEiTScore++ the score is the calibrated median.

If the torch build that pip installs does not match your GPU, set `TORCH_INDEX` (for example `TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash baselines/fleur/setup.sh`).
Set `HF_HOME` to a folder with space for the 13B LLaVA weights (about 26 GB).

## 2. Score

```bash
BENCH="vela vicr polaris nebula flickr8k-expert composite"
# environment of this repository: prepares the benchmark items, then the encoder metrics
python baselines/encoders/run_encoders.py --benchmarks $BENCH

# the LLM-based metrics, each with its own interpreter
baselines/fleur/.venv/bin/python baselines/fleur/run_fleur.py --benchmarks $BENCH
baselines/expert/.venv/bin/python baselines/expert/run_expert.py --benchmarks $BENCH
baselines/vqascore/.venv/bin/python baselines/vqascore/run_vqascore.py --benchmarks $BENCH
```

Common options: `--models` (`all` or a subset of the runner's metrics), `--recompute`, `--device`, `--batch_size`, `--data` (default `$BEITSCORE_DATA`), `--results`.
`encoders/run_encoders.py` also takes `--prepare_only`, `--prepare_again` (after downloading a benchmark again) and the checkpoint options.

## 3. Table

```bash
python baselines/make_table.py --benchmarks vela vicr polaris nebula flickr8k-expert composite --models all
```

It computes every metric from the saved scores (nothing is scored), writes `metrics/` and `table.tex`, and lists the pairs of model and benchmark without scores (left as `--`).
Options: `--corr_metric tau_b|tau_c|rho` (default τc), `--pair_metric acc|acc_macro|auroc`, `--wino_metric text|image|group`, `--vela_targets` (default `pca pca_harmonic_mean pca_harmonic_minimum`; also `desc rel flu`), `--out`.
The table needs `booktabs` and, for `\rowcolor`, `\usepackage[table]{xcolor}`.

## VELA

Only the two evaluation groups, TestA (`dci_val`, 98 pairs) and TestB (`dci_test`, 108 pairs), are scored, each reported on its own.
The principal component behind the single human scores is fitted on the ratings of the train and val groups, which need no metric score; [beitscore/vela.py](../beitscore/vela.py) defines the targets, and the fitted weights are saved under `fit` in `metrics/<model>/vela.json`.

### Descriptiveness, relevance and fluency

`--vela_targets desc rel flu` reports the correlation with each rated dimension of LongCap-Arena instead of (or besides) the combined scores, for example the descriptiveness table:

```bash
python baselines/make_table.py --benchmarks vela --vela_targets desc --models all
```

`--vela_paper without_refs with_refs` adds rows with the values the VELA paper reports for VELA on the same test sets (Matsuda et al., EMNLP 2025, Table 7: without and with reference captions), for the desc, rel and flu columns. They are quoted, not computed (VELA is not run here), and are not ranked; cite the paper with the BibTeX key `matsuda2025vela` used in the table, or edit it in `common/table.py`.

## FLEUR and EXPERT

- Each runs on the LLaVA code its repository pins (a git submodule), installed as the LLaVA README asks (Python 3.10, `pip install -e`), so the torch and transformers versions are those of the original setup. EXPERT also needs `peft` 0.12.0 (the newest version that works with the `accelerate` 0.21.0 its LLaVA pins and reads the `layer_replication` field of its LoRA config).
- Same model (`liuhaotian/llava-v1.5-13b`; EXPERT adds the LoRA `hjkim811/EXPERT-llava-13b-lora`), conversation template (`llava_v1`), image padding, prompt, greedy decoding (`do_sample=False`, `num_beams=1`, at most 512 new tokens), fp16, and the score smoothing over the digit tokens.
- A reply without a number in [0, 1] gives no score (NaN). Such pairs are left out of the metrics and reported by `make_table.py`.
- Check the setup on a benchmark with published numbers before trusting new ones: `--benchmarks flickr8k-expert composite`, then compare with the FLEUR and EXPERT papers. Small differences can come from fp16 on different GPUs.

## Things to know

- Long captions are cut to what each encoder reads: 77 tokens for CLIPScore and PAC-S++ (by the PAC-S code itself, after its "A photo depicts" prompt), 248 for SPECS, 256 BPE tokens for BEiTScore. The LLaVA judges and VQAScore read the whole caption.
- Correlations use every human rating of a pair, the metric score repeated per rating (the CLIPScore protocol), as in `scripts/evaluate.py`.
- Pairwise accuracy counts a pair as correct only when the correct caption scores strictly higher.

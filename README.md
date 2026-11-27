# BEiTScore and BEiTScore++

Code for two reference-free image captioning metrics built on a BEiT-3 cross-encoder:

- **BEiTScore** ([arXiv:2605.21728](https://arxiv.org/abs/2605.21728)) predicts one quality score per image and caption. It is initialised from a BEiT-3 VQA checkpoint, trained in two stages (binary alignment with LLM-generated negatives, then regression on human ratings), and evaluated on correlation, error-detection and long-caption benchmarks, including the LongCapVLCP benchmark introduced with it.
- **BEiTScore++** ([Findings of EMNLP 2026](https://aclanthology.org/2026.findings-emnlp.2160/)) extends BEiTScore with a quantile head. For each image and caption it returns the predicted median of the human rating together with a confidence interval, built by a chained construction that rules out quantile crossing and calibrated post hoc with conformal risk control.

## Installation

```bash
git clone https://github.com/gecgomes/Uncertainty_BEiTScore_pp.git && cd Uncertainty_BEiTScore_pp
pip install -e ".[data]"     # the package (with torch 2.8.0) and the tools to download the models and data
source configs/paths.sh      # where data, checkpoints and outputs live
```

`pip install -e ".[data]"` installs PyTorch 2.8.0 from PyPI, built for CUDA 12.8.
For another CUDA version, install it first from the PyTorch index, for example `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126` (see https://pytorch.org).

`configs/paths.sh` puts everything in `data/`, `models/` and `outputs/` at the root of the repository.
To use other folders, for example on a larger disk, set `BEITSCORE_DATA`, `BEITSCORE_MODELS` or `BEITSCORE_OUTPUT` before sourcing it (in `~/.bashrc` or a job script); the file keeps any value already set.
The Stage-1 data alone takes about 150 GB.

## Download

Models and data are hosted on the Hugging Face Hub, and the evaluation benchmarks come from their authors.
Run everything from the repository root after `source configs/paths.sh`.

| What | How | Hugging Face |
|---|---|---|
| BEiTScore and BEiTScore++ weights | `python scripts/download_release.py models` | [beitscore](https://huggingface.co/goncaloecgomes/beitscore), [beitscorepp](https://huggingface.co/goncaloecgomes/beitscorepp) |
| Stage-1 training data (BEiTScore) | `python scripts/download_release.py stage1` | [beitscore-stage1](https://huggingface.co/datasets/goncaloecgomes/beitscore-stage1) |
| Stage-2 training and calibration data (BEiTScore Stage 2, BEiTScore++) | `python scripts/download_release.py stage2` | [beitscore-stage2](https://huggingface.co/datasets/goncaloecgomes/beitscore-stage2) |
| LongCapVLCP | `python scripts/download_release.py longcapvlcp` | [LongCapVLCP](https://huggingface.co/datasets/goncaloecgomes/LongCapVLCP) |
| Other evaluation benchmarks of both papers, including LongCap-Arena | `python scripts/prepare_data/benchmarks.py download`, see [docs/benchmarks.md](docs/benchmarks.md) | downloaded from their authors |

`python scripts/download_release.py all` downloads the models, Stage 1, Stage 2 and LongCapVLCP.

### Models

```bash
python scripts/download_release.py models
```

This downloads the base and large models of both metrics to `$BEITSCORE_MODELS/beitscore/<size>/` and `$BEITSCORE_MODELS/beitscorepp/<size>/`, with the BEiT-3 tokenizer (`$BEITSCORE_MODELS/beit3.spm`).
Each folder holds `model.safetensors` and `config.json`; the BEiTScore++ folders also hold `calibration.json`, the conformal calibration of the 90% interval, which the evaluation scripts read with `--calibration auto`.
`--sizes base` or `--sizes large` downloads one size only.

### Stage-1 training data

```bash
python scripts/download_release.py stage1
```

This downloads `goncaloecgomes/beitscore-stage1` (https://huggingface.co/datasets/goncaloecgomes/beitscore-stage1) to `$BEITSCORE_DATA/release/stage1/` (deleted at the end, `--keep_release` keeps it), rebuilds the images in `$BEITSCORE_DATA/Stage1/images/` (about 150 GB) and writes the training table `$BEITSCORE_DATA/Stage1/stage1.pkl`, with an `image_path` column pointing to them.

- PixelProse images, and the few images no longer available from their source, are stored in the dataset itself.
- COCO, Open Images and TextCaps images are downloaded from their official servers, only the ones used. COCO 2014 images are converted to RGB and saved as JPEG, as in training.
- Flickr30k images come from the Kaggle archive https://www.kaggle.com/datasets/adityajn105/flickr30k, downloaded automatically (about 4 GB) into `$BEITSCORE_DATA/flickr30k/Images`. They are hard-linked into `Stage1/images/flickr30k/` (no second copy on the same disk), and the folder stays, since the Composite benchmark uses it too. If the download fails (Kaggle may ask you to log in: `kagglehub.login()` or `~/.kaggle/kaggle.json`), download and unzip it yourself into that folder, or pass another folder with `--flickr30k`.

Every image is checked against the SHA-256 of the file used in training.
`$BEITSCORE_DATA/Stage1/images/images_report.csv` gives the status of each image, and the script prints how many rows, if any, could not be rebuilt (an official server may have removed a file since the release).
Rerunning the command resumes where it stopped; `scripts/get_images.py` does the image step on its own.

### Stage-2 training and calibration data

```bash
python scripts/download_release.py stage2
```

This downloads `goncaloecgomes/beitscore-stage2` to `$BEITSCORE_DATA/release/stage2/` (deleted at the end, like Stage 1's), rebuilds its images in `$BEITSCORE_DATA/stage2/images/` and writes `$BEITSCORE_DATA/stage2/stage2.pkl`, the file `scripts/train.py --benchmark regression|quant` trains on and `scripts/beitscore_pp/calibrate.py` calibrates on.
It holds the training (`train`) and validation (`val`) pairs of VICR, Polaris, Nebula and EvalMuse-40K: 92,342 training and 17,618 validation pairs.
The ratings of each dataset were min-max scaled to [0, 1] with that dataset's rating range, and the scaled ratings of a pair averaged, so `human_scores` holds one target per pair; `dataset` names the source and `cluster` the image cluster (K = 63) used by cluster batching and the clustered calibration.
Polaris, Nebula and EvalMuse-40K images are stored in the dataset, COCO images are downloaded from cocodataset.org, and Flickr8K images come from the Flickr8K archive (downloaded automatically into `$BEITSCORE_DATA/flickr8k/Flicker8k_Dataset`, or pass `--flickr8k`); every image is checked against the SHA-256 of the file used in training.

### LongCapVLCP

```bash
python scripts/download_release.py longcapvlcp
```

This downloads `goncaloecgomes/LongCapVLCP` and writes `$BEITSCORE_DATA/longcapvlcp/longcapvlcp.pkl`, with its images in `$BEITSCORE_DATA/longcapvlcp/images/` (all stored in the dataset).
It holds 2,924 pairs of a long image description and an incorrect version of it, from ImageInWords, DOCCI and TextCaps.
The release leaves out 6 pairs of the version used in the BEiTScore paper whose incorrect description was identical to the correct one, so results can differ slightly from the published ones.

### Evaluation benchmarks

The benchmarks are not redistributed here; they are downloaded from their authors and converted with `scripts/prepare_data/benchmarks.py`.
Each one ends up in `$BEITSCORE_DATA/<name>/<name>.pkl`, with its images in `$BEITSCORE_DATA/<name>/images/`, where `<name>` is the name passed to the evaluation scripts: vicr, polaris, nebula, flickr8k-expert, flickr8k-cf, composite, genai, nocaps-foil, valse, sugarcrepe, winoground, vela (and longcapvlcp, from `download_release.py`, see above). CapArena goes to `$BEITSCORE_DATA/caparena/`, read by `scripts/beitscore_v1/caparena.py`.

1. VICR has to be downloaded by hand, because OpenReview only serves it to a browser.
   Open https://openreview.net/forum?id=xKYtTmtyI2, download the supplementary material (a zip of about 200 MB), and copy its `Dataset/VICR.csv` to `$BEITSCORE_DATA/vicr/VICR.csv`:

   ```bash
   mkdir -p $BEITSCORE_DATA/vicr
   unzip -j <downloaded zip> Dataset/VICR.csv -d $BEITSCORE_DATA/vicr
   ```

2. Download and convert all the benchmarks of both papers (VICR is converted from the file of step 1):

   ```bash
   python scripts/prepare_data/benchmarks.py download
   ```

   Downloads go to `$BEITSCORE_DATA/downloads/`, which is deleted at the end once every benchmark is done, so no data is kept twice.
   Winoground is gated on Hugging Face: accept its terms on the dataset page and run `huggingface-cli login` first.

   LongCap-Arena (`vela`), the long-caption benchmark of VELA (Matsuda et al., EMNLP 2025), is one of them: its ratings come from the CSV files of the authors' repository (github.com/Ka2ukiMatsuda/VELA) and the images of its two test sets from the authors' Hugging Face copy (`Ka2ukiMatsuda/LongCap-Arena`), or from `sa_000138.tar` of SA-1B with `--archive vela_images=<file>`.

3. Check sizes, ratings per pair and missing images against the papers:

   ```bash
   python scripts/prepare_data/benchmarks.py check
   ```

Every human rating of an image-caption pair is kept, and the correlation repeats the metric score once per rating, as in CLIPScore, so the numbers are comparable with the papers.
[docs/benchmarks.md](docs/benchmarks.md) gives the source of each benchmark and the choices that follow the papers (VALSE validated instances, Composite score).

## Quick start

```python
from beitscore import BEiTScore, BEiTScorePP

images = ["dog.jpg", "street.jpg"]                      # paths or PIL images
captions = ["A dog catches a frisbee.", "A red bus on a busy street."]

BEiTScore.from_pretrained("large").score(images, captions)        # array of scores in [0, 1]

out = BEiTScorePP.from_pretrained("large").score(images, captions)
out["median"], out["low"], out["high"]                            # calibrated 90% interval
```

`BEiTScorePP.score(..., calibrated=False)` gives the raw intervals and `return_quantiles=True` all 11 outputs.

## Repository layout

| Path | Content |
|---|---|
| `beitscore/` | The package: BEiT-3 wrapper and heads, data loading, training loop, metrics, calibration, scoring API |
| `scripts/download_release.py`, `scripts/get_images.py` | Download the models and data; get and check the images |
| `scripts/prepare_data/` | Download the evaluation benchmarks from their authors and convert them ([docs/benchmarks.md](docs/benchmarks.md)) |
| `scripts/train.py` | Training: `--benchmark stage1` (BEiTScore Stage 1), `regression` (BEiTScore Stage 2), `quant` (BEiTScore++) |
| `scripts/evaluate.py` | Correlation with human ratings, for both metrics, including LongCap-Arena |
| `baselines/` | Other reference-free metrics (CLIPScore, PAC-S++, SPECS, FLEUR, EXPERT, VQAScore) on the same benchmarks, and the comparison table (see [Baselines](#baselines)) |
| `scripts/beitscore_v1/` | Incorrect-caption detection and compositionality (`evaluate_errors.py`), CapArena (`caparena.py`) |
| `scripts/beitscore_pp/calibrate.py` | Conformal calibration of a BEiTScore++ model |
| `scripts/model_cost.py` | Parameters, GFLOPs and latency |
| `configs/runs/` | The training command of every model in the papers |
| `docs/` | [benchmarks.md](docs/benchmarks.md) (preparing the evaluation benchmarks) and [data_and_outputs.md](docs/data_and_outputs.md) (columns of the training data, outputs of the models and scripts) |

## Training

Training starts from Microsoft's BEiT-3 checkpoints. Download them from https://github.com/microsoft/unilm/tree/master/beit3 into `$BEITSCORE_MODELS`: `beit3_{base,large}_indomain_patch16_480_vqa.pth` (and `beit3_{base,large}_indomain_patch16_224_nlvr2.pth` for the NLVR2 ablation).

| Model | Command |
|---|---|
| BEiTScore, Stage 1 | `bash configs/runs/beitscore_stage1_large.sh` (reads `Stage1/stage1.pkl`) |
| BEiTScore, Stage 2 | `STAGE1_RUN=<run> bash configs/runs/beitscore_stage2_large.sh` |
| BEiTScore++ | `bash configs/runs/beitscorepp_large.sh` (starts from the downloaded BEiTScore weights) |
| BEiTScore++ ablations | `bash configs/runs/beitscorepp_large_ablation_*.sh` |

Every run is written to `$BEITSCORE_OUTPUT/finetuning/<task>/run-<n>/`, with `checkpoint-best.pth` selected on the validation split; the test evaluation uses that checkpoint.
Without flags, `--benchmark quant` trains the BEiTScore++ paper model: chained head, centered interval loss, cluster batching, bias offset −2.5, patience 15.

After training a BEiTScore++ model, calibrate it before evaluating its intervals:

```bash
python scripts/beitscore_pp/calibrate.py --model_size large --use_run <run>
```

The 90% interval is calibrated with conformal risk control in its upper-confidence-bound form (Bates et al., 2021): a Hoeffding bound with δ = 0.05, so that the miscoverage of the calibrated intervals is at most 10% with probability at least 95%, under exchangeability with the calibration data. `scripts/beitscore_pp/calibrate.py --bound` also accepts `bentkus` and `hb` (the minimum of the Hoeffding and Bentkus bounds); in every cluster, n in the bound is the cluster size.
Calibration searches a scale ω and a bias β (Equations 3 and 4 of the paper) inside each image cluster of the calibration set, and keeps the most conservative pair.
`--no_clustering` calibrates over the whole set; `--strict` also accounts for the search over β in the bound.
The result is written to `calibration.json`, which the evaluation and the scoring API read.

## Evaluation

| Benchmarks | Command |
|---|---|
| Correlation with human ratings: VICR, Polaris, Nebula, Flickr8K-Expert, Flickr8K-CF, Composite, GenAI-Bench | `python scripts/evaluate.py --model_size large --benchmark vicr` (BEiTScore) or `... --quant --calibration auto` (BEiTScore++); benchmark names: vicr, polaris, nebula, flickr8k-expert, flickr8k-cf, composite, genai |
| Incorrect captions and compositionality: nocaps-FOIL, VALSE, SugarCrepe, Winoground, LongCapVLCP | `python scripts/beitscore_v1/evaluate_errors.py --model_size large --benchmark valse` (benchmark names: nocaps-foil, valse, sugarcrepe, winoground, longcapvlcp) |
| LongCap-Arena (VELA): TestA and TestB | `python scripts/evaluate.py --model_size large --benchmark vela` (BEiTScore) or `... --quant --calibration auto` (BEiTScore++, its median) |
| CapArena | `python scripts/beitscore_v1/caparena.py --model_size large` |
| Model cost | `python scripts/model_cost.py --device cuda --timing` |

The scripts evaluate the downloaded models by default; `--use_run <n>` evaluates a model you trained, and `--checkpoint <path>` any checkpoint file.
They write the raw scores to `$BEITSCORE_OUTPUT/scores/<model id>/<benchmark>_test.pkl` and the metrics to `$BEITSCORE_OUTPUT/results/<model id>/<benchmark>.csv`.
For BEiTScore++, the interval metrics are coverage, sharpness, UPS and the quantile ECE.
Human ratings are mapped to [0, 1] with each benchmark's rating scale; `--normalization observed` uses each split's own range instead, as for the published BEiTScore++ tables (correlations are the same either way).
Ablations of BEiTScore: `--without_stage1`, `--without_stage2`, `--pretrained nlvr2`.

LongCap-Arena rates every caption on descriptiveness, relevance and fluency.
`--benchmark vela` reports TestA (`dci_val`, 98 pairs) and TestB (`dci_test`, 108 pairs) separately, against each rating and against three single scores that combine them (`beitscore/vela.py`): the projection on the first principal component of the ratings (`pca`), their harmonic mean weighted by that component (`pca_harmonic_mean`), and the average of that harmonic mean and the lowest rating (`pca_harmonic_minimum`).
The principal component is fitted on the ratings of the train and val groups only; `--vela_targets` selects the targets.

### Data files

The scripts read these files under `$BEITSCORE_DATA`.

| Data | File | Columns |
|---|---|---|
| Correlation benchmarks (vicr, polaris, nebula, flickr8k-expert, flickr8k-cf, composite, genai) | `<name>/<name>.pkl` | `image_path`, `caption`, `human_scores` (list of every rating of the pair), `subset` |
| Incorrect-caption benchmarks (nocaps-foil, valse, sugarcrepe) | `<name>/<name>.pkl` | `image_path`, `caption`, `foil` (the incorrect caption), `data_type` |
| Winoground | `winoground/winoground.pkl` | `image_path`, `caption`, `foil`, `pair_id` prefixed with the two-digit image-caption tag |
| LongCapVLCP | `longcapvlcp/longcapvlcp.pkl` | `image_path`, `caption`, `foil` (the incorrect description), `data_type` (error type), `dataset` (IIW, DOCCI, DOCCI-IIW or TextCaps) |
| LongCap-Arena (vela) | `vela/vela.pkl` | `image_path`, `caption`, `desc`, `rel`, `flu` (the three ratings, in [0, 1]), `subset` (train, val, dci_val or dci_test), `data_type` (TestA for dci_val, TestB for dci_test), `imgid`, `refs` |
| Stage 1 | `Stage1/stage1.pkl` | `image_path`, `caption`, `foil` (stored incorrect caption), `tokenized_words`, `pos_tagging`, `retrieval_{hard,medium,easy}_foil_list`, `subset` (see [docs/data_and_outputs.md](docs/data_and_outputs.md)) |
| Stage 2 (BEiTScore Stage 2, BEiTScore++, calibration) | `stage2/stage2.pkl` | `image_path`, `candidates`, `human_scores` (one target in [0, 1]), `dataset`, `subset` (train or val), `cluster` |

`image_path` is relative to `$BEITSCORE_DATA`.
[docs/data_and_outputs.md](docs/data_and_outputs.md) describes every column of the Stage-1 and Stage-2 files, the outputs of the scoring API (including the 11 quantiles of BEiTScore++), and the score, metric, calibration and training files the scripts write.

## Baselines

[baselines/](baselines/README.md) runs other reference-free captioning metrics on the same benchmark files as BEiTScore and builds a LaTeX table that compares them with BEiTScore and BEiTScore++.
None of their code or weights is redistributed here: each setup script clones the authors' repository and downloads their checkpoints from the authors' links.
The metrics need incompatible library versions, so each family has its own folder, environment and runner.

| Metric | Runner | Environment | Code | Weights |
|---|---|---|---|---|
| CLIPScore (B, L) | `baselines/encoders/run_encoders.py` | this repository's | (in `baselines/encoders/`) | LAION CLIP ViT-B/32 and ViT-L/14, Hugging Face |
| PAC-S++ (B, L) | `baselines/encoders/run_encoders.py` | this repository's | github.com/aimagelab/pacscore | `PAC++_clip_ViT-{B-32,L-14}.pth`, the PAC-S README links |
| SPECS | `baselines/encoders/run_encoders.py` | this repository's | github.com/mbzuai-nlp/SPECS | `spec.pt`, Hugging Face `Xiaohud/SPECS` |
| BEiTScore, BEiTScore++ (B, L) | `baselines/encoders/run_encoders.py` | this repository's | this repository | the release (`scripts/download_release.py models`) |
| FLEUR | `baselines/fleur/run_fleur.py` | `baselines/fleur/.venv` (LLaVA 1.1.3) | github.com/Yebin46/FLEUR | LLaVA-1.5-13B, Hugging Face |
| EXPERT | `baselines/expert/run_expert.py` | `baselines/expert/.venv` (LLaVA 1.2.2) | github.com/hjkim811/EXPERT | LLaVA-1.5-13B and the EXPERT LoRA, Hugging Face |
| VQAScore | `baselines/vqascore/run_vqascore.py` | `baselines/vqascore/.venv` (t2v_metrics 1.2) | PyPI `t2v_metrics` | CLIP-FlanT5-XXL, Hugging Face |

The Hugging Face weights are downloaded on first use; set `HF_HOME` to a folder with room for them (LLaVA-1.5-13B alone takes about 26 GB).

**1. Download the code and checkpoints, and create the environments (once).**
Run from the repository root, with this repository's environment active and `source configs/paths.sh` done:

```bash
bash baselines/encoders/setup.sh     # PAC-S++ and SPECS code and checkpoints, in this environment
bash baselines/fleur/setup.sh        # each LLM-based metric gets its own virtual environment (Python 3.10)
bash baselines/expert/setup.sh
bash baselines/vqascore/setup.sh
```

If the PyTorch build pip installs does not match your GPU driver, pass its index: `TORCH_INDEX=https://download.pytorch.org/whl/cu118 bash baselines/fleur/setup.sh`.

**2. Score the benchmarks.**
The benchmarks must have been downloaded (see [Evaluation benchmarks](#evaluation-benchmarks)).
The encoders runner goes first: it also turns the benchmark files into the items every runner scores.

```bash
BENCH="vela vicr polaris nebula flickr8k-expert composite"
python baselines/encoders/run_encoders.py --benchmarks $BENCH
baselines/fleur/.venv/bin/python baselines/fleur/run_fleur.py --benchmarks $BENCH
baselines/expert/.venv/bin/python baselines/expert/run_expert.py --benchmarks $BENCH
baselines/vqascore/.venv/bin/python baselines/vqascore/run_vqascore.py --benchmarks $BENCH
```

Every runner takes `--benchmarks` (`all` or names), `--models` (`all` or a subset of its metrics), `--recompute`, `--device` and `--batch_size`.
Scores are saved in `$BEITSCORE_OUTPUT/baselines/scores/` and reused by later runs, so an interrupted run resumes.

**3. Build the table.**

```bash
python baselines/make_table.py --benchmarks $BENCH --models all
```

It computes every metric from the saved scores (nothing is scored), writes them to `$BEITSCORE_OUTPUT/baselines/metrics/` and the table to `$BEITSCORE_OUTPUT/baselines/table.tex`, and lists any metric and benchmark without scores.
For LongCap-Arena, `--vela_targets` chooses the human scores (default `pca pca_harmonic_mean pca_harmonic_minimum`; also `desc rel flu`), and `--vela_paper without_refs with_refs` adds the values the VELA paper reports for VELA on the same test sets, quoted and not ranked.
For example, descriptiveness and the weighted harmonic mean:

```bash
python baselines/make_table.py --benchmarks vela --vela_targets desc pca_harmonic_mean --vela_paper without_refs with_refs
```

[baselines/README.md](baselines/README.md) has the details: the exact versions, how FLEUR and EXPERT reproduce the original setups, and how to check them against their published numbers.

## Errata (BEiTScore++ paper)

- The released `calibration.json` files apply the bound inside each image cluster with that cluster's size as n, as the per-cluster guarantee requires; the published clustered results used the size of the whole calibration set, so the released intervals are wider and their coverage higher than in the paper's tables.
- Equation 4 of the paper should read û = min(1, (u − m)·ω + m + β); the code uses the correct form.
- The regression head has one hidden layer of 216 units, not two layers of 256.
- Both backbones use patch size 16 at 480 × 480 pixels; "B/32" and "L/14" in Table 2 and Figure 6 refer to BEiTScore++-B and BEiTScore++-L.
- The bias offset of −2.5 is probit-scaled rather than subtracted uniformly: the lower heads start between 1.58 and 2.5 above the median head's bias and the upper heads between 1.58 and 2.5 below it, so the initial intervals are narrow (see `quantile_bias_offsets` in `beitscore/models/checkpoint.py`).

## Citation

```bibtex
@inproceedings{gomes-etal-2026-beitscore-pp,
  title     = {{BEiTScore++}: Reference-free Image Captioning Evaluation with {BEiT}-3 and Conformalized Quantile Regression},
  author    = {Gomes, Gon{\c{c}}alo and Martins, Bruno and Zerva, Chrysoula},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2026},
  year      = {2026},
  pages     = {46192--46226},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2026.findings-emnlp.2160/}
}

@article{gomes2026beitscore,
  title   = {{BEiTScore}: Reference-free Image Captioning Evaluation with an Efficient Cross-Encoder Model},
  author  = {Gomes, Gon{\c{c}}alo and Martins, Bruno and Zerva, Chrysoula},
  journal = {arXiv preprint arXiv:2605.21728},
  year    = {2026}
}
```

## Acknowledgements

The model code builds on BEiT-3 (https://github.com/microsoft/unilm/tree/master/beit3, MIT License).
Funding acknowledgements are listed in the papers.

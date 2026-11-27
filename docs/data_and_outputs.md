# Data files and outputs

This page describes the columns of the training data (Stage 1 and Stage 2) and everything BEiTScore and BEiTScore++ produce: the scoring API, the score and metric files of the evaluation scripts, the calibration file and the training runs.
The evaluation benchmarks are described in [benchmarks.md](benchmarks.md).
Paths use the variables of `configs/paths.sh`: `$BEITSCORE_DATA`, `$BEITSCORE_MODELS` and `$BEITSCORE_OUTPUT`.

## Training data

Both tables are pandas DataFrames saved with pickle; read them with `pandas.read_pickle` or `joblib.load`.
They are written by `scripts/download_release.py` from the release tables on the Hugging Face Hub (`stage1.parquet` and `stage2.parquet`, which hold an `image_id` in place of the image path), once the images are on disk.
Rows whose image could not be obtained are dropped, and the script prints how many.

### Stage 1: `$BEITSCORE_DATA/Stage1/stage1.pkl`

One row per image and correct caption, with the incorrect captions it can be contrasted with.
BEiTScore is trained in Stage 1 to score the correct caption above an incorrect one (`scripts/train.py --benchmark stage1`).

| Column | Content |
|---|---|
| `image_path` | Absolute path of the image under `$BEITSCORE_DATA/Stage1/images/` |
| `image_id` | Identifier of the image in the release (`images_index.parquet`) |
| `caption` | The correct caption |
| `foil` | The stored incorrect caption: written by an LLM from the correct one, or the dataset's own incorrect caption |
| `original_foil` | FOIL-it only: the human-written incorrect caption of FOIL-it, kept next to the LLM one (not used by the released checkpoints; `--use_original_foil` in training) |
| `tokenized_words`, `pos_tagging` | The words of the correct caption and their part-of-speech tags, used to build an incorrect caption on the fly by repeating, swapping or shuffling words |
| `retrieval_hard_foil_list`, `retrieval_medium_foil_list`, `retrieval_easy_foil_list` | Captions of other images retrieved at three similarity levels (hard, medium, easy), used as incorrect captions; `<empty>` marks a level with no neighbour |
| `subset` | `train`, `val` or `test`, split at the image level |

At each training step one incorrect caption is drawn at random among the word-level one, `foil` and one caption of each non-empty retrieval level (see `beitscore/stage1/dataset.py`).

### Stage 2: `$BEITSCORE_DATA/stage2/stage2.pkl`

One row per image-caption pair rated by humans, from the official training and validation splits of VICR, Polaris, Nebula and EvalMuse-40K.
It trains BEiTScore in Stage 2 (`--benchmark regression`) and BEiTScore++ (`--benchmark quant`), and its validation split is the calibration set of BEiTScore++ (`scripts/beitscore_pp/calibrate.py`).

| Column | Content |
|---|---|
| `image_path` | Absolute path of the image under `$BEITSCORE_DATA/stage2/images/` (Flickr8K images under `$BEITSCORE_DATA/flickr8k/`) |
| `image_id` | Identifier of the image in the release |
| `candidates` | The caption |
| `human_scores` | One target in [0, 1]: the ratings of the pair, each min-max scaled with its dataset's rating range, then averaged |
| `dataset` | Source dataset: VICR, Polaris, Nebula or EvalMuse-40K |
| `subset` | `train` (92,342 pairs) or `val` (17,618 pairs) |
| `cluster` | k-means cluster of the image (K = 63, from 0 to 62), used by cluster batching and the clustered calibration |

## Scoring API

```python
from beitscore import BEiTScore, BEiTScorePP

scores = BEiTScore.from_pretrained("large").score(images, captions)
out = BEiTScorePP.from_pretrained("large").score(images, captions, return_quantiles=True)
```

`images` are file paths or PIL images, `captions` strings; captions longer than 256 BPE tokens are truncated.

| Model | Output |
|---|---|
| `BEiTScore.score` | NumPy array of shape (n,): one quality score per pair, in [0, 1] (higher is better) |
| `BEiTScorePP.score` | dict with `median`, `low` and `high`, each of shape (n,): the predicted median of the human rating and the bounds of its 90% interval, calibrated with the model's `calibration.json` |
| `BEiTScorePP.score(..., calibrated=False)` | the same, with the raw interval of the model |
| `BEiTScorePP.score(..., return_quantiles=True)` | also `quantiles`, shape (n, 11): the raw outputs, sorted by quantile level (table below) |

The 11 outputs of BEiTScore++ are quantiles of the human rating, in increasing order of level:

| Column | `q00` | `q01` | `q02` | `q03` | `q04` | `q05` | `q06` | `q07` | `q08` | `q09` | `q10` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Level | 0.05 | 0.075 | 0.10 | 0.125 | 0.15 | 0.50 | 0.85 | 0.875 | 0.90 | 0.925 | 0.95 |

They are the bounds of the central 90%, 85%, 80%, 75% and 70% intervals around the median.
The reported interval is the 90% one: `low` = `q00`, `median` = `q05`, `high` = `q10`.
The chained construction makes the outputs non-decreasing, so the quantiles never cross.
Calibration changes only the bounds (Equations 3 and 4 of the BEiTScore++ paper, with ω and β from `calibration.json`):

```
low  = max(0, (q00 - q05) · ω + q05 - β)
high = min(1, (q10 - q05) · ω + q05 + β)
```

## Evaluation outputs

Every evaluation script writes two files, named after the model (`<model id>`) and the benchmark:

- `$BEITSCORE_OUTPUT/scores/<model id>/<benchmark>_test.pkl`: the raw scores, one row per instance.
- `$BEITSCORE_OUTPUT/results/<model id>/<benchmark>.csv`: the metrics.

`<model id>` is `beitscore_<size>_<pretrained>_<variant>` for the downloaded BEiTScore (for example `beitscore_large_vqa_with_stage1`) and `beitscorepp_<size>` for the downloaded BEiTScore++; `beitscore_<size>_run-<n>` and `beitscorepp_<size>_run-<n>` for runs of your own (`--use_run <n>`); and `<folder>_<file name>` for a checkpoint given with `--checkpoint`.

### Score files (`scores/<model id>/<benchmark>_test.pkl`)

| Column | Content |
|---|---|
| `image_path`, `candidates`, `data_type` | The image, the caption and the subset of the instance |
| `human_scores` | Correlation benchmarks: the ratings of the pair, mapped to [0, 1] (see `--normalization` in `scripts/evaluate.py`) |
| `score` | BEiTScore: the score; error benchmarks evaluated with BEiTScore++: its median |
| `q00` to `q10` | BEiTScore++: the 11 raw outputs (uncalibrated) |
| `pair_id`, `foil` | Error benchmarks: the pair an instance belongs to, and whether its caption is the incorrect one |
| `tag_combination` | Winoground: the image-caption tag (`00`, `01`, `10` or `11`) |
| `subset`, `desc`, `rel`, `flu` | LongCap-Arena: the group and the three ratings |

### Metric files (`results/<model id>/<benchmark>.csv`)

Correlation benchmarks (`scripts/evaluate.py`), one row per subset (`data_type`):

| Column | Content |
|---|---|
| `benchmark`, `data_type`, `n` | Benchmark, subset and number of pairs |
| `tau_b`, `tau_c`, `rho` | Kendall's τb and τc and Spearman's ρ with the human ratings, x100; every rating of a pair is paired with its score (CLIPScore protocol) |
| `coverage` | BEiTScore++: percentage of pairs whose mean rating lies in the interval |
| `sharpness`, `sharpness_std` | BEiTScore++: mean and standard deviation of the interval width |
| `ups`, `uss`, `uks` | BEiTScore++: Pearson, Spearman and Kendall correlations (x100) between the absolute error of the median and the interval width |
| `mae` | BEiTScore++: mean absolute error of the median against the mean rating |
| `ece` | BEiTScore++: quantile calibration error, the mean over the 11 outputs of the gap between the observed and nominal coverage of each quantile |
| `bimodality_low`, `bimodality_high` | BEiTScore++: bimodality coefficient of the lower and upper bounds (above 0.555, the bounds collapse to the extremes) |

LongCap-Arena (`--benchmark vela`) has one row per test set (`data_type`: TestA or TestB) and target (`target`), with `n`, `tau_b`, `tau_c` and `rho`.

Error benchmarks (`scripts/beitscore_v1/evaluate_errors.py`), one row per subset plus the overall rows `all (micro)` and, with several subsets, `all (macro accuracy)`:

| Column | Content |
|---|---|
| `data_type`, `n_pairs` | Subset and number of (correct, incorrect) pairs |
| `accuracy` | Fraction of pairs whose correct caption scores strictly above the incorrect one |
| `auroc`, `ap` | Area under the ROC curve and average precision of the score as a detector of the correct caption |

Winoground has one row with `text`, `image` and `group`, the fractions of examples solved under each criterion.
Unlike the correlations, these values are fractions in [0, 1].

CapArena (`scripts/beitscore_v1/caparena.py`) writes `caparena_annots_eval_beitscore_<size>.json` (the battles judged by the model) and `caparena_results_<size>.json` (the scores of the competition protocol).

## Calibration file (`calibration.json`)

Written by `scripts/beitscore_pp/calibrate.py` to `$BEITSCORE_OUTPUT/calibration/<model id>/` and released next to each BEiTScore++ checkpoint.

| Field | Content |
|---|---|
| `omega`, `beta` | Scale and bias applied to the interval (formula above) |
| `target_coverage`, `delta` | Coverage guaranteed (0.90) and the probability it may fail (0.05) |
| `bound` | Upper confidence bound: `hoeffding`, `bentkus` or `hb` |
| `strict` | Whether the bound also accounts for the search over β |
| `clustered`, `n_clusters`, `selected_cluster`, `min_cluster_size` | Clustered calibration: number of clusters, the cluster whose pair was kept (the most conservative) and the smallest cluster calibrated on its own |
| `n_calibration` | Size of the calibration set |
| `raw_coverage`, `raw_width`, `coverage`, `width` | Coverage and mean width on the calibration set, before and after calibration |
| `clusters` | Per cluster: `size`, its own `omega` and `beta`, coverage and width before (`raw_*`) and after calibration with its own pair (`coverage`, `width`) and with the pair kept (`*_selected`) |
| `model`, `cluster_file`, `cluster_file_sha256`, `format` | Provenance of the calibration |

## Released models (`$BEITSCORE_MODELS`)

| Folder | Files |
|---|---|
| `beitscore/<size>/` | `model.safetensors` (weights), `config.json` (head settings of the run) |
| `beitscorepp/<size>/` | `model.safetensors`, `config.json`, `calibration.json` |
| `beit3.spm` | The BEiT-3 sentencepiece model (tokenizer) |

## Training runs (`$BEITSCORE_OUTPUT/finetuning/<task>/run-<n>/`)

`scripts/train.py` writes each run to a new folder (`<task>` is `stage1`, `regression` or `quant`):

| File | Content |
|---|---|
| `arguments_used.txt` | Every argument of the run, one per line |
| `log.txt` | One JSON line per epoch: the training statistics (`train_*`), the validation metrics (`val_*`), `epoch` and `n_parameters` |
| `checkpoint-<epoch>.pth` | Checkpoint saved every `--save_ckpt_freq` epochs and at the last one |
| `checkpoint-best.pth` | Checkpoint with the best validation metric, used for the test evaluation and by `--use_run <n>` |

# Preparing the evaluation benchmarks

This guide downloads every evaluation benchmark of BEiTScore and BEiTScore++ from its authors and converts it into the file the evaluation reads.
All commands run from the repository root, after `pip install -e ".[data]"` and `source configs/paths.sh` (see `configs/paths.example.sh`).

## Benchmarks used in the papers

| Paper | Correlation with human judgments | Incorrect captions and compositionality |
|---|---|---|
| BEiTScore | VICR, Polaris, Nebula, Flickr8K-Expert, Flickr8K-CF, Composite, GenAI-Bench | nocaps-FOIL, VALSE, SugarCrepe, Winoground, LongCapVLCP, CapArena |
| BEiTScore++ | VICR, Polaris, Nebula, Flickr8K-Expert, Composite | |

## Where the files go

Each benchmark has one folder under `$BEITSCORE_DATA`, named after the benchmark, with one file of the same name and its images:

```
$BEITSCORE_DATA/
  vicr/vicr.pkl                          vicr/images/            vicr/VICR.csv (downloaded by hand, see below)
  polaris/polaris.pkl                    polaris/images/
  nebula/nebula.pkl                      nebula/images/
  flickr8k-expert/flickr8k-expert.pkl    flickr8k-expert/images/
  flickr8k-cf/flickr8k-cf.pkl            flickr8k-cf/images/
  composite/composite.pkl                composite/images/
  genai/genai.pkl                        genai/images/
  nocaps-foil/nocaps-foil.pkl            nocaps-foil/images/
  valse/valse.pkl                        valse/images/<source dataset>/
  sugarcrepe/sugarcrepe.pkl              sugarcrepe/images/
  winoground/winoground.pkl              winoground/images/
  longcapvlcp/longcapvlcp.pkl            longcapvlcp/images/     (scripts/download_release.py longcapvlcp)
  caparena/caparena_annots_eval.json     caparena/images/        (read by scripts/beitscore_v1/caparena.py)
```

These names are also the `--benchmark` values of `scripts/evaluate.py` (vicr, polaris, nebula, flickr8k-expert, flickr8k-cf, composite, genai) and `scripts/beitscore_v1/evaluate_errors.py` (nocaps-foil, valse, sugarcrepe, winoground, longcapvlcp).
Images shared by two benchmarks (Flickr8K) are hard links to the same files, so they take the space once.

## Step 1: VICR, by hand

VICR cannot be downloaded by a script, because OpenReview only serves its files to a browser.

1. Open https://openreview.net/forum?id=xKYtTmtyI2 (Narins et al., "Validated Image Caption Rating Dataset", NeurIPS 2023 Datasets and Benchmarks).
2. Download the supplementary material (the "Supplementary Material" link, a zip file of about 200 MB).
3. Unzip it and copy `Dataset/VICR.csv` to `$BEITSCORE_DATA/vicr/VICR.csv`:

   ```bash
   mkdir -p $BEITSCORE_DATA/vicr
   unzip -j <downloaded zip> Dataset/VICR.csv -d $BEITSCORE_DATA/vicr
   ```

`VICR.csv` has one row per image-caption pair: the image URL, the caption, the official split (`subset`: 10,175 train, 2,310 val, 3,161 test) and its 3 to 7 ratings (`ratings`, `rating_2`, ..., `rating_7`).
The download step below turns it into `vicr/vicr.pkl` with every rating of a pair in its `human_scores` list, and downloads its images (COCO val2014 from cocodataset.org, Flickr8K from the Flickr8K archive).
The rest of the supplementary material is not needed and can be deleted.

## Step 2: download and convert everything else

```bash
python scripts/prepare_data/benchmarks.py download                       # every benchmark of both papers
python scripts/prepare_data/benchmarks.py download valse composite       # or some of them
```

| Benchmark | Source | Images |
|---|---|---|
| VICR | `$BEITSCORE_DATA/vicr/VICR.csv` (step 1) | COCO val2014 from cocodataset.org, Flickr8K from the Flickr8K archive |
| Polaris, Nebula | Hugging Face `yuwd/Polaris`, `Ka2ukiMatsuda/Nebula` | in the dataset |
| Flickr8K-Expert, Flickr8K-CF | the two Google Drive archives used by CLIPScore (`flickr8k_example/download.py` of jmhessel/clipscore), processed the same way | in the archive |
| Composite | the AMT evaluation files of Aditya et al. (2015) (`AMT_eval.zip`, linked from the official page imagesdg.wordpress.com/image-to-scene-description-graph) | COCO val2014 from cocodataset.org, Flickr8K from the Flickr8K archive, Flickr30k from Kaggle |
| nocaps-FOIL | `data/nocaps-val-foil.json` of DavidMChan/aloha (2,500 pairs) | Open Images validation, one by one |
| VALSE | the 11 files of `data/` in Heidelberg-NLP/VALSE | COCO from cocodataset.org; Visual7W, SWiG and VisDial val2018 from their archives (only the needed files are extracted) |
| SugarCrepe | `data/*.json` of RAIVNLab/sugar-crepe | COCO val2017, one by one from cocodataset.org |
| GenAI-Bench | Hugging Face `BaiqiL/GenAI-Bench` | in the dataset |
| Winoground | Hugging Face `facebook/winoground` (gated: accept the terms on its page and `huggingface-cli login` first) | in the dataset |
| LongCapVLCP | `python scripts/download_release.py longcapvlcp` | in the release |
| CapArena | `caparena_annots_eval.json` from the links of njucckevin/CapArena | the DOCCI image archive of the same page (only the needed files are extracted) |

Everything is downloaded to `$BEITSCORE_DATA/downloads/` first, including the Hugging Face caches of Polaris, Nebula, GenAI-Bench and Winoground.
When every requested benchmark is done, that folder is deleted, so no data is kept twice; after a failure it is kept, so that the next run does not download it again (`--keep_downloads` always keeps it).
The Flickr30k images used by Composite stay in `$BEITSCORE_DATA/flickr30k/Images`, the folder the Stage-1 data also uses.
At the end the command prints one line per benchmark: done, failed (with the error), or manual (with what to do).
If an archive link stops working, download the file by hand and pass it with `--archive KEY=FILE` (keys: `composite`, `visual7w`, `SWiG`, `VisDial_v1.0`, `Flickr8k_Dataset.zip`, `Flickr8k_text.zip`, `caparena` and `caparena_images`, and `vicr` for a `VICR.csv` kept elsewhere).

Choices that follow the papers:

- VALSE keeps the 7,702 instances validated by the annotators (at least two of three chose the correct caption); `--keep_all` keeps all 8,782.
- Composite has 13,146 pairs over 3,995 images: three candidates per Flickr8K image and four per Flickr30k and COCO image, a fourth candidate identical to the third counted once.
  The score of a pair is the mean of its correctness and thoroughness ratings (the PAC-S authors also use both rating files, see aimagelab/pacscore issue 5); `--composite_score correctness` uses the correctness rating alone, as Kilickaya et al. (2017) describe.
  A pair rated twice keeps its first rating; `--composite_duplicates keep` keeps both.

## Step 3: check

```bash
python scripts/prepare_data/benchmarks.py check
```

It loads each benchmark the way the evaluation does, reports the test size, the mean number of ratings per pair and image files that cannot be found, and compares the correlation benchmarks with the sizes in the papers (VICR 3,161 pairs with 4.4 ratings each, Polaris 8,726 with 3.0, Nebula 3,298 with 1.0, Flickr8K-Expert 5,664 with 3.0, Composite 13,146 with 1.0).
A line marked `[check]` differs from the papers or has missing images.
`benchmarks.py compare --reference <other data folder>` compares the prepared benchmarks row by row with those of another data folder.

## How ratings are kept

Human ratings are never averaged in the benchmark files.
Every rating of an image-caption pair is kept in the `human_scores` list of that pair, and at evaluation time the metric score is repeated once per rating before Kendall's tau and Spearman's rho are computed (`beitscore.metrics.explode_pairs`), the protocol of CLIPScore (Hessel et al., 2021).
Averaging the ratings first gives different, higher correlations that are not comparable with the papers.

## GenAI-Bench and CapArena

Both are downloaded by `benchmarks.py download` with the other benchmarks of the BEiTScore paper.

- GenAI-Bench (Li et al., 2024) comes from the Hugging Face dataset `BaiqiL/GenAI-Bench`: 1,600 prompts, one image from each of six text-to-image models (9,600 pairs), with three 1-to-5 alignment ratings each. The images are written as `genai/images/<model>_<prompt index>.png`.
- CapArena (Cheng et al., 2025) comes from the links of github.com/njucckevin/CapArena: the human battle annotations (`caparena_annots_eval.json`, 6,523 pairwise battles) and the DOCCI images they use, in `$BEITSCORE_DATA/caparena/`. It is evaluated by its own script, `python scripts/beitscore_v1/caparena.py --model_size large`. If the box.nju.edu.cn links fail, download the two files by hand from that GitHub page and pass them with `--archive caparena=<annotation file> caparena_images=<image archive>`.

`ratings` reads CSV, TSV, JSON, JSONL, Parquet or pickle files, with one row per judgment or one row per pair whose score column holds a list of ratings; `pairs` reads a table of (image, correct caption, incorrect caption).

Each benchmark keeps the license of its authors; this repository redistributes none of them.

"""Download the evaluation benchmarks of BEiTScore and BEiTScore++ from their authors and
convert them into the files that beitscore.data.datasets.get_data reads. See docs/benchmarks.md.

    python scripts/prepare_data/benchmarks.py download                 # every benchmark of both papers (or: download all)
    python scripts/prepare_data/benchmarks.py download valse composite # or some of them
    python scripts/prepare_data/benchmarks.py check                    # sizes and missing images
    python scripts/prepare_data/benchmarks.py compare --reference DIR  # against another data folder

Every benchmark is written to $BEITSCORE_DATA/<name>/<name>.pkl, with its images in
$BEITSCORE_DATA/<name>/images/, where <name> is one of: vicr, polaris, nebula, flickr8k-expert,
flickr8k-cf, composite, genai, nocaps-foil, valse, sugarcrepe, winoground, vela. CapArena, which has its
own evaluation script (scripts/beitscore_v1/caparena.py), goes to $BEITSCORE_DATA/caparena/. Downloads go to $BEITSCORE_DATA/downloads/, which is
deleted once every requested benchmark is done (--keep_downloads keeps it).

vela is LongCap-Arena, the long-caption benchmark of VELA (Matsuda et al., EMNLP 2025).

VICR cannot be downloaded automatically: put Dataset/VICR.csv of its supplementary material
(https://openreview.net/forum?id=xKYtTmtyI2) in $BEITSCORE_DATA/vicr/VICR.csv first.

Manual conversions, for files downloaded by hand:
    python scripts/prepare_data/benchmarks.py ratings --name vicr --table FILE --image_col .. --caption_col .. --score_col .. --images DIR
    python scripts/prepare_data/benchmarks.py pairs --name nocaps-foil --table FILE --image_col .. --caption_col .. --incorrect_col .. --images DIR

Human ratings are never averaged here. Every rating of an image-caption pair is kept in the
`human_scores` list of that pair, and the evaluation repeats the metric score once per rating
before computing the correlation (the CLIPScore protocol, beitscore.metrics.explode_pairs).
Image paths are stored relative to $BEITSCORE_DATA.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from beitscore.paths import DATA_ROOT

from beitscore.data.datasets import BENCHMARKS, CORRELATION_BENCHMARKS, ERROR_BENCHMARKS, canonical_benchmark

# get_data reads $BEITSCORE_DATA/<name>/<name>.pkl
NAMES = list(BENCHMARKS)
CORRELATION = list(CORRELATION_BENCHMARKS)
ERRORS = list(ERROR_BENCHMARKS)

# test-split sizes reported in the papers (pairs, ratings per pair), used by `check`
EXPECTED = {
    "vicr": (3161, 4.4), "polaris": (8726, 3.0), "nebula": (3298, 1.0),
    "flickr8k-expert": (5664, 3.0), "composite": (13146, 1.0), "genai": (9600, 3.0),
}

# VELA evaluation groups (pairs rated on all three dimensions; the paper counts 294 and 324 ratings,
# three per pair)
VELA_EXPECTED = {"dci_val": 98, "dci_test": 108}

SPLIT_NAMES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
               "dev": "val", "test": "test"}


# --------------------------------------------------------------------------- helpers
def out_path(name, data_root=DATA_ROOT):
    """$BEITSCORE_DATA/<name>/<name>.pkl"""
    name = canonical_benchmark(name)
    return os.path.join(data_root, name, f"{name}.pkl")


def bench_dir(name, data_root=DATA_ROOT):
    """Folder of a benchmark relative to the data root: its standard name."""
    return canonical_benchmark(name)


def normalise_split(value):
    key = str(value).strip().lower()
    if key not in SPLIT_NAMES:
        raise ValueError(f"Unknown split name {value!r}; expected one of {sorted(SPLIT_NAMES)}")
    return SPLIT_NAMES[key]


def link_images(source, folder, data_root=DATA_ROOT, name="images"):
    """Make <data_root>/<folder>/<name> point to `source`; returns the relative prefix."""
    target = os.path.join(data_root, folder, name)
    source = os.path.abspath(source)
    if os.path.abspath(target) != source:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.islink(target):
            if os.path.realpath(target) != os.path.realpath(source):
                os.remove(target)
                os.symlink(source, target)
        elif os.path.exists(target):
            raise FileExistsError(f"{target} exists and is not a link; move it or pass --images {target}")
        else:
            os.symlink(source, target)
    return f"{folder}/{name}"


def as_ratings(value):
    """A cell of a score column as a list of floats: scalars, lists and strings such as
    "[1, 2, 2]" are accepted."""
    if isinstance(value, str):
        value = value.strip()
        value = ast.literal_eval(value) if value.startswith("[") else float(value)
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return [float(v) for v in value if not pd.isna(v)]
    return [] if pd.isna(value) else [float(value)]


def group_ratings(df, keys=("image_path", "caption")):
    """One row per image-caption pair with every rating of that pair in `human_scores`.
    Other columns keep their first value."""
    keys = list(keys)
    df = df.copy()
    df["human_scores"] = df["human_scores"].map(as_ratings)
    agg = {c: "first" for c in df.columns if c not in keys + ["human_scores"]}
    agg["human_scores"] = lambda s: [r for ratings in s for r in ratings]
    out = df.groupby(keys, sort=False).agg(agg).reset_index()
    out = out[out["human_scores"].map(len) > 0].reset_index(drop=True)
    out["candidates"] = out["caption"]
    return out


def save(df, name, data_root=DATA_ROOT):
    name = canonical_benchmark(name)
    path = out_path(name, data_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(df.reset_index(drop=True), path)
    summary = df.groupby("subset").size().to_dict() if "subset" in df.columns else {"all": len(df)}
    print(f"{name}: {len(df)} rows {summary} -> {path}")
    if "human_scores" in df.columns:
        n = df["human_scores"].map(len)
        print(f"{name}: {n.sum()} ratings, {n.mean():.2f} per pair (min {n.min()}, max {n.max()})")
    return path


def read_table(path):
    path = str(path)
    if path.endswith((".csv", ".tsv")):
        return pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):  # {id: record} or {"annotations": [...]}
            data = next((v for v in data.values() if isinstance(v, list)), None) or \
                [dict(v, _key=k) for k, v in data.items()]
        return pd.DataFrame(data)
    if path.endswith((".pkl", ".pickle")):
        return joblib.load(path)
    raise ValueError(f"Unsupported table format: {path}")


def save_image(image_feature, folder, data_root=DATA_ROOT):
    """Write a Hugging Face image (decode=False dict with bytes/path) once; returns the
    path relative to the data root. Files are named by the hash of their bytes so the same
    picture is stored once however many rows use it."""
    data = image_feature["bytes"]
    if data is None:
        with open(image_feature["path"], "rb") as f:
            data = f.read()
    ext = os.path.splitext(image_feature.get("path") or "")[1].lower() or ".jpg"
    rel = f"{folder}/images/{hashlib.sha1(data).hexdigest()[:20]}{ext}"
    full = os.path.join(data_root, rel)
    if not os.path.exists(full):
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(data)
    return rel


def load_hf(repo, token=None, cache_dir=None):
    from datasets import Image, load_dataset

    ds = load_dataset(repo, token=token, cache_dir=cache_dir)
    for split in ds:
        for col, feat in ds[split].features.items():
            if isinstance(feat, Image):
                ds[split] = ds[split].cast_column(col, Image(decode=False))
    return ds


# --------------------------------------------------------------------------- correlation benchmarks
def polaris_frame(splits, data_root=DATA_ROOT):
    """splits: {split name: iterable of records with img, cand, refs, human_score}.
    Polaris has one row per judgment (about three per pair)."""
    rows = []
    for split, records in splits.items():
        for r in records:
            rows.append({"image_path": save_image(r["img"], "polaris", data_root), "caption": r["cand"],
                         "refs": list(r["refs"]), "human_scores": r["human_score"],
                         "subset": normalise_split(split), "data_type": "polaris"})
    return group_ratings(pd.DataFrame(rows), keys=("subset", "image_path", "caption"))


def nebula_frame(splits, data_root=DATA_ROOT):
    """splits: {split name: iterable of records with image, mt, refs, human_score}."""
    rows = []
    for split, records in splits.items():
        for r in records:
            rows.append({"image_path": save_image(r["image"], "nebula", data_root), "caption": r["mt"],
                         "refs": list(r["refs"]), "human_scores": r["human_score"],
                         "subset": normalise_split(split), "data_type": "nebula"})
    return group_ratings(pd.DataFrame(rows), keys=("subset", "image_path", "caption"))


# GenAI-Bench (Li et al., 2024): 1,600 prompts, one image from each of six text-to-image models,
# three 1-to-5 ratings of prompt-image alignment per image (Hugging Face BaiqiL/GenAI-Bench).
GENAI_REPO = "BaiqiL/GenAI-Bench"
GENAI_MODELS = ["DALLE_3", "DeepFloyd_I_XL_v1", "Midjourney_6", "SDXL_2_1", "SDXL_Base", "SDXL_Turbo"]


def genai_ratings(record, model):
    """The ratings of one model's image: HumanRatings[model], or the <model>_HumanRating field."""
    if isinstance(record.get("HumanRatings"), dict):
        value = record["HumanRatings"][model]
    else:
        value = record[f"{model}_HumanRating"]
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return [float(v) for v in value]


def genai_frame(records, data_root=DATA_ROOT):
    """One row per (prompt, model) with the three ratings. The image of row i and model m is
    written as genai/images/<m>_<i>.png, i being the position of the prompt (0 to 1599)."""
    import io

    from PIL import Image

    rows = []
    for i, r in enumerate(records):
        for m in GENAI_MODELS:
            rel = f"genai/images/{m}_{i}.png"
            full = os.path.join(data_root, rel)
            if not os.path.exists(full):
                os.makedirs(os.path.dirname(full), exist_ok=True)
                feature = r[m]
                data = feature["bytes"] if feature.get("bytes") is not None else open(feature["path"], "rb").read()
                Image.open(io.BytesIO(data)).save(full, format="PNG")
            rows.append({"image_path": rel, "caption": r["Prompt"], "human_scores": genai_ratings(r, m),
                         "model": m, "tags": r.get("Tags"), "subset": "test", "data_type": "genai"})
    return pd.DataFrame(rows)


# CapArena (Cheng et al., 2025): 6,523 pairwise battles between detailed captions of DOCCI images,
# from the links of github.com/njucckevin/CapArena. Evaluated by scripts/beitscore_v1/caparena.py.
CAPARENA_ANNOTS_URL = "https://box.nju.edu.cn/f/0fd0a0d3dce243ab8c12/?dl=1"
CAPARENA_IMAGES_URL = "https://box.nju.edu.cn/f/9d2b9ded47d54999926c/?dl=1"
CAPARENA_ANNOTS = "caparena_annots_eval.json"


def caparena_files(data_root, downloads, archives):
    """$BEITSCORE_DATA/caparena/caparena_annots_eval.json and the images its battles use, in
    $BEITSCORE_DATA/caparena/images/. --archive caparena=FILE (the annotation file, or an archive
    holding it) and caparena_images=FILE (an archive of the images) replace the downloads."""
    import zipfile

    folder = os.path.join(data_root, "caparena")
    annots = os.path.join(folder, CAPARENA_ANNOTS)
    if not os.path.exists(annots):
        src = archives.get("caparena") or http_download(CAPARENA_ANNOTS_URL,
                                                        os.path.join(downloads, "caparena", "annotations"))
        if zipfile.is_zipfile(src) or src.endswith((".tar", ".tar.gz", ".tgz")):
            if CAPARENA_ANNOTS not in extract(src, folder, names={CAPARENA_ANNOTS}):
                raise FileNotFoundError(f"{CAPARENA_ANNOTS} not found in {src}")
        else:
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(src, annots)
    with open(annots) as f:
        needed = {r["img"] for r in json.load(f)}
    images = os.path.join(folder, "images")
    missing = {n for n in needed if not os.path.exists(os.path.join(images, n))}
    if missing:
        archive = archives.get("caparena_images") or http_download(
            CAPARENA_IMAGES_URL, os.path.join(downloads, "caparena", "images_archive"))
        extract(archive, images, names=missing)
        still = sum(not os.path.exists(os.path.join(images, n)) for n in needed)
        if still:
            print(f"caparena: {still} of {len(needed)} images not found in {archive}")
    print(f"caparena: {annots} and {len(needed)} images in {images}")
    return annots


def flickr8k_frames(json_dir, images, data_root=DATA_ROOT):
    """flickr8k-expert and flickr8k-cf from the CLIPScore-format files (flickr8k.json,
    crowdflower_flickr8k.json): one row per image-caption pair with all its ratings, exactly
    what the evaluation used to build from the JSON files. The Flickr8K images are hard-linked
    into each benchmark's images/ folder. Returns {name: DataFrame}."""
    frames = {}
    for name, fname in (("flickr8k-expert", "flickr8k.json"), ("flickr8k-cf", "crowdflower_flickr8k.json")):
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
        rows = []
        for key, v in data.items():
            refs = [" ".join(gt.split()) for gt in v["ground_truth"]]
            for j in v["human_judgement"]:
                rows.append({"image_ids": key, "file": os.path.basename(j["image_path"]),
                             "caption": " ".join(j["caption"].split()), "refs": refs, "human_scores": j["rating"]})
        df = pd.DataFrame(rows)
        place(sorted(set(df["file"])), images, os.path.join(data_root, name, "images"))
        df["image_path"] = f"{name}/images/" + df["file"]
        df = group_ratings(df.drop(columns=["file"]), keys=("image_ids", "caption"))
        df["subset"], df["data_type"] = "test", name
        frames[name] = df
    return frames


def ratings_frame(table, image_col, caption_col, score_col, name, split_col=None, split=None,
                  type_col=None, images=None, data_root=DATA_ROOT, extra_cols=()):
    """Any table with one row per judgment, or one row per pair with a list of ratings."""
    df = read_table(table) if not isinstance(table, pd.DataFrame) else table
    out = pd.DataFrame({"image_path": df[image_col].astype(str), "caption": df[caption_col].astype(str),
                        "human_scores": df[score_col]})
    if images:
        prefix = link_images(images, bench_dir(name), data_root)
        out["image_path"] = [f"{prefix}/{p.lstrip('/')}" for p in out["image_path"]]
    out["subset"] = df[split_col].map(normalise_split) if split_col else normalise_split(split or "test")
    out["data_type"] = df[type_col].astype(str) if type_col else name
    for c in extra_cols:
        out[c] = df[c]
    return group_ratings(out, keys=("subset", "image_path", "caption"))


# --------------------------------------------------------------------------- error-detection benchmarks
def pairs_frame(rows, name, images=None, data_root=DATA_ROOT):
    """rows: DataFrame with image_path, caption (correct), foil (incorrect caption), data_type."""
    df = rows.copy()
    if images:
        prefix = link_images(images, bench_dir(name), data_root)
        df["image_path"] = [f"{prefix}/{str(p).lstrip('/')}" for p in df["image_path"]]
    if "data_type" not in df.columns:
        df["data_type"] = name
    return df[["image_path", "caption", "foil", "data_type"]].reset_index(drop=True)


def sugarcrepe_frame(json_dir, images, data_root=DATA_ROOT):
    """The seven JSON files of the SugarCrepe repository ({id: {filename, caption,
    negative_caption}}); the file name is the category."""
    rows = []
    for path in sorted(Path(json_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        for r in data.values():
            rows.append({"image_path": r["filename"], "caption": r["caption"], "foil": r["negative_caption"],
                         "data_type": path.stem})
    return pairs_frame(pd.DataFrame(rows), "sugarcrepe", images, data_root)


def valse_folder(dataset):
    """Image folder of a VALSE `dataset` value ("FOIL dataset" -> images/FOIL_dataset)."""
    return "images/" + re.sub(r"[^A-Za-z0-9.]+", "_", dataset)


def valse_frame(json_dir, image_roots, keep_all=False, data_root=DATA_ROOT):
    """The VALSE instrument files ({id: {dataset, image_file, caption, foil(s), mturk}}).
    By default only validated instances are kept (at least two of three annotators chose the
    correct caption). image_roots maps the `dataset` field to an image folder."""
    prefixes = {k: link_images(v, "valse", data_root, name=valse_folder(k)) for k, v in image_roots.items()}
    rows, skipped = [], 0
    for path in sorted(Path(json_dir).glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        for r in data.values():
            votes = (r.get("mturk") or {}).get("caption")
            if not keep_all and votes is not None and votes < 2:
                skipped += 1
                continue
            foil = r.get("foil") or (r.get("foils") or [None])[0]
            source = r.get("dataset")
            if source not in prefixes:
                raise KeyError(f"{path.name}: no image folder for dataset {source!r}; pass --images {source}=DIR")
            rows.append({"image_path": f"{prefixes[source]}/{r['image_file']}", "caption": r["caption"],
                         "foil": foil, "data_type": path.stem})
    print(f"valse: {skipped} instances not validated by the annotators were left out")
    return pairs_frame(pd.DataFrame(rows), "valse", None, data_root)


def winoground_frame(records, data_root=DATA_ROOT):
    """Four rows per example, one per (image i, caption j), tagged "ij" in pair_id as
    beitscore.metrics.winoground expects."""
    rows = []
    for r in records:
        paths = [save_image(r["image_0"], "winoground", data_root), save_image(r["image_1"], "winoground", data_root)]
        for i in (0, 1):
            for j in (0, 1):
                rows.append({"image_path": paths[i], "caption": r[f"caption_{j}"], "foil": i != j,
                             "pair_id": f"{i}{j}{r['id']}", "data_type": "winoground",
                             "collapsed_tag": r.get("collapsed_tag")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- check
def check(names, data_root=DATA_ROOT):
    """Load every prepared benchmark through get_data and report sizes, ratings per pair and
    missing images. Returns the number of problems found."""
    from beitscore.data.datasets import get_data

    problems = 0
    for name in names:
        path = out_path(name, data_root)
        if not os.path.exists(path):
            print(f"[missing] {name}: {path}")
            continue
        try:
            df, task, _ = get_data(name, data_root, "test")
        except Exception as exc:  # noqa: BLE001  (report and go on with the others)
            print(f"[error]   {name}: {exc!r}")
            problems += 1
            continue
        paths = df["image_path"].unique()
        missing = sum(not os.path.exists(p) for p in paths)
        before = problems
        line = f"{name}: {len(df)} test rows, {len(paths)} images"
        if task == "vela":
            sizes = df.groupby("subset", sort=False).size().to_dict()
            line += f", {sizes}"
            if sizes != VELA_EXPECTED:
                line += f" (paper: {VELA_EXPECTED})"
                problems += 1
        if task == "correlation":
            n = df["human_scores"].map(lambda s: len(s) if isinstance(s, (list, tuple, np.ndarray)) else 1)
            line += f", {n.mean():.2f} ratings per pair"
            if name in EXPECTED:
                exp_n, exp_r = EXPECTED[name]
                if len(df) != exp_n or abs(n.mean() - exp_r) > 0.15:
                    line += f" (paper: {exp_n} pairs, {exp_r} ratings per pair)"
                    problems += 1
        if missing:
            line += f", {missing} image files not found (e.g. {next(p for p in paths if not os.path.exists(p))})"
            problems += 1
        print(("[ok]      " if problems == before else "[check]   ") + line)
    return problems


# --------------------------------------------------------------------------- VELA (LongCap-Arena)
def vela_ratings(folder):
    """One row per image-caption pair rated on all three dimensions, for every group of
    VELA_GROUPS: the three CSV files of a group are joined on (imgid, cand), exactly."""
    from beitscore.vela import DIMS, GROUP_NAMES

    frames = []
    for group in VELA_GROUPS:
        tables = {d: pd.read_csv(os.path.join(folder, f"{d}_{group}.csv")) for d in DIMS}
        merged = None
        for d, t in tables.items():
            t = t.rename(columns={"score": d})
            keep = ["imgid", "cand", d] + (["refs"] if merged is None and "refs" in t.columns else [])
            merged = t[keep] if merged is None else merged.merge(t[["imgid", "cand", d]], on=["imgid", "cand"],
                                                                     how="inner", validate="one_to_one")
        merged["subset"] = group
        merged["data_type"] = GROUP_NAMES.get(group, group)
        sizes = ", ".join(f"{d} {len(t)}" for d, t in tables.items())
        print(f"vela {group}: {sizes} rated captions, {len(merged)} rated on all three")
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True).rename(columns={"cand": "caption"})
    if "refs" in df.columns:
        df["refs"] = df["refs"].map(lambda r: ast.literal_eval(r) if isinstance(r, str) and r.startswith("[") else r)
    return df


def vela_images(needed, data_root, hf_cache, archives):
    """Write the images of `needed` (SA-1B file names) to $BEITSCORE_DATA/vela/images/ and return
    the names still missing. From --archive vela_images=sa_000138.tar when given, else from the
    Hugging Face copy of LongCap-Arena (only the splits of the evaluation groups are downloaded)."""
    from beitscore.vela import DIMS, EVAL_GROUPS

    folder = os.path.join(data_root, "vela", "images")
    os.makedirs(folder, exist_ok=True)
    missing = {n for n in needed if not os.path.exists(os.path.join(folder, n))}
    if missing and "vela_images" in archives:
        extract(archives["vela_images"], folder, names=missing)
        missing = {n for n in missing if not os.path.exists(os.path.join(folder, n))}
    if missing:
        from datasets import Image, load_dataset

        for split in [f"{g}.{d}" for d in DIMS for g in EVAL_GROUPS]:
            if not missing:
                break
            ds = load_dataset(VELA_HF, split=split, cache_dir=hf_cache).cast_column("image", Image(decode=False))
            for row in ds:
                name = str(row["file_name"]).replace("\\", "/").rsplit("/", 1)[-1]
                if name in missing:
                    data = row["image"]["bytes"]
                    if data is None:
                        with open(row["image"]["path"], "rb") as f:
                            data = f.read()
                    with open(os.path.join(folder, name), "wb") as f:
                        f.write(data)
                    missing.discard(name)
    return sorted(missing)


def vela_frame(data_root, downloads, hf_cache, archives):
    from beitscore.vela import DIMS, EVAL_GROUPS

    folder = os.path.join(downloads, "vela")
    for group, sub in VELA_GROUPS.items():
        for d in DIMS:
            http_download(VELA_RAW.format(repo=VELA_REPO, commit=VELA_COMMIT, folder=sub, dim=d, group=group),
                          os.path.join(folder, f"{d}_{group}.csv"), quiet=True)
    df = vela_ratings(folder)
    df["image_path"] = "vela/images/" + df["imgid"].astype(str)  # relative to $BEITSCORE_DATA
    needed = set(df.loc[df["subset"].isin(EVAL_GROUPS), "imgid"])  # train and val only give ratings
    missing = vela_images(needed, data_root, hf_cache, archives)
    if missing:
        print(f"vela: {len(missing)} of {len(needed)} images not found (e.g. {missing[0]}); pass "
              f"--archive vela_images=<sa_000138.tar of SA-1B> to take them from SA-1B")
    cols = ["image_path", "caption", "desc", "rel", "flu", "subset", "data_type", "imgid"]
    return df[cols + (["refs"] if "refs" in df.columns else [])]


# --------------------------------------------------------------------------- compare
def _norm(text):
    return " ".join(str(text).lower().split())


def benchmark_keys(df, task):
    """One key per row that does not depend on where the images are stored: the caption and its
    sorted ratings (correlation), or the correct and incorrect caption (incorrect captions)."""
    if task == "correlation":
        ratings = df["human_scores"].map(lambda s: tuple(sorted(round(float(r), 4) for r in
                                                              (s if isinstance(s, (list, tuple, np.ndarray)) else [s]))))
        return list(zip(df["caption"].map(_norm), ratings))
    cols = [c for c in ("caption", "foil", "tag_combination") if c in df.columns]
    return [tuple(_norm(v) for v in row) for row in df[cols].to_numpy()]


def compare_benchmarks(names, data_root, reference_root, split="test", examples=3):
    """Load each benchmark from both data folders and compare the rows (images are not compared,
    since their file names differ). Returns {name: (rows here, rows in reference, matched)}."""
    from collections import Counter

    from beitscore.data.datasets import get_data

    result = {}
    for name in names:
        try:
            mine, task, _ = get_data(name, data_root, split)
            ref, _, _ = get_data(name, reference_root, split)
        except Exception as exc:  # noqa: BLE001  (report and go on)
            print(f"[error]   {name}: {exc!r}")
            continue
        a, b = Counter(benchmark_keys(mine, task)), Counter(benchmark_keys(ref, task))
        matched = sum((a & b).values())
        result[name] = (len(mine), len(ref), matched)
        ok = matched == len(mine) == len(ref)
        line = f"{name}: {len(mine)} rows here, {len(ref)} in the reference, {matched} identical"
        if task == "correlation":
            r_mine = sum(len(k[1]) for k in a.elements())
            r_ref = sum(len(k[1]) for k in b.elements())
            line += f"; ratings {r_mine} here, {r_ref} in the reference"
            ok = ok and r_mine == r_ref
        print(("[same]    " if ok else "[differs] ") + line)
        if not ok:
            for key in list((a - b).elements())[:examples]:
                print(f"            only here:          {key}")
            for key in list((b - a).elements())[:examples]:
                print(f"            only in reference:  {key}")
    return result


# --------------------------------------------------------------------------- download
# Where each benchmark comes from. `benchmarks.py download` fetches these and converts them;
# images are downloaded one by one from their official host when there is one (COCO, Open
# Images), otherwise only the needed files are extracted from the official archive.
COCO_URL = "http://images.cocodataset.org/{split}/{file}"
OPENIMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/{split}/{file}"
GITHUB_RAW = "https://raw.githubusercontent.com/{repo}/main/{path}"
SUGARCREPE_FILES = ["add_att", "add_obj", "replace_att", "replace_obj", "replace_rel", "swap_att", "swap_obj"]
VALSE_FILES = ["existence", "plurals", "counting-hard", "counting-small-quant", "counting-adversarial", "relations",
               "action-replacement", "actant-swap", "coreference-standard", "coreference-hard", "foil-it"]
# Flickr8K as in CLIPScore (jmhessel/clipscore, flickr8k_example/download.py): the same two
# Google Drive archives, processed the same way.
FLICKR8K_DRIVE = {"Flickr8k_Dataset.zip": "1WNY8pV-u8xtBYBVal03qwjQs4VKurUZn",
                  "Flickr8k_text.zip": "1ljB7DR-YM-q9WKnHDW5dHjauK029B2s6"}
# Archives for the VALSE images that have no per-image host. These links come from the
# dataset pages and could not be tested from where this script was written: if one fails,
# download it by hand and pass it with --archive DATASET=FILE.
VALSE_ARCHIVES = {
    "visual7w": "http://vision.stanford.edu/yukezhu/visual7w_images.zip",
    "SWiG": "https://swig-data-weights.s3.us-east-2.amazonaws.com/images_512.zip",
    "VisDial_v1.0/val2018": "https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip?dl=1",
}
# VICR (Narins et al., 2023) cannot be downloaded by a script: OpenReview needs a browser.
VICR_PAGE = "https://openreview.net/forum?id=xKYtTmtyI2"
VICR_CSV = os.path.join("vicr", "VICR.csv")  # relative to $BEITSCORE_DATA
# Composite (Aditya et al., 2015): the AMT evaluation files linked from
# imagesdg.wordpress.com/image-to-scene-description-graph, the official page cited by
# Kilickaya et al. (2017) and the TACL 2025 captioning-evaluation survey.
COMPOSITE_URL = "https://www.dropbox.com/scl/fi/pktyhl9th4ycluq/AMT_eval.zip?rlkey=r63cr4ff2tx7su6u5ckqk81rn&dl=1"
# LongCap-Arena of VELA (Matsuda et al., EMNLP 2025): the ratings are the CSV files of the authors'
# repository (one per dimension and group, joined on image and caption); the images are SA-1B
# photos, taken from the authors' Hugging Face copy of the benchmark or from sa_000138.tar of SA-1B.
VELA_REPO = "Ka2ukiMatsuda/VELA"
VELA_COMMIT = "666e33c74b179028a5caabb3036b1f646503c1bd"
VELA_RAW = "https://raw.githubusercontent.com/{repo}/{commit}/data/{folder}/{dim}_{group}.csv"
VELA_HF = "Ka2ukiMatsuda/LongCap-Arena"
VELA_GROUPS = {"train": "train", "val": "val", "dci_val": "test", "dci_test": "test"}  # group -> folder
PAPERS = {
    "BEiTScore": ["vicr", "polaris", "nebula", "flickr8k-expert", "flickr8k-cf", "composite", "genai", "nocaps-foil",
                  "valse", "sugarcrepe", "winoground", "longcapvlcp", "caparena", "vela"],
    "BEiTScore++": ["vicr", "polaris", "nebula", "flickr8k-expert", "composite"],
}
MANUAL = {
    "longcapvlcp": "LongCapVLCP: `python scripts/download_release.py longcapvlcp`",
}


def http_download(url, dest, chunk=1 << 22, quiet=False):
    """Download url to dest unless it is already there (resumes with a .part file)."""
    import requests
    from tqdm import tqdm

    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as r:
        if r.status_code == 416:  # already complete
            os.replace(part, dest)
            return dest
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        total = int(r.headers.get("content-length", 0)) + (done if mode == "ab" else 0)
        with open(part, mode) as f, tqdm(total=total or None, initial=done if mode == "ab" else 0, unit="B",
                                         unit_scale=True, desc=os.path.basename(dest), disable=quiet) as bar:
            for block in r.iter_content(chunk):
                f.write(block)
                bar.update(len(block))
    os.replace(part, dest)
    return dest


def fetch_many(jobs, workers=16, desc="Downloading images"):
    """jobs: [(list of candidate urls, dest)]; each file from the first url that works.
    Returns the dests that could not be downloaded."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    def one(job):
        urls, dest = job
        if os.path.exists(dest):
            return None
        for url in urls:
            for _ in range(3):
                try:
                    http_download(url, dest, quiet=True)
                    return None
                except Exception:  # noqa: BLE001  (try again, then the next url)
                    continue
        return dest

    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed = [d for d in tqdm(ex.map(one, jobs), total=len(jobs), desc=desc, unit="img") if d]
    if failed:
        print(f"{len(failed)} of {len(jobs)} files could not be downloaded, e.g. {failed[:3]}")
    return failed


def gdrive_download(file_id, dest):
    """A Google Drive file (with gdown, which handles the large-file confirmation)."""
    if os.path.exists(dest):
        return dest
    import gdown

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if gdown.download(id=file_id, output=dest, quiet=False) is None:
        raise RuntimeError(f"Google Drive download of {file_id} failed")
    return dest


def extract(archive, dest, names=None):
    """Extract a .zip or .tar archive to dest; with `names`, only members whose base name is
    in it (so a large archive does not have to be unpacked whole). Returns {base name: path}."""
    import tarfile
    import zipfile

    found = {}
    os.makedirs(dest, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                base = os.path.basename(m.filename)
                if m.is_dir() or not base or (names is not None and base not in names):
                    continue
                out = os.path.join(dest, base if names is not None else m.filename)
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with z.open(m) as src, open(out, "wb") as f:
                        f.write(src.read())
                found[base] = out
    else:
        with tarfile.open(archive) as tar:
            for m in tar.getmembers():
                base = os.path.basename(m.name)
                if not m.isfile() or (names is not None and base not in names):
                    continue
                out = os.path.join(dest, base if names is not None else m.name)
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with tar.extractfile(m) as src, open(out, "wb") as f:
                        f.write(src.read())
                found[base] = out
    return found


def coco_jobs(files, folder):
    """Per-image jobs for COCO file names (COCO_train2014_..., COCO_val2014_..., or 2017 ids)."""
    jobs = []
    for f, split in files:
        m = re.match(r"COCO_(train|val)2014_", f)
        split = f"{m.group(1)}2014" if m else split
        jobs.append(([COCO_URL.format(split=split, file=f)], os.path.join(folder, f)))
    return jobs


def clipscore_flickr8k(text_dir, out_dir):
    """flickr8k.json and crowdflower_flickr8k.json exactly as CLIPScore builds them
    (adapted from jmhessel/clipscore, flickr8k_example/download.py): expert captions that are
    also references of their image are skipped (as in SPICE), CrowdFlower keeps one entry per
    caption with the proportion of "yes"."""
    import collections

    image2ann, caption = collections.defaultdict(list), {}
    with open(os.path.join(text_dir, "Flickr8k.token.txt")) as f:
        for line in f:
            image, ann = line.strip().split("\t")
            image2ann[image.split("#")[0]].append(ann)
            caption[image] = ann
    image2ann = {k.split(".")[0]: v for k, v in image2ann.items()}

    crowd = {}
    with open(os.path.join(text_dir, "CrowdFlowerAnnotations.txt")) as f:
        for line in f:
            image_id, caption_id, yes_prec, _n_yes, _n_no = line.strip().split("\t")
            path = f"Flicker8k_Dataset/{image_id}"
            entry = crowd.setdefault(image_id, {"human_judgement": [], "image_id": image_id, "image_path": path,
                                                "ground_truth": list(image2ann[image_id.split(".")[0]])})
            if caption[caption_id] in entry["ground_truth"]:
                entry["ground_truth"].remove(caption[caption_id])
            entry["human_judgement"].append({"image_id": image_id, "image_path": path,
                                             "caption": caption[caption_id], "rating": float(yes_prec)})
    expert = {}
    with open(os.path.join(text_dir, "ExpertAnnotations.txt")) as f:
        for line in f:
            image_id, caption_id, *ratings = line.strip().split("\t")
            key = image_id.split(".")[0]
            if caption[caption_id] in image2ann[key]:
                continue
            path = f"Flicker8k_Dataset/{image_id}"
            entry = expert.setdefault(key, {"human_judgement": [], "image_id": key, "image_path": path,
                                            "ground_truth": image2ann[key]})
            for r in ratings:
                entry["human_judgement"].append({"image_id": key, "image_path": path,
                                                 "caption": caption[caption_id], "rating": float(r)})
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("flickr8k.json", expert), ("crowdflower_flickr8k.json", crowd)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(data, f)
    return out_dir


def nocaps_foil_frame(records, data_root=DATA_ROOT, workers=16):
    """nocaps-FOIL of ALOHa (data/nocaps-val-foil.json): image_path, baseline (correct
    caption), foil (incorrect caption), domain. Images are nocaps validation images, that is
    Open Images validation images."""
    folder = os.path.join(data_root, "nocaps-foil", "images")
    files = sorted({r["image_path"] for r in records})
    fetch_many([([OPENIMAGES_URL.format(split=s, file=f) for s in ("validation", "test", "train")],
                 os.path.join(folder, f)) for f in files], workers)
    rows = pd.DataFrame({"image_path": [f"nocaps-foil/images/{r['image_path']}" for r in records],
                         "caption": [r["baseline"].strip() for r in records],
                         "foil": [r["foil"].strip() for r in records],
                         "data_type": [r["domain"] for r in records]})
    return pairs_frame(rows, "nocaps-foil", None, data_root)


FLICKR30K_KAGGLE = "adityajn105/flickr30k"


def flickr8k_folder(data_root=DATA_ROOT, archives=None):
    """Flicker8k_Dataset/ and the annotation text folder of the CLIPScore Flickr8K archives
    (downloaded once). Returns (image folder, text folder)."""
    archives = archives or {}
    folder = os.path.join(data_root, "downloads", "flickr8k")
    for fname, file_id in FLICKR8K_DRIVE.items():
        if os.path.exists(os.path.join(folder, fname[:-4] + ".done")):
            continue
        archive = archives.get(fname) or gdrive_download(file_id, os.path.join(folder, fname))
        extract(archive, folder)
        open(os.path.join(folder, fname[:-4] + ".done"), "w").close()
    walk = [(r, ds, fs) for r, ds, fs in os.walk(folder) if "__MACOSX" not in r]
    images = next((os.path.join(r, d) for r, ds, _ in walk for d in ds if d == "Flicker8k_Dataset"), None)
    texts = next((r for r, _, fs in walk if "ExpertAnnotations.txt" in fs), None)
    if images is None or texts is None:
        raise FileNotFoundError(f"Flicker8k_Dataset/ or ExpertAnnotations.txt not found under {folder}")
    return images, texts


def flickr30k_folder(data_root=DATA_ROOT):
    """The Flickr30k images of the Kaggle archive (the one the Stage-1 release uses), in
    $BEITSCORE_DATA/flickr30k/Images (downloaded once)."""
    import shutil

    target = os.path.join(data_root, "flickr30k", "Images")
    if os.path.isdir(target) and any(f.lower().endswith(".jpg") for f in os.listdir(target)):
        return target
    import kagglehub

    root = kagglehub.dataset_download(FLICKR30K_KAGGLE)
    best = _largest_jpg_folder(root)
    if best is None:  # the cached copy was already moved out (e.g. by download_release.py stage1)
        root = kagglehub.dataset_download(FLICKR30K_KAGGLE, force_download=True)
        best = _largest_jpg_folder(root)
    if best is None:
        raise FileNotFoundError(f"no .jpg files in the Kaggle download of {FLICKR30K_KAGGLE}")
    if os.path.lexists(target):
        os.remove(target) if os.path.islink(target) else shutil.rmtree(target)
    os.makedirs(target)
    # only the .jpg files of that folder: the archive also holds a second copy of them in a subfolder
    for name in os.listdir(best):
        if name.lower().endswith(".jpg"):
            shutil.move(os.path.join(best, name), os.path.join(target, name))
    shutil.rmtree(root, ignore_errors=True)  # the rest of the Kaggle download, so nothing is kept twice
    return target


def _largest_jpg_folder(root):
    """The folder under root with the most .jpg files, or None."""
    best, most = None, 0
    for folder, _, files in os.walk(root):
        n = sum(f.lower().endswith(".jpg") for f in files)
        if n > most:
            best, most = folder, n
    return best


def place(files, source_dir, dest_dir):
    """Hard-link (or copy) the named files from source_dir into dest_dir; returns the missing."""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    missing = []
    for f in files:
        src, dst = os.path.join(source_dir, f), os.path.join(dest_dir, f)
        if os.path.exists(dst):
            continue
        if not os.path.exists(src):
            missing.append(f)
            continue
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    if missing:
        print(f"{len(missing)} files not found in {source_dir}, e.g. {missing[:3]}")
    return missing


def vicr_frame(table, data_root=DATA_ROOT, workers=16, archives=None):
    """VICR (Narins et al., 2023), Dataset/VICR.csv of the supplementary material: one row per
    pair with image (URL), caption, subset and the ratings in ratings, rating_2, ..., rating_7.
    Every rating of a pair is kept. COCO images come from cocodataset.org, the Flickr8K ones
    from the Flickr8K archive (same files as validatedicr.org/images/)."""
    df = read_table(table) if not isinstance(table, pd.DataFrame) else table
    cols = [c for c in df.columns if c == "ratings" or c.startswith("rating_")]
    files = df["image"].map(lambda u: u.rsplit("/", 1)[-1])
    folder = os.path.join(data_root, "vicr", "images")
    coco = sorted({f for f in files if f.startswith("COCO_")})
    fetch_many(coco_jobs([(f, "val2014") for f in coco], folder), workers, desc="VICR COCO images")
    others = sorted(set(files) - set(coco))
    if others:
        images, _ = flickr8k_folder(data_root, archives)
        missing = set(place(others, images, folder))
        if missing:  # fall back to the copies hosted by the VICR authors
            fetch_many([([u], os.path.join(folder, f)) for u, f in zip(df["image"], files) if f in missing],
                       workers, desc="VICR images from validatedicr.org")
    out = pd.DataFrame({
        "image_path": "vicr/images/" + files,
        "caption": df["caption"].astype(str),
        "human_scores": [[float(v) for v in row if not pd.isna(v)] for row in df[cols].to_numpy()],
        "subset": df["subset"].map(normalise_split),
        "data_type": "vicr",
    })
    return group_ratings(out, keys=("subset", "image_path", "caption"))


def composite_frame(folder, data_root=DATA_ROOT, workers=16, archives=None, score="average", duplicates="first"):
    """Composite (Aditya et al., 2015) from the AMT evaluation files: one HIT per image with
    three candidates (Flickr8K) or four (Flickr30k, COCO), each rated once for correctness and
    once for thoroughness on 1-5. A fourth candidate identical to the third is the same pair
    and is dropped (its first rating is kept), which gives 13,146 pairs over 3,995 images.

    score="average" (the default) uses the mean of the correctness and thoroughness ratings as
    the single score of a pair; the PAC-S authors also use both files (aimagelab/pacscore,
    issue 5). It reproduces the Composite data of the papers (mean 3.00,
    standard deviation 1.45, quartiles 1.5/3.0/4.5). score="correctness" uses the correctness
    rating alone, as Kilickaya et al. (2017) and the TACL 2025 survey describe.

    duplicates="first" (the default, as in the papers) keeps one rating for a pair rated twice
    (a fourth candidate identical to the third); duplicates="keep" keeps both ratings in the
    pair's list, so the correlation counts both, as for the other datasets with several
    ratings per pair (14,983 ratings over the same 13,146 pairs)."""
    def load(kind):
        rows = []
        for source in ("8k", "30k", "coco"):
            d = pd.read_csv(os.path.join(folder, f"{source}_{kind}.csv"), sep=";", dtype=str, keep_default_na=False)
            for k in range(1, 5):
                if f"Input.text{k}" not in d.columns:
                    continue
                for url, cap, r in zip(d["Input.image_url"], d[f"Input.text{k}"], d[f"Answer.Q{k}Answer"]):
                    if cap.strip() and r.strip():
                        rows.append({"source": source, "file": url.rsplit("/", 1)[-1], "caption": cap, "k": k,
                                     "rating": float(r)})
        return pd.DataFrame(rows)

    rows = load("correctness")
    if score == "average":
        thorough = load("throughness").rename(columns={"rating": "thoroughness"})
        rows = rows.merge(thorough, on=["source", "file", "caption", "k"], how="left")
        rows["rating"] = rows[["rating", "thoroughness"]].mean(axis=1)
    elif score != "correctness":
        raise ValueError(f"score must be 'average' or 'correctness', not {score!r}")
    rows["human_scores"] = rows["rating"].map(lambda r: [float(r)])
    if duplicates == "keep":
        df = (rows.groupby(["source", "file", "caption"], sort=False)
              .agg(human_scores=("human_scores", lambda s: [r for rs in s for r in rs])).reset_index())
    elif duplicates == "first":
        df = rows.drop_duplicates(["source", "file", "caption"]).reset_index(drop=True)
    else:
        raise ValueError(f"duplicates must be 'first' or 'keep', not {duplicates!r}")
    folder_out = os.path.join(data_root, "composite", "images")
    coco = sorted(set(df.loc[df["source"] == "coco", "file"]))
    fetch_many(coco_jobs([(f, "val2014") for f in coco], folder_out), workers, desc="Composite COCO images")
    place(sorted(set(df.loc[df["source"] == "8k", "file"])), flickr8k_folder(data_root, archives)[0], folder_out)
    place(sorted(set(df.loc[df["source"] == "30k", "file"])), flickr30k_folder(data_root), folder_out)
    out = pd.DataFrame({"image_path": "composite/images/" + df["file"], "caption": df["caption"],
                        "human_scores": df["human_scores"], "subset": "test", "data_type": "composite",
                        "source": df["source"].map({"8k": "flickr8k", "30k": "flickr30k", "coco": "coco"})})
    out["candidates"] = out["caption"]
    return out


def download_benchmark(name, data_root=DATA_ROOT, archives=None, workers=16, keep_all=False,
                       composite_score="average", composite_duplicates="first"):
    """Download and convert one benchmark into $BEITSCORE_DATA/<name>/<name>.pkl. Raises
    NotImplementedError with what to do by hand when it cannot."""
    name = canonical_benchmark(name)
    archives = archives or {}
    downloads = os.path.join(data_root, "downloads")
    hf_cache = os.path.join(downloads, "huggingface")  # removed with the downloads
    if name == "polaris":
        ds = load_hf("yuwd/Polaris", cache_dir=hf_cache)
        return save(polaris_frame({s: ds[s] for s in ds}, data_root), name, data_root)
    if name == "nebula":
        ds = load_hf("Ka2ukiMatsuda/Nebula", cache_dir=hf_cache)
        return save(nebula_frame({s: ds[s] for s in ds}, data_root), name, data_root)
    if name == "genai":
        ds = load_hf(GENAI_REPO, cache_dir=hf_cache)
        return save(genai_frame(ds["train"] if "train" in ds else ds[next(iter(ds))], data_root), name, data_root)
    if name == "caparena":
        return caparena_files(data_root, downloads, archives)
    if name == "vela":
        return save(vela_frame(data_root, downloads, hf_cache, archives), name, data_root)
    if name == "winoground":
        ds = load_hf("facebook/winoground", token=os.environ.get("HF_TOKEN"), cache_dir=hf_cache)
        return save(winoground_frame(ds["test"], data_root), name, data_root)
    if name == "sugarcrepe":
        folder = os.path.join(downloads, "sugarcrepe")
        for f in SUGARCREPE_FILES:
            http_download(GITHUB_RAW.format(repo="RAIVNLab/sugar-crepe", path=f"data/{f}.json"),
                          os.path.join(folder, f"{f}.json"))
        files = set()
        for f in SUGARCREPE_FILES:
            with open(os.path.join(folder, f"{f}.json")) as fh:
                files |= {r["filename"] for r in json.load(fh).values()}
        images = os.path.join(data_root, name, "images")
        fetch_many(coco_jobs([(f, "val2017") for f in sorted(files)], images), workers)
        return save(sugarcrepe_frame(folder, images, data_root), name, data_root)
    if name in ("flickr8k-expert", "flickr8k-cf"):
        images, texts = flickr8k_folder(data_root, archives)
        json_dir = clipscore_flickr8k(texts, os.path.join(downloads, "flickr8k", "clipscore_json"))
        return [save(df, n, data_root) for n, df in flickr8k_frames(json_dir, images, data_root).items()]
    if name == "nocaps-foil":
        path = http_download(GITHUB_RAW.format(repo="DavidMChan/aloha", path="data/nocaps-val-foil.json"),
                             os.path.join(downloads, "nocaps-val-foil.json"))
        with open(path) as f:
            records = json.load(f)
        return save(nocaps_foil_frame(records, data_root, workers), name, data_root)
    if name == "valse":
        folder = os.path.join(downloads, "valse")
        for f in VALSE_FILES:
            http_download(GITHUB_RAW.format(repo="Heidelberg-NLP/VALSE", path=f"data/{f}.json"),
                          os.path.join(folder, f"{f}.json"))
        needed = {}
        for f in VALSE_FILES:
            with open(os.path.join(folder, f"{f}.json")) as fh:
                for r in json.load(fh).values():
                    needed.setdefault(r["dataset"], set()).add((r["image_file"], r.get("original_split")))
        roots = {}
        for dataset, files in needed.items():
            roots[dataset] = os.path.join(data_root, name, valse_folder(dataset))
            coco = {(f, s) for f, s in files if dataset.lower().replace("_", "") == "coco2017" or f.startswith("COCO_")}
            if coco:
                fetch_many(coco_jobs([(f, f"{s}2017") for f, s in coco], roots[dataset]), workers,
                           desc=f"VALSE {dataset} images")
            rest = {f for f, _ in files} - {f for f, _ in coco}
            if rest:
                key = "VisDial_v1.0/val2018" if dataset == "VisDial_v1.0" else dataset
                if key not in VALSE_ARCHIVES and dataset not in archives:
                    raise NotImplementedError(f"VALSE {dataset}: no source for {len(rest)} images; "
                                              f"pass --archive {dataset}=FILE")
                archive = archives.get(dataset) or http_download(
                    VALSE_ARCHIVES[key], os.path.join(downloads, "valse_archives", re.sub(r"\W+", "_", key) + ".zip"))
                got = extract(archive, roots[dataset], names=rest)
                if len(got) < len(rest):
                    print(f"VALSE {dataset}: {len(rest) - len(got)} images not found in {archive}")
        return save(valse_frame(folder, roots, keep_all, data_root), name, data_root)
    if name == "vicr":
        csv = archives.get("vicr") or os.path.join(data_root, VICR_CSV)
        if not os.path.exists(csv):
            raise NotImplementedError(
                f"VICR: download the supplementary material from {VICR_PAGE} (Supplementary Material link), "
                f"unzip it and copy Dataset/VICR.csv to {os.path.join(data_root, VICR_CSV)}, then run "
                f"`benchmarks.py download vicr` again")
        return save(vicr_frame(csv, data_root, workers, archives), name, data_root)
    if name == "composite":
        archive = archives.get("composite") or http_download(COMPOSITE_URL,
                                                             os.path.join(downloads, "composite", "AMT_eval.zip"))
        extract(archive, os.path.join(downloads, "composite"))
        return save(composite_frame(os.path.join(downloads, "composite"), data_root, workers, archives,
                                    composite_score, composite_duplicates), name, data_root)
    if name in MANUAL:
        raise NotImplementedError(MANUAL[name])
    raise ValueError(f"unknown benchmark {name!r}; expected one of {sorted(BENCHMARKS)}")


def download(names, data_root=DATA_ROOT, archives=None, workers=16, keep_all=False, composite_score="average",
             composite_duplicates="first", keep_downloads=False):
    """Download every benchmark it can and print what has to be done by hand. When every
    requested benchmark is done (or only needs a manual step), $BEITSCORE_DATA/downloads is
    deleted, so no data is kept twice. Returns {name: "done" | "manual: ..." | "failed: ..."}."""
    import shutil

    status = {}
    for name in dict.fromkeys(canonical_benchmark(n) for n in names):
        if name == "flickr8k-cf" and "flickr8k-expert" in status:  # written together
            status[name] = status["flickr8k-expert"]
            continue
        try:
            download_benchmark(name, data_root, archives, workers, keep_all, composite_score, composite_duplicates)
            status[name] = "done"
        except NotImplementedError as exc:
            status[name] = f"manual: {exc}"
        except Exception as exc:  # noqa: BLE001  (network, gated access, ...): report and go on
            status[name] = f"failed: {exc!r}"
    print("\nBenchmarks:")
    for name, s in status.items():
        print(f"  {name:16s} {s}")
    downloads = os.path.join(data_root, "downloads")
    if os.path.isdir(downloads):
        if keep_downloads:
            print(f"Downloads kept in {downloads}")
        elif any(s.startswith("failed") for s in status.values()):
            print(f"Downloads kept in {downloads} so that a new run does not fetch them again; "
                  f"they are deleted after a run without failures")
        else:
            shutil.rmtree(downloads)
            print(f"Deleted {downloads}")
    return status


# --------------------------------------------------------------------------- CLI
def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data_root", default=DATA_ROOT)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download", help="Download and convert benchmarks")
    p.add_argument("names", nargs="*", default=["all"],
                   help="Benchmark names, or 'all' (default: every benchmark of the two papers)")
    p.add_argument("--archive", nargs="+", default=[], metavar="KEY=FILE",
                   help="Use a file you downloaded yourself (keys: vicr (VICR.csv), composite, visual7w, SWiG, "
                        "VisDial_v1.0, Flickr8k_Dataset.zip, Flickr8k_text.zip, caparena, caparena_images, "
                        "vela_images (sa_000138.tar of SA-1B))")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--keep_downloads", action="store_true", help="Keep $BEITSCORE_DATA/downloads afterwards")
    p.add_argument("--keep_all", action="store_true", help="VALSE: keep the instances the annotators did not validate")
    p.add_argument("--composite_score", choices=["average", "correctness"], default="average",
                   help="Composite score: mean of correctness and thoroughness (as in the papers) or correctness only")
    p.add_argument("--composite_duplicates", choices=["first", "keep"], default="first",
                   help="Composite pairs rated twice: one rating (as in the papers) or both ratings")
    p = sub.add_parser("ratings", help="Generic table of human ratings")
    p.add_argument("--name", required=True, type=canonical_benchmark, choices=CORRELATION)
    p.add_argument("--table", required=True, nargs="+", help="One or more files (concatenated)")
    p.add_argument("--image_col", required=True)
    p.add_argument("--caption_col", required=True)
    p.add_argument("--score_col", required=True, help="One rating per row, or a list of ratings per row")
    p.add_argument("--split_col")
    p.add_argument("--split", help="Split of every row when there is no split column")
    p.add_argument("--type_col", help="Column used as data_type (subsets reported separately)")
    p.add_argument("--images", help="Image folder; image_col is relative to it")
    p = sub.add_parser("pairs", help="Generic table of (image, correct caption, incorrect caption)")
    p.add_argument("--name", required=True, type=canonical_benchmark, choices=ERRORS)
    p.add_argument("--table", required=True)
    p.add_argument("--image_col", required=True)
    p.add_argument("--caption_col", required=True)
    p.add_argument("--incorrect_col", required=True)
    p.add_argument("--type_col")
    p.add_argument("--images")
    p = sub.add_parser("check", help="Load prepared benchmarks and report sizes and missing images")
    p.add_argument("names", nargs="*", default=[n for n in PAPERS["BEiTScore"] if n != "caparena"])
    p = sub.add_parser("compare", help="Compare the prepared benchmarks with those of another data folder")
    p.add_argument("--reference", required=True, help="Data folder holding the benchmark files used for the papers")
    p.add_argument("--split", default="test")
    p.add_argument("names", nargs="*", default=[n for n in PAPERS["BEiTScore"] if n not in ("longcapvlcp", "caparena", "vela")])
    return parser.parse_args(argv)


def main(args):
    root = args.data_root
    if args.cmd == "download":
        names = list(args.names)
        for alias in ("all", "papers"):
            if alias in names:
                names.remove(alias)
                names += [n for paper in PAPERS.values() for n in paper]
        return download(names, root, dict(a.split("=", 1) for a in args.archive), args.workers, args.keep_all,
                        args.composite_score, args.composite_duplicates, args.keep_downloads)
    if args.cmd == "ratings":
        table = pd.concat([read_table(t) for t in args.table], ignore_index=True)
        df = ratings_frame(table, args.image_col, args.caption_col, args.score_col, args.name,
                           split_col=args.split_col, split=args.split, type_col=args.type_col,
                           images=args.images, data_root=root)
        return save(df, args.name, root)
    if args.cmd == "pairs":
        t = read_table(args.table)
        rows = pd.DataFrame({"image_path": t[args.image_col].astype(str), "caption": t[args.caption_col],
                             "foil": t[args.incorrect_col]})
        if args.type_col:
            rows["data_type"] = t[args.type_col].astype(str)
        return save(pairs_frame(rows, args.name, args.images, root), args.name, root)
    if args.cmd == "compare":
        res = compare_benchmarks([canonical_benchmark(n) for n in args.names], root, args.reference, args.split)
        sys.exit(0 if all(m == n == r for n, r, m in res.values()) else 1)
    if args.cmd == "check":
        sys.exit(1 if check([canonical_benchmark(n) for n in args.names], root) else 0)


if __name__ == "__main__":
    main(get_args())

"""Get the images of a release (Stage-1 data or LongCapVLCP) and rebuild its data file.

    python scripts/get_images.py --release stage1_release --out_dir stage1_images

scripts/download_release.py runs this for you after downloading a release; run it by hand
only to resume, or to use folders other than the defaults.

For every image of <release>/images_index.parquet:
  * embedded images (PixelProse by default) are extracted from <release>/images/;
  * Flickr30k comes from its archive: download
    https://www.kaggle.com/datasets/adityajn105/flickr30k, unzip it, and pass
    --archive flickr30k=<folder>/Images;
  * the others are downloaded from their official URL (COCO, Open Images, TextCaps), or
    taken from a folder you already have, with --archive SOURCE=DIR (e.g. the COCO
    train2014 folder, or the TextVQA train_images folder for TextCaps);
then checked against their SHA-256 and saved as <out_dir>/<source>/<file_name>.

Statuses in <out_dir>/images_report.csv:
  ok            byte-identical to the file used in training, either as downloaded or after
                the RGB/JPEG re-encoding that was applied when the training images were saved
                images (PIL convert("RGB"), JPEG with default settings)
  same_picture  the source served different bytes, but the perceptual hash is within
                --max_phash_distance (a re-encoded copy; needs imagehash); kept, and used for
                training only with --accept_same_picture
  mismatch      a different picture; not saved
  missing       not available

Then writes <out_dir>/stage1.pkl (--out_file): the rows of <release>/stage1.parquet with
`image_path` pointing at the local files, ready for scripts/train.py. Rows whose image is
not available are dropped, and the numbers are printed. Rerunning skips verified files.
Needs pandas, pyarrow and requests (imagehash optional).
"""
import argparse
import glob
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm.auto import tqdm

USER_AGENT = "Uncertainty_BEiTScore_pp image downloader (research use)"
ARCHIVE_HINTS = {"flickr30k": "https://www.kaggle.com/datasets/adityajn105/flickr30k, folder Images/"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_parquet_in_batches(path, batch_size=50_000):
    """pd.read_parquet, batch by batch: tables with list columns larger than 2 GB in one row
    group (the Stage-1 table) fail to load in one piece with "Nested data conversions not
    implemented for chunked array outputs"."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    parts = [b.to_pandas() for b in pf.iter_batches(batch_size=batch_size)]
    return pd.concat(parts, ignore_index=True) if parts else pf.schema_arrow.empty_table().to_pandas()


def is_verified(path, expected):
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        return sha256(f.read()) == expected


def save(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "wb") as f:
        f.write(data)
    os.replace(path + ".part", path)


def phash_distance(data, expected_hex):
    try:
        import imagehash
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return imagehash.phash(img.convert("RGB")) - imagehash.hex_to_hash(expected_hex)
    except Exception:
        return None


def rgb_jpeg(data):
    """PIL convert("RGB") and save as JPEG with the default settings, as the training
    images were saved."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            out = io.BytesIO()
            img.convert("RGB").save(out, format="JPEG")
        return out.getvalue()
    except Exception:
        return None


def fetch(url, timeout=20, retries=3):
    import requests

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 200 and r.content:
                return r.content
            if r.status_code in (403, 404, 410):
                return None
        except Exception:
            pass
        time.sleep(1.5 * (attempt + 1))
    return None


# ---------------------------------------------------------------- embedded images
def extract_embedded(release, out_dir):
    """{image_id: (status, path)} for the images stored in the shards."""
    import pyarrow.parquet as pq

    shards = sorted(glob.glob(os.path.join(release, "images", "*.parquet")))
    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    result = {}
    with tqdm(total=total, desc="Extracting embedded images", unit="img") as bar:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(
                    batch_size=256, columns=["image_id", "source", "file_name", "sha256", "image"]):
                for row in batch.to_pylist():
                    dest = os.path.join(out_dir, row["source"], row["file_name"])
                    if is_verified(dest, row["sha256"]):
                        result[row["image_id"]] = ("ok", dest)
                    elif sha256(row["image"]["bytes"]) == row["sha256"]:
                        save(dest, row["image"]["bytes"])
                        result[row["image_id"]] = ("ok", dest)
                    else:
                        result[row["image_id"]] = ("mismatch", None)  # corrupted shard: download it again
                    bar.update(1)
    return result


# ---------------------------------------------------------------- downloaded images
def get_one(row, out_dir, archives, max_phash_distance, fetch_fn):
    dest = os.path.join(out_dir, row.source, row.file_name)
    if is_verified(dest, row.sha256):
        return row.image_id, "ok", dest
    data = None
    if row.source in archives:
        candidate = os.path.join(archives[row.source], row.native_image_id)  # name inside the archive
        if os.path.exists(candidate):
            with open(candidate, "rb") as f:
                data = f.read()
    if data is None and isinstance(row.url, str):
        data = fetch_fn(row.url)
    if data is None:
        return row.image_id, "missing", None
    if sha256(data) != row.sha256:
        redone = rgb_jpeg(data)
        if redone is not None and sha256(redone) == row.sha256:
            data = redone
    if sha256(data) == row.sha256:
        status = "ok"
    else:
        dist = phash_distance(data, row.phash)
        status = "same_picture" if dist is not None and dist <= max_phash_distance else "mismatch"
    if status == "mismatch":
        return row.image_id, status, None
    save(dest, data)
    return row.image_id, status, dest


def main(args, fetch_fn=fetch):
    index = pd.read_parquet(os.path.join(args.release, "images_index.parquet"))
    if args.sources:
        index = index[index["source"].isin(args.sources)]
    archives = dict(a.split("=", 1) for a in args.archive or [])
    for source, folder in archives.items():
        if not os.path.isdir(folder):
            raise SystemExit(f"--archive {source}: {folder} is not a folder")
    needs_archive = sorted(set(index.loc[~index["embedded"] & index["url"].isna(), "source"]) - set(archives))
    for source in needs_archive:
        print(f"WARNING: {source} images have no URL; download its archive and pass --archive {source}=DIR "
              f"({ARCHIVE_HINTS.get(source, 'see the dataset card')}). They will be reported missing.")

    results = {}
    if index["embedded"].any():
        results.update(extract_embedded(args.release, args.out_dir))
    todo = list(index[~index["embedded"]].itertuples(index=False))
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        jobs = ex.map(lambda r: get_one(r, args.out_dir, archives, args.max_phash_distance, fetch_fn), todo)
        for image_id, status, path in tqdm(jobs, total=len(todo), desc="Downloading images", unit="img",
                                           mininterval=2):
            results[image_id] = (status, path)

    report = index[["image_id", "source", "embedded"]].copy()
    report["status"] = report["image_id"].map(lambda i: results.get(i, ("missing", None))[0])
    report["path"] = report["image_id"].map(lambda i: results.get(i, ("missing", None))[1])
    os.makedirs(args.out_dir, exist_ok=True)
    report.to_csv(os.path.join(args.out_dir, "images_report.csv"), index=False)
    print(pd.crosstab(report["source"], report["status"], margins=True).to_string())

    keep = {"ok"} | ({"same_picture"} if args.accept_same_picture else set())
    local = {i: p for i, s, p in zip(report["image_id"], report["status"], report["path"]) if s in keep}
    df = read_parquet_in_batches(os.path.join(args.release, f"{args.table}.parquet"))
    df["image_path"] = df["image_id"].map(local).map(lambda p: os.path.abspath(p) if isinstance(p, str) else p)
    lost = df["image_path"].isna()
    print(f"Rows: {len(df):,}; with their image: {(~lost).sum():,}; dropped: {lost.sum():,} "
          f"({100 * lost.mean():.2f}%)")
    df = df[~lost].reset_index(drop=True)
    df = df[["image_path"] + [c for c in df.columns if c != "image_path"]]
    out_file = args.out_file or os.path.join(args.out_dir, f"{args.table}.pkl")
    df.to_pickle(out_file)
    print(f"Data file written to {out_file}")
    return report, df


def get_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", required=True, help="Folder with stage1.parquet, images_index.parquet, images/")
    parser.add_argument("--out_dir", required=True, help="Where the images are written")
    parser.add_argument("--out_file", default=None, help="Data file written (default <out_dir>/<table>.pkl)")
    parser.add_argument("--table", default="stage1", help="Main table of the release: <table>.parquet")
    parser.add_argument("--archive", action="append", metavar="SOURCE=DIR",
                        help="Folder with the extracted official archive of a source (repeatable)")
    parser.add_argument("--sources", nargs="+", default=None, help="Only these sources")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max_phash_distance", type=int, default=4)
    parser.add_argument("--accept_same_picture", action="store_true",
                        help="Use re-encoded copies (status same_picture) for training")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(get_args())

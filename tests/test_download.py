"""scripts/download_release.py and scripts/get_images.py on a small fake release."""
import hashlib
import io
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from conftest import load_script

pytest.importorskip("pyarrow")


def jpeg(seed, mode="RGB"):
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (32, 32, 3), dtype=np.uint8)).convert(mode).save(
        buf, format="PNG" if mode == "RGBA" else "JPEG")
    return buf.getvalue()


def rgb_jpeg(data):
    out = io.BytesIO()
    Image.open(io.BytesIO(data)).convert("RGB").save(out, format="JPEG")
    return out.getvalue()


def write_release(folder, table, rows, images, embedded_bytes):
    """A release as published: <table>.parquet, images_index.parquet, images/*.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(folder / f"{table}.parquet", index=False)
    index = pd.DataFrame(images)
    index.to_parquet(folder / "images_index.parquet", index=False)
    emb = index[index["embedded"]]
    (folder / "images").mkdir()
    cols = ["image_id", "source", "native_image_id", "file_name", "sha256", "phash", "width", "height", "n_bytes"]
    data = {c: emb[c].tolist() for c in cols}
    data["image"] = [{"bytes": embedded_bytes[i], "path": n} for i, n in zip(emb["image_id"], emb["file_name"])]
    pq.write_table(pa.table(data), folder / "images" / "images-00000-of-00001.parquet")


def image_row(image_id, source, data, url=None, embedded=False):
    return dict(image_id=image_id, source=source, native_image_id=image_id.split(":")[1] + ".jpg",
                file_name=image_id.split(":")[1] + ".jpg", sha256=hashlib.sha256(data).hexdigest(),
                phash="0" * 16, width=32, height=32, n_bytes=len(data), url=url, embedded=embedded)


def test_download_and_get_images(tmp_path, monkeypatch):
    # Stage 1: one embedded PixelProse image, one COCO 2014 image saved with the RGB/JPEG step
    # (the URL serves the original PNG), one Open Images image as served, one Flickr30k image
    # from the Kaggle archive, one Open Images image whose URL is dead
    pixel, coco_png, oi, flickr = jpeg(1), jpeg(2, "RGBA"), jpeg(3), jpeg(4)
    coco_local = rgb_jpeg(coco_png)
    web = {"http://coco/x.jpg": coco_png, "http://oi/y.jpg": oi}
    stage1_images = [
        image_row("pixelprose:p1", "pixelprose", pixel, embedded=True),
        image_row("coco2014:x", "coco2014", coco_local, url="http://coco/x.jpg"),
        image_row("openimages:y", "openimages", oi, url="http://oi/y.jpg"),
        image_row("flickr30k:z", "flickr30k", flickr),
        image_row("openimages:dead", "openimages", jpeg(9), url="http://oi/dead.jpg"),
    ]
    rows = [dict(image_id=r["image_id"], caption=f"c{i}", foil=f"f{i}", tokenized_words=["a"])
            for i, r in enumerate(stage1_images)]
    hub = tmp_path / "hub"
    write_release(hub / "stage1", "stage1", rows, stage1_images, {"pixelprose:p1": pixel})
    long_img = jpeg(5)
    write_release(hub / "longcapvlcp", "longcapvlcp",
                  [dict(image_id="longcapvlcp:d1", caption="long", foil=False, pair_id="0")],
                  [image_row("longcapvlcp:d1", "longcapvlcp", long_img, embedded=True)], {"longcapvlcp:d1": long_img})
    for repo in ("beitscore", "beitscorepp"):
        for size in ("base", "large"):
            (hub / repo / size).mkdir(parents=True)
            (hub / repo / size / "model.safetensors").write_bytes(b"weights")
    (hub / "beitscore" / "beit3.spm").write_bytes(b"spm")
    kaggle = tmp_path / "flickr30k" / "Images"
    kaggle.mkdir(parents=True)
    (kaggle / "z.jpg").write_bytes(flickr)

    names = {"beitscore": "beitscore", "beitscorepp": "beitscorepp", "LongCapVLCP": "longcapvlcp",
             "beitscore-stage1": "stage1"}

    def fake_snapshot(repo_id, repo_type, local_dir, allow_patterns=None):
        name = repo_id.split("/")[1]
        if name not in names:  # a repository that is not published (stage2 in this test)
            from huggingface_hub.utils import RepositoryNotFoundError
            raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)
        shutil.copytree(hub / names[name], local_dir, dirs_exist_ok=True)
        return local_dir

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot)
    data, models = tmp_path / "data", tmp_path / "models"
    dl = load_script("scripts/download_release.py", "download_release")
    monkeypatch.setattr(dl, "DATA_ROOT", str(data))
    monkeypatch.setattr(dl, "MODELS_ROOT", str(models))

    def get_images_offline(*args, **kwargs):  # the Hub copy is local; the URLs come from `web`
        get = load_script("scripts/get_images.py", "get_images_offline")
        release_dir, out_dir, out_file, table, archives, workers = args
        argv = ["--release", release_dir, "--out_dir", out_dir, "--out_file", out_file, "--table", table]
        for s, f in archives.items():
            argv += ["--archive", f"{s}={f}"]
        return get.main(get.get_args(argv), fetch_fn=lambda url: web.get(url))

    monkeypatch.setattr(dl, "get_images", get_images_offline)
    dl.main(dl.get_args(["all", "--flickr30k", str(kaggle)]))

    assert (models / "beitscorepp" / "large" / "model.safetensors").exists()
    assert (models / "beitscore" / "base" / "model.safetensors").exists() and (models / "beit3.spm").exists()
    long = pd.read_pickle(data / "longcapvlcp" / "longcapvlcp.pkl")
    assert open(long["image_path"][0], "rb").read() == long_img

    stage1 = pd.read_pickle(data / "Stage1" / "stage1.pkl")
    assert sorted(stage1["image_id"]) == ["coco2014:x", "flickr30k:z", "openimages:y", "pixelprose:p1"]
    expected = {"pixelprose:p1": pixel, "coco2014:x": coco_local, "openimages:y": oi, "flickr30k:z": flickr}
    for path, image_id in zip(stage1["image_path"], stage1["image_id"]):
        assert open(path, "rb").read() == expected[image_id]  # byte-identical, COCO via the RGB/JPEG step
    report = pd.read_csv(data / "Stage1" / "images" / "images_report.csv").set_index("image_id")["status"]
    assert report["openimages:dead"] == "missing" and (report.drop("openimages:dead") == "ok").all()
    assert list(stage1.columns[:2]) == ["image_path", "image_id"] and stage1["tokenized_words"][0] == ["a"]
    assert os.path.exists(data / "Stage1" / "images" / "pixelprose" / "p1.jpg")


def test_released_beitscore_checkpoint_is_preferred(tmp_path, monkeypatch):
    import beitscore.inference as inference

    monkeypatch.setattr(inference, "MODELS_ROOT", str(tmp_path))
    assert inference.beitscore_checkpoint("large").endswith("vqa/large_with_stage1.pth")
    (tmp_path / "beitscore" / "large").mkdir(parents=True)
    (tmp_path / "beitscore" / "large" / "model.safetensors").write_bytes(b"x")
    assert inference.beitscore_checkpoint("large") == str(tmp_path / "beitscore" / "large" / "model.safetensors")


def test_flickr30k_from_kaggle(tmp_path, monkeypatch):
    """Without --flickr30k the Kaggle archive is downloaded (here a fake kagglehub) and its
    Images/ folder moved to $BEITSCORE_DATA/flickr30k/Images; a second call reuses it."""
    import sys
    import types

    dl = load_script("scripts/download_release.py", "download_release_flickr")
    cache = tmp_path / "kaggle_cache" / "versions" / "1"
    (cache / "Images").mkdir(parents=True)
    (cache / "Images" / "1000092795.jpg").write_bytes(b"jpg")
    (cache / "captions.txt").write_text("x")
    calls = []
    fake = types.SimpleNamespace(dataset_download=lambda handle, **kw: calls.append(handle) or str(cache))
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    target = tmp_path / "data" / "flickr30k" / "Images"
    assert dl.get_flickr30k(str(target)) == str(target)
    assert (target / "1000092795.jpg").exists() and calls == ["adityajn105/flickr30k"]
    assert dl.get_flickr30k(str(target)) == str(target) and len(calls) == 1


def test_stage2_download_and_loading(tmp_path, monkeypatch):
    """download_release.py stage2: the release is fetched, Flickr8K comes from its archive (a
    fake gdown here), get_images writes stage2/stage2.pkl, and get_data("regression") reads it."""
    import sys
    import types
    import zipfile

    import joblib

    from beitscore.data.datasets import get_data

    dl = load_script("scripts/download_release.py", "download_release_stage2")
    data = tmp_path / "data"
    monkeypatch.setattr(dl, "DATA_ROOT", str(data))

    def fake_gdown(id, output, quiet):
        with zipfile.ZipFile(output, "w") as z:
            z.writestr("Flicker8k_Dataset/1_ab.jpg", b"jpg")
            z.writestr("__MACOSX/Flicker8k_Dataset/._1_ab.jpg", b"junk")
        return output

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download=fake_gdown))
    target = data / "flickr8k" / "Flicker8k_Dataset"
    assert dl.get_flickr8k(str(target)) == str(target)
    assert os.listdir(target) == ["1_ab.jpg"] and not (data / "flickr8k" / "Flickr8k_Dataset.zip").exists()

    calls = []
    monkeypatch.setattr(dl, "snapshot", lambda key, local_dir, ns, allow_patterns=None: calls.append(key) or local_dir)
    image = data / "stage2" / "images" / "nebula" / "a.png"

    def fake_get_images(release, out_dir, out_file, table, archives, workers):
        calls.append((table, archives))
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        joblib.dump(pd.DataFrame({"image_path": [str(image)] * 4, "candidates": list("abcd"),
                                  "human_scores": [0.1, 0.2, 0.3, 0.4], "dataset": "nebula",
                                  "subset": ["train", "train", "val", "val"], "cluster": [0, 1, 0, 1]}), out_file)

    monkeypatch.setattr(dl, "get_images", fake_get_images)
    monkeypatch.setattr(dl, "get_flickr8k", lambda: str(target))
    dl.download_stage2(dl.get_args(["stage2"]))
    assert calls == ["stage2", ("stage2", {"flickr8k": str(target)})]
    val, task, _ = get_data("regression", str(data), "val")
    assert task == "regression" and sorted(val["human_scores"]) == [0.3, 0.4]
    assert val["image_path"].tolist() == [str(image)] * 2  # absolute paths are kept


def test_flickr30k_kaggle_cache_already_emptied(tmp_path, monkeypatch):
    """An earlier run moved the images out of the Kaggle cache: the download is forced again."""
    import sys
    import types

    dl = load_script("scripts/download_release.py", "download_release_flickr_empty")
    bench = load_script("scripts/prepare_data/benchmarks.py", "benchmarks_flickr_empty")
    empty, full = tmp_path / "cache_empty", tmp_path / "cache_full"

    def dataset_download(handle, force_download=False):
        if not force_download:
            empty.mkdir(exist_ok=True)
            return str(empty)
        (full / "Images").mkdir(parents=True, exist_ok=True)
        (full / "Images" / "1.jpg").write_bytes(b"jpg")
        return str(full)

    monkeypatch.setitem(sys.modules, "kagglehub", types.SimpleNamespace(dataset_download=dataset_download))
    target = tmp_path / "a" / "flickr30k" / "Images"
    assert dl.get_flickr30k(str(target)) == str(target) and (target / "1.jpg").exists()
    import shutil
    shutil.rmtree(full)
    assert bench.flickr30k_folder(str(tmp_path / "b")) == str(tmp_path / "b" / "flickr30k" / "Images")
    assert (tmp_path / "b" / "flickr30k" / "Images" / "1.jpg").exists()

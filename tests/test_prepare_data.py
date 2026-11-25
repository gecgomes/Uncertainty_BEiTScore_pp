"""Benchmark downloads and converters (scripts/prepare_data/benchmarks.py)."""
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from PIL import Image

from beitscore.data.datasets import get_data
from beitscore.metrics import explode_pairs

bench = load_script("scripts/prepare_data/benchmarks.py", "prep_benchmarks")


def _jpeg(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "JPEG")
    return buf.getvalue()


def _image_dir(tmp_path, names):
    d = tmp_path / "originals"
    d.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        (d / n).write_bytes(_jpeg((i * 40 % 255, 10, 10)))
    return d


def test_ratings_are_kept_per_pair_not_averaged(tmp_path):
    images = _image_dir(tmp_path, ["a.jpg", "b.jpg"])
    table = pd.DataFrame({
        "img": ["a.jpg", "a.jpg", "a.jpg", "b.jpg", "b.jpg", "a.jpg"],
        "cap": ["x", "x", "x", "y", "y", "z"],
        "rating": [1, 2, 5, 3, "[4, 4]", 2],
        "split": ["test"] * 5 + ["validation"],
    })
    df = bench.ratings_frame(table, "img", "cap", "rating", "vicr", split_col="split",
                             images=images, data_root=str(tmp_path))
    bench.save(df, "vicr", str(tmp_path))
    test, task, _ = get_data("vicr", str(tmp_path), "test")
    assert task == "correlation"
    scores = dict(zip(test["caption"], test["human_scores"]))
    assert scores == {"x": [1.0, 2.0, 5.0], "y": [3.0, 4.0, 4.0]}
    assert all(os.path.exists(p) for p in test["image_path"])
    x, y = explode_pairs([0.1, 0.2], test["human_scores"].tolist())
    assert len(x) == len(y) == 6  # one metric score per human rating
    val, _, _ = get_data("vicr", str(tmp_path), "val")
    assert val["caption"].tolist() == ["z"]


def test_polaris_groups_judgments_and_stores_images_once(tmp_path):
    img = {"bytes": _jpeg((200, 0, 0)), "path": "p.jpg"}
    records = [{"img": img, "cand": "c", "refs": ["r"], "human_score": s} for s in (0.25, 0.5, 1.0)]
    df = bench.polaris_frame({"test": records, "validation": records[:1]}, str(tmp_path))
    assert len(df) == 2 and len(os.listdir(tmp_path / "polaris" / "images")) == 1
    assert df.set_index("subset").loc["test", "human_scores"] == [0.25, 0.5, 1.0]


def test_flickr8k_paths_and_grouping(tmp_path):
    images = _image_dir(tmp_path, ["1.jpg"])
    src = tmp_path / "json"
    src.mkdir()
    data = {"1": {"image_path": "some/old/place/1.jpg", "ground_truth": ["a dog"],
                  "human_judgement": [{"caption": "a  dog", "rating": r, "image_path": "x/1.jpg"} for r in (1, 3, 4)]}}
    (src / "flickr8k.json").write_text(json.dumps(data))
    (src / "crowdflower_flickr8k.json").write_text(json.dumps(data))
    for name, df in bench.flickr8k_frames(str(src), str(images), str(tmp_path)).items():
        bench.save(df, name, str(tmp_path))
    df, _, _ = get_data("flickr8k-expert", str(tmp_path), "test")
    assert df["human_scores"].tolist() == [[1.0, 3.0, 4.0]] and df["caption"].tolist() == ["a dog"]
    assert df["image_path"][0] == str(tmp_path / "flickr8k-expert" / "images" / "1.jpg")
    assert os.path.exists(df["image_path"][0])


def test_sugarcrepe_pairs(tmp_path):
    images = _image_dir(tmp_path, ["000000000001.jpg"])
    js = tmp_path / "sc"
    js.mkdir()
    (js / "swap_att.json").write_text(json.dumps({"0": {"filename": "000000000001.jpg", "caption": "a red car",
                                                         "negative_caption": "a car red"}}))
    bench.save(bench.sugarcrepe_frame(str(js), str(images), str(tmp_path)), "sugarcrepe", str(tmp_path))
    df, task, _ = get_data("sugarcrepe", str(tmp_path), "test")
    assert task == "foil" and df["foil"].tolist() == ["a car red"] and os.path.exists(df["image_path"][0])



def test_winoground_tags(tmp_path):
    rec = {"id": 3, "image_0": {"bytes": _jpeg((0, 0, 0)), "path": "0.png"},
           "image_1": {"bytes": _jpeg((255, 255, 255)), "path": "1.png"},
           "caption_0": "a b", "caption_1": "b a", "collapsed_tag": "Object"}
    bench.save(bench.winoground_frame([rec], str(tmp_path)), "winoground", str(tmp_path))
    df, _, wino = get_data("winoground", str(tmp_path), "test")
    assert wino and sorted(df["tag_combination"]) == ["00", "01", "10", "11"]
    assert set(df["pair_id"]) == {"3"}
    assert df.set_index("tag_combination")["foil"].to_dict() == {"00": False, "01": True, "10": True, "11": False}


def test_check_reports_missing_images(tmp_path, capsys):
    df = pd.DataFrame({"image_path": ["Nebula/images/none.jpg"], "caption": ["c"], "human_scores": [[0.5]],
                       "subset": ["test"], "data_type": ["nebula"]})
    bench.save(df, "nebula", str(tmp_path))
    assert bench.check(["nebula", "vicr"], str(tmp_path)) == 2  # size differs from the paper, image missing
    out = capsys.readouterr().out
    assert "[check]   nebula" in out and "image files not found" in out and "[missing] vicr" in out


@pytest.mark.parametrize("value,expected", [(3, [3.0]), ("[1, 2]", [1.0, 2.0]), ([4, np.nan], [4.0]), ("2.5", [2.5])])
def test_as_ratings(value, expected):
    assert bench.as_ratings(value) == expected


def test_download_does_what_it_can_and_lists_the_rest(tmp_path, monkeypatch):
    payload = json.dumps({"0": {"filename": "000000000001.jpg", "caption": "a red car",
                                "negative_caption": "a car red"}})

    def fake_http(url, dest, **kw):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w") as f:
            f.write(payload)
        return dest

    def fake_fetch(jobs, workers=16, desc=""):
        for _, dest in jobs:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_bytes(_jpeg((1, 2, 3)))
        return []

    def no_hub(repo, token=None):
        raise ConnectionError("offline")

    monkeypatch.setattr(bench, "http_download", fake_http)
    monkeypatch.setattr(bench, "fetch_many", fake_fetch)
    monkeypatch.setattr(bench, "load_hf", no_hub)
    status = bench.download(["sugarcrepe", "longcapvlcp", "polaris"], str(tmp_path))
    assert status["sugarcrepe"] == "done"
    assert status["longcapvlcp"].startswith("manual:")
    assert status["polaris"].startswith("failed:")
    df, _, _ = get_data("sugarcrepe", str(tmp_path), "test")
    assert len(df) == 7 and all(os.path.exists(p) for p in df["image_path"])  # one instance per category file


def test_flickr8k_built_as_in_clipscore(tmp_path):
    import zipfile

    text = tmp_path / "Flickr8k_text.zip"
    with zipfile.ZipFile(text, "w") as z:
        z.writestr("Flickr8k.token.txt", "1.jpg#0\ta dog runs\n1.jpg#1\ta brown dog\n2.jpg#0\ta cat sits\n")
        z.writestr("ExpertAnnotations.txt", "1.jpg\t2.jpg#0\t1\t2\t2\n1.jpg\t1.jpg#1\t4\t4\t4\n")
        z.writestr("CrowdFlowerAnnotations.txt", "1.jpg\t2.jpg#0\t0.333333\t1\t2\n")
    imgs = tmp_path / "Flickr8k_Dataset.zip"
    with zipfile.ZipFile(imgs, "w") as z:
        z.writestr("Flicker8k_Dataset/1.jpg", _jpeg((9, 9, 9)))
        z.writestr("__MACOSX/Flicker8k_Dataset/._1.jpg", b"junk")
    archives = {"Flickr8k_text.zip": str(text), "Flickr8k_Dataset.zip": str(imgs)}
    status = bench.download(["flickr8k-expert", "flickr8k-cf"], str(tmp_path / "data"), archives)
    assert status == {"flickr8k-expert": "done", "flickr8k-cf": "done"}
    ex, _, _ = get_data("flickr8k-expert", str(tmp_path / "data"), "test")
    assert ex["caption"].tolist() == ["a cat sits"]  # the reference caption of image 1 is skipped, as in SPICE
    assert sorted(ex["human_scores"][0]) == [1.0, 2.0, 2.0] and os.path.exists(ex["image_path"][0])
    cf, _, _ = get_data("flickr8k-cf", str(tmp_path / "data"), "test")
    assert np.allclose(cf["human_scores"][0], [0.333333])


def test_nocaps_foil_and_valse(tmp_path, monkeypatch):
    import zipfile

    def fake_fetch(jobs, workers=16, desc=""):
        for urls, dest in jobs:
            assert urls[0].startswith(("https://open-images-dataset", "http://images.cocodataset.org"))
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_bytes(_jpeg((5, 5, 5)))
        return []

    monkeypatch.setattr(bench, "fetch_many", fake_fetch)
    data = str(tmp_path / "data")
    df = bench.nocaps_foil_frame([{"image_path": "0013ea2087020901.jpg", "baseline": "A kid in a shirt. ",
                                   "foil": "A kid in a skirt. ", "domain": "in-domain"}], data)
    assert df[["caption", "foil", "data_type"]].values.tolist() == [["A kid in a shirt.", "A kid in a skirt.",
                                                                     "in-domain"]]

    valse = tmp_path / "data" / "downloads" / "valse"
    valse.mkdir(parents=True)
    recs = {"existence": {"0": {"dataset": "visual7w", "image_file": "v7w_1.jpg", "caption": "c", "foil": "f",
                                "original_split": "test", "mturk": {"foil": 0, "caption": 3, "other": 0}},
                          "1": {"dataset": "visual7w", "image_file": "v7w_2.jpg", "caption": "c2", "foil": "f2",
                                "original_split": "test", "mturk": {"foil": 2, "caption": 1, "other": 0}}},
            "plurals": {"0": {"dataset": "coco2017", "image_file": "000000291619.jpg", "caption": "c", "foil": "f",
                              "original_split": "val", "mturk": {"foil": 0, "caption": 2, "other": 1}}},
            "foil-it": {"0": {"dataset": "FOIL dataset", "image_file": "COCO_val2014_000000165831.jpg",
                              "caption": "c", "foil": "f", "original_split": "test",
                              "mturk": {"foil": 0, "caption": 3, "other": 0}}}}
    for name in bench.VALSE_FILES:
        (valse / f"{name}.json").write_text(json.dumps(recs.get(name, {})))
    v7w = tmp_path / "visual7w.zip"
    with zipfile.ZipFile(v7w, "w") as z:
        z.writestr("images/v7w_1.jpg", _jpeg((1, 1, 1)))
        z.writestr("images/v7w_2.jpg", _jpeg((2, 2, 2)))
        z.writestr("images/v7w_3.jpg", _jpeg((3, 3, 3)))  # not used: not extracted
    status = bench.download(["valse"], data, {"visual7w": str(v7w)})
    assert status == {"valse": "done"}
    df, task, _ = get_data("valse", data, "test")
    assert sorted(df["data_type"]) == ["existence", "foil-it", "plurals"]  # the unvalidated instance is left out
    assert all(os.path.exists(p) for p in df["image_path"])
    assert not os.path.exists(os.path.join(data, "valse", "images", "visual7w", "v7w_3.jpg"))


def test_vicr_and_composite_from_their_releases(tmp_path, monkeypatch):
    def fake_fetch(jobs, workers=16, desc=""):
        for urls, dest in jobs:
            assert urls[0].startswith("http://images.cocodataset.org/val2014/")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_bytes(_jpeg((5, 5, 5)))
        return []

    flickr = _image_dir(tmp_path, ["111_aa.jpg", "4911020894.jpg"])
    monkeypatch.setattr(bench, "fetch_many", fake_fetch)
    monkeypatch.setattr(bench, "flickr8k_folder", lambda data_root=None, archives=None: (str(flickr), None))
    monkeypatch.setattr(bench, "flickr30k_folder", lambda data_root=None: str(flickr))
    data = str(tmp_path / "data")

    vicr = pd.DataFrame({
        "image": ["http://images.cocodataset.org/val2014/COCO_val2014_000000003817.jpg",
                  "https://validatedicr.org/images/111_aa.jpg"],
        "caption": ["a man with bananas", "a dog"], "subset": ["test", "train"], "num_ratings": [5, 3],
        "ratings": [4, 1], "rating_2": [3, 2], "rating_3": [4, 2], "rating_4": [3.0, np.nan],
        "rating_5": [3.0, np.nan], "rating_6": [np.nan, np.nan], "rating_7": [np.nan, np.nan]})
    bench.save(bench.vicr_frame(vicr, data), "vicr", data)
    df, _, _ = get_data("vicr", data, "test")
    assert df["human_scores"].tolist() == [[4.0, 3.0, 4.0, 3.0, 3.0]] and os.path.exists(df["image_path"][0])

    comp = tmp_path / "amt"
    comp.mkdir()
    head3 = "Input.image_url;Input.text1;Input.text2;Input.text3;Answer.Q1Answer;Answer.Q2Answer;Answer.Q3Answer\n"
    head4 = ("Input.image_url;Input.text1;Input.text2;Input.text3;Input.text4;"
             "Answer.Q1Answer;Answer.Q2Answer;Answer.Q3Answer;Answer.Q4Answer\n")
    u8, u30 = "http://x/Flicker8k_Dataset/111_aa.jpg", "http://x/Flicker30k_Dataset/4911020894.jpg"
    uc = "http://x/Coco_Dataset/COCO_val2014_000000243955.jpg"
    for kind, a, b in (("correctness", "5;1;2", "4;2;2;3"), ("throughness", "3;1;2", "2;2;4;1")):
        (comp / f"8k_{kind}.csv").write_text(head3 + f"{u8};r;m1;m2;{a}\n")
        (comp / f"30k_{kind}.csv").write_text(head4 + f"{u30};r;m1;m2;m2;{b}\n")  # 4th = 3rd: one pair
        (comp / f"coco_{kind}.csv").write_text(head4 + f"{uc};r;m1;m2;m3;{b}\n")
    out = bench.composite_frame(str(comp), data)
    assert len(out) == 3 + 3 + 4
    scores = dict(zip(zip(out["source"], out["caption"]), out["human_scores"]))
    assert scores[("flickr8k", "r")] == [4.0] and scores[("flickr30k", "m2")] == [3.0]  # mean of (2, 4)
    assert scores[("coco", "m3")] == [2.0]
    corr = bench.composite_frame(str(comp), data, score="correctness")
    assert dict(zip(zip(corr["source"], corr["caption"]), corr["human_scores"]))[("flickr8k", "r")] == [5.0]
    both = bench.composite_frame(str(comp), data, duplicates="keep")
    assert len(both) == len(out) and dict(zip(zip(both["source"], both["caption"]), both["human_scores"]))[
        ("flickr30k", "m2")] == [3.0, 2.0]  # the 3rd and the identical 4th candidate: (2+4)/2 and (3+1)/2
    bench.save(out, "composite", data)
    df, _, _ = get_data("composite", data, "test")
    assert len(df) == 10 and all(os.path.exists(p) for p in df["image_path"])


def test_compare_ignores_image_locations(tmp_path, capsys):
    for root, path, extra in ((tmp_path / "a", "VICR/images/x.jpg", []), (tmp_path / "b", "/old/data/VICR/y.jpg", [4.0])):
        df = pd.DataFrame({"image_path": [path, path], "caption": ["A dog", "a cat"],
                           "human_scores": [[1.0, 2.0], [3.0] + extra], "subset": ["test", "test"]})
        bench.save(df, "vicr", str(root))
    res = bench.compare_benchmarks(["vicr"], str(tmp_path / "a"), str(tmp_path / "b"))
    assert res["vicr"] == (2, 2, 1)
    out = capsys.readouterr().out
    assert "[differs] vicr" in out and "only in reference" in out


def test_vicr_needs_its_csv_and_downloads_are_removed(tmp_path, monkeypatch):
    data = tmp_path / "data"

    def fake_fetch(jobs, workers=16, desc=""):
        for _, dest in jobs:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            Path(dest).write_bytes(_jpeg((5, 5, 5)))
        return []

    monkeypatch.setattr(bench, "fetch_many", fake_fetch)
    status = bench.download(["vicr"], str(data))
    assert status["vicr"].startswith("manual:") and "openreview.net/forum?id=xKYtTmtyI2" in status["vicr"]
    assert str(data / "vicr" / "VICR.csv") in status["vicr"]

    (data / "vicr").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image": ["http://images.cocodataset.org/val2014/COCO_val2014_000000003817.jpg"],
                  "caption": ["a man"], "subset": ["test"], "num_ratings": [3], "ratings": [4], "rating_2": [3],
                  "rating_3": [4]}).to_csv(data / "vicr" / "VICR.csv", index=False)
    (data / "downloads" / "something").mkdir(parents=True)
    status = bench.download(["vicr"], str(data))
    assert status == {"vicr": "done"}
    assert (data / "vicr" / "vicr.pkl").exists() and (data / "vicr" / "images" / "COCO_val2014_000000003817.jpg").exists()
    assert not (data / "downloads").exists()  # removed once every requested benchmark is done
    df, _, _ = get_data("vicr", str(data), "test")
    assert df["human_scores"].tolist() == [[4.0, 3.0, 4.0]]

    (data / "downloads" / "something").mkdir(parents=True)
    bench.download(["vicr"], str(data), keep_downloads=True)
    assert (data / "downloads").exists()



def _png_bytes(color):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="JPEG")
    return buf.getvalue()


def test_genai_frame_names_images_by_model_and_prompt(tmp_path):
    records = []
    for i in range(2):
        r = {"Prompt": f"prompt {i}", "Tags": {"basic": ["Scene"], "advanced": []}}
        for m in bench.GENAI_MODELS:
            r[m] = {"bytes": _png_bytes((i * 50, 0, 0)), "path": None}
        if i == 0:
            r["HumanRatings"] = {m: [5, 4, 4] for m in bench.GENAI_MODELS}
        else:  # the other layout of the Hub card: one string field per model
            for m in bench.GENAI_MODELS:
                r[f"{m}_HumanRating"] = "[1, 2, 3]"
        records.append(r)
    bench.save(bench.genai_frame(records, str(tmp_path)), "genai", str(tmp_path))
    df, task, _ = get_data("genai", str(tmp_path), "test")
    assert task == "correlation" and len(df) == 2 * len(bench.GENAI_MODELS)
    assert df["image_path"].map(os.path.basename).str.match(r"^[A-Za-z0-9_]+_[01]\.png$").all()
    assert sorted(map(list, df["human_scores"]))[0] == [1.0, 2.0, 3.0]
    assert all(os.path.exists(p) for p in df["image_path"])


def test_caparena_files_from_local_archives(tmp_path):
    import zipfile

    battles = [{"img": "test_00001.jpg", "source1": "human", "source2": "GPT-4o-0806", "caption1": "a",
                "caption2": "b", "winner": "human", "cluster": "level 1"}]
    annots = tmp_path / "annots.json"
    annots.write_text(json.dumps(battles))
    archive = tmp_path / "images.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("all_images/test_00001.jpg", _png_bytes((0, 0, 0)))
        z.writestr("all_images/test_99999.jpg", _png_bytes((0, 0, 0)))
    data = tmp_path / "data"
    path = bench.caparena_files(str(data), str(tmp_path / "dl"),
                                {"caparena": str(annots), "caparena_images": str(archive)})
    assert json.load(open(path)) == battles
    assert sorted(os.listdir(data / "caparena" / "images")) == ["test_00001.jpg"]

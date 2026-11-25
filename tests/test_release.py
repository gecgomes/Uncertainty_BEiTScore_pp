"""Tests T-01 to T-12 of RELEASE_CHECKLIST.md."""
import argparse
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import torch
from PIL import Image

from conftest import REPO, load_script
from beitscore import metrics
from beitscore.pp.calibration.bounds import hoeffding_UCB
from beitscore.pp.calibration.clustered import Calibration, calibrate
from beitscore.pp.calibration.intervals import interval_fn3

N_CAND = 40  # bias candidates in the tests (1,000 in the paper)


def synthetic(n=600, seed=0, clusters=3):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, clusters, n)
    noise = np.array([0.05, 0.12, 0.25])[labels]  # cluster 2 is the hardest
    mid = rng.uniform(0.2, 0.8, n)
    gt = np.clip(mid + rng.normal(0, noise), 0, 1)
    return np.clip(mid - 0.05, 0, 1), mid, np.clip(mid + 0.05, 0, 1), gt, labels


# ---------------------------------------------------------------- T-01 clustered calibration (C-06)
def test_clustered_calibration_pairs_and_conservative_choice():
    low, mid, high, gt, labels = synthetic()
    res = calibrate(low, mid, high, gt, clusters=labels, n_candidates=N_CAND)
    assert res.clustered and len(res.clusters) == 3
    for c in res.clusters:
        mask = labels == c.cluster
        assert c.size == mask.sum()
        # each cluster's pair is what an independent calibration of that cluster gives (n = cluster size)
        alone = calibrate(low[mask], mid[mask], high[mask], gt[mask], n_candidates=N_CAND)
        assert (c.omega, c.beta) == pytest.approx((alone.omega, alone.beta))
        lo, _, hi = interval_fn3(low, mid, high, b=c.beta, l=c.omega)
        assert c.width_on_calibration_set == pytest.approx(np.mean(hi - lo))
    best = max(res.clusters, key=lambda c: c.width_on_calibration_set)
    assert (res.omega, res.beta, res.selected_cluster) == (best.omega, best.beta, best.cluster)
    assert res.selected_cluster == 2  # the noisiest cluster needs the widest intervals


def test_small_clusters_are_skipped():
    low, mid, high, gt, labels = synthetic()
    labels = labels.copy()
    labels[:5] = 99
    res = calibrate(low, mid, high, gt, clusters=labels, min_cluster_size=30, n_candidates=N_CAND)
    assert 99 not in [c.cluster for c in res.clusters]


def test_every_script_uses_the_shared_calibration():
    for path in ("scripts/beitscore_pp/calibrate.py",):
        src = (REPO / path).read_text()
        assert "from beitscore.pp.calibration.clustered import" in src
        assert "def interval_fn3" not in src and "confromal_risk_control" not in src


# ---------------------------------------------------------------- T-02 calibration.json
def test_calibration_json_round_trip(tmp_path):
    low, mid, high, gt, labels = synthetic()
    res = calibrate(low, mid, high, gt, clusters=labels, n_candidates=N_CAND)
    res.model = "beitscorepp_large"
    res.save(tmp_path / "calibration.json")
    back = Calibration.load(tmp_path / "calibration.json")
    assert back == res
    np.testing.assert_allclose(back.apply(low, mid, high), interval_fn3(low, mid, high, b=res.beta, l=res.omega))


# ---------------------------------------------------------------- T-03 scoring API (N-01)
class FakeTokenizer:
    bos_token_id, eos_token_id, pad_token_id, mask_token_id = 0, 2, 1, 4

    def tokenize(self, text):
        return text.split()

    def convert_tokens_to_ids(self, tokens):
        return [3 + (sum(map(ord, t)) % 90) for t in tokens]


@pytest.fixture
def tiny_scorers(tiny_cfg, monkeypatch):
    from beitscore import scorer
    from beitscore.models.heads import BEiT3ForQuantileRegression, BEiT3ForRegression

    # the tiny model uses 32 x 32 images
    from beitscore.data.datasets import build_transform

    monkeypatch.setattr(scorer, "build_transform", lambda: build_transform(32))
    torch.manual_seed(0)
    pp = scorer.BEiTScorePP(BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True).eval(), FakeTokenizer(),
                            "cpu", calibration=Calibration(omega=1.3, beta=0.02))
    bs = scorer.BEiTScore(BEiT3ForRegression(tiny_cfg, num_classes=1).eval(), FakeTokenizer(), "cpu")
    return bs, pp


def _inputs(n=5):
    rng = np.random.default_rng(0)
    images = [Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)) for _ in range(n)]
    captions = ["a dog", "a red bus parked on a busy street", "two cats", "a man riding a horse on the beach", "x"]
    return images, captions[:n]


def test_scoring_api_beitscorepp(tiny_scorers):
    _, pp = tiny_scorers
    images, captions = _inputs()
    out = pp.score(images, captions, batch_size=2, return_quantiles=True)
    assert out["quantiles"].shape == (5, 11)
    for k in ("median", "low", "high"):
        assert np.all((out[k] >= 0) & (out[k] <= 1))
    assert np.all(out["low"] <= out["median"]) and np.all(out["median"] <= out["high"])
    raw = pp.score(images, captions, calibrated=False)
    assert np.all(out["high"] - out["low"] >= raw["high"] - raw["low"] - 1e-6)  # omega > 1 and beta > 0 widen
    single = np.concatenate([pp.score([i], [c])["median"] for i, c in zip(images, captions)])
    np.testing.assert_allclose(single, out["median"], atol=1e-5)


def test_scoring_api_beitscore(tiny_scorers):
    bs, _ = tiny_scorers
    images, captions = _inputs()
    s = bs.score(images, captions, batch_size=3)
    assert s.shape == (5,) and np.all((s >= 0) & (s <= 1))
    np.testing.assert_allclose([bs.score([i], [c])[0] for i, c in zip(images, captions)], s, atol=1e-5)


def test_package_exports_the_api():
    import beitscore

    assert beitscore.BEiTScorePP.__name__ == "BEiTScorePP" and beitscore.BEiTScore.__name__ == "BEiTScore"


# ---------------------------------------------------------------- T-04 per-subset metrics (C-10)
def test_metrics_are_computed_on_each_subset():
    evaluate = load_script("scripts/evaluate.py", "evaluate_script")
    rng = np.random.default_rng(1)
    q = np.sort(rng.random((40, 11)), axis=1)
    df = pd.DataFrame({f"q{i:02d}": q[:, i] for i in range(11)})
    df["human_scores"] = [[float(v)] for v in rng.random(40)]
    df["data_type"] = ["a"] * 20 + ["b"] * 20
    levels = [0.9, 0.85, 0.8, 0.75, 0.7]
    from beitscore.pp.calibration.clustered import IDENTITY

    sub = df[df["data_type"] == "b"]
    res = evaluate.evaluate_subset(sub, True, IDENTITY, levels)
    gt = metrics.mean_human_score(sub["human_scores"])
    assert res["coverage"] == pytest.approx(100 * np.mean((gt >= sub["q00"]) & (gt <= sub["q10"])))
    assert res["sharpness"] == pytest.approx(np.mean(sub["q10"] - sub["q00"]))
    assert res["ece"] == pytest.approx(metrics.quantile_ece(q[20:], gt, metrics.quantile_levels(levels)))


# ---------------------------------------------------------------- T-05 nominal scales (C-13)
def test_nominal_scaling():
    from beitscore.data.datasets import scale_scores

    assert scale_scores([[1, 5], 3.0], "vicr") == [[0.0, 1.0], 0.5]
    # a split whose ratings do not reach the scale ends keeps its place on the scale
    assert scale_scores([2.0, 4.0], "vicr") == [0.25, 0.75]
    assert scale_scores([2.0, 4.0], "vicr", mode="observed") == [0.0, 1.0]
    assert scale_scores([2.0, 4.0], "unknown-benchmark") == [0.0, 1.0]
    with pytest.raises(ValueError):
        scale_scores([0.0, 6.0], "vicr")


# ---------------------------------------------------------------- T-06 seeding (C-12)
def test_get_data_shuffle_is_seeded(tmp_path):
    from beitscore.data.datasets import get_data

    (tmp_path / "stage2").mkdir()
    df = pd.DataFrame({"image_path": [f"/x/data/img{i}.jpg" for i in range(30)], "candidates": [f"c{i}" for i in range(30)],
                       "human_scores": [[0.5]] * 30, "split": ["train"] * 30})
    joblib.dump(df, tmp_path / "stage2" / "stage2.pkl")
    a = get_data("regression", str(tmp_path), "train", seed=0)[0]["candidates"].tolist()
    b = get_data("regression", str(tmp_path), "train", seed=0)[0]["candidates"].tolist()
    c = get_data("regression", str(tmp_path), "train", seed=1)[0]["candidates"].tolist()
    assert a == b and a != c


# ---------------------------------------------------------------- T-07 head settings (C-09)
def test_head_settings_follow_the_checkpoint(tmp_path):
    from beitscore.inference import head_settings

    ckpt = tmp_path / "checkpoint-best.pth"
    torch.save({"model": {}, "args": argparse.Namespace(chained_construction=False, quantiles=[0.9])}, ckpt)
    assert head_settings(str(ckpt)) == {"chained_construction": False, "quantiles": [0.9]}
    assert head_settings(str(ckpt), chained_construction=True)["chained_construction"] is True
    released = tmp_path / "release"
    released.mkdir()
    (released / "config.json").write_text('{"chained_construction": true}')
    assert head_settings(str(released / "model.safetensors"))["chained_construction"] is True
    assert head_settings(str(tmp_path / "missing.pth"))["chained_construction"] is True  # paper default


# ---------------------------------------------------------------- T-08 strict mode (C-11)
def test_strict_mode_controls_the_bound_after_the_search():
    from beitscore.pp.calibration.risk_control import confromal_risk_control_using_CI_and_WB, risk_coverage_fn

    low, mid, high, gt, _ = synthetic(n=800)
    pred = np.stack((low, mid, high), axis=1)
    l0, b0 = confromal_risk_control_using_CI_and_WB(pred, gt, 0.9, 0.05, len(gt), n_candidates=N_CAND, progress=False)
    l1, b1 = confromal_risk_control_using_CI_and_WB(pred, gt, 0.9, 0.05, len(gt), n_candidates=N_CAND, progress=False,
                                                    strict=True)
    def width(l, b):
        lo, _, hi = interval_fn3(low, mid, high, b=b, l=l)
        return np.mean(hi - lo)

    assert width(l1, b1) >= width(l0, b0) - 1e-9
    risk = risk_coverage_fn(pred, gt, l1, b1)
    assert hoeffding_UCB(risk, len(gt), 0.05 / N_CAND) <= 0.1 + 1e-6


# ---------------------------------------------------------------- T-09 metrics against scipy
def test_metrics_match_scipy():
    rng = np.random.default_rng(2)
    s = rng.random(50)
    h = [list(rng.random(rng.integers(1, 4))) for _ in range(50)]
    x, y = metrics.explode_pairs(s, h)
    c = metrics.correlations(s, h)
    assert c["tau_c"] == pytest.approx(100 * stats.kendalltau(x, y, variant="c")[0])
    assert c["tau_b"] == pytest.approx(100 * stats.kendalltau(x, y, variant="b")[0])
    assert c["rho"] == pytest.approx(100 * stats.spearmanr(x, y)[0])
    gt = metrics.mean_human_score(h)
    low, high = s - 0.1, s + 0.2
    m = metrics.interval_metrics(gt, low, s, high)
    assert m["ups"] == 0.0  # constant width: UPS is defined as 0
    assert m["coverage"] == pytest.approx(100 * np.mean((gt >= low) & (gt <= high)))
    width = high - low + rng.random(50) * 0.1
    m = metrics.interval_metrics(gt, low, s, low + width)
    assert m["ups"] == pytest.approx(100 * stats.pearsonr(np.abs(gt - s), width)[0])
    assert metrics.quantile_levels([0.9]) == pytest.approx([0.05, 0.5, 0.95])


# ---------------------------------------------------------------- T-10 Stage 1 dataset and handler (B-01)
def _stage1_root(tmp_path, retrieval):
    (tmp_path / "Stage1").mkdir()
    img = tmp_path / "img.png"
    Image.new("RGB", (24, 24), "blue").save(img)
    n = 6
    df = pd.DataFrame({
        "image_path": [str(img)] * n, "caption": ["a blue square"] * n, "foil": ["a red circle"] * n,
        "tokenized_words": [["a", "blue", "square"]] * n, "pos_tagging": [["DET", "ADJ", "NOUN"]] * n,
        "retrieval_hard_foil_list": [retrieval] * n, "retrieval_medium_foil_list": [retrieval] * n,
        "retrieval_easy_foil_list": [retrieval] * n, "split": ["val"] * n, "task": ["stage1"] * n,
        "pair_id": list(range(n)),
    })
    joblib.dump(df, tmp_path / "Stage1" / "stage1.pkl")
    return str(tmp_path)


def test_stage1_dataset_and_handler(tmp_path):
    from beitscore.data.collate import merge_batch_tensors_by_dict_key
    from beitscore.stage1.dataset import BCEDataset
    from beitscore.stage1.handler import BCEHandler

    root = _stage1_root(tmp_path, ["a green triangle", "a dog"])
    ds = BCEDataset(benchmark="stage1", base_path=root, tokenizer=FakeTokenizer(), split="val")
    batch = merge_batch_tensors_by_dict_key([ds[0], ds[1]])
    assert batch["image"].shape[0] == 4 and batch["targets"].tolist() == [[1.0], [0.0], [1.0], [0.0]]

    handler = BCEHandler()
    loader = argparse.Namespace(dataset=ds)
    handler.before_eval(metric_logger=None, data_loader=loader)
    probs = torch.tensor([0.9, 0.2, 0.8, 0.6, 0.3, 0.4, 0.7, 0.1, 0.9, 0.5, 0.6, 0.55])
    handler.scores = [probs]
    handler.target_scores = [torch.tensor([1.0, 0.0] * 6)]
    res = handler.after_eval(return_results=True)
    assert res["acc_on_order"] == pytest.approx(5 / 6)
    assert 0 <= res["f1"] <= 1


# ---------------------------------------------------------------- T-11 error detection (C-15)
def test_error_detection_on_a_foil_style_table():
    errors = load_script("scripts/beitscore_v1/evaluate_errors.py", "evaluate_errors")
    df = pd.DataFrame({
        "pair_id": ["0", "1", "2", "0", "1", "2"], "foil": [False, False, False, True, True, True],
        "score": [0.9, 0.4, 0.8, 0.1, 0.6, 0.2], "data_type": ["n", "n", "v", "n", "n", "v"],
    })
    rows = {r["data_type"]: r for r in errors.error_results(df)}
    assert rows["all (micro)"]["accuracy"] == pytest.approx(2 / 3)
    assert rows["n"]["accuracy"] == pytest.approx(0.5) and rows["v"]["accuracy"] == 1.0
    assert metrics.winoground([1, 0], [0, 1], [0, 0], [1, 1])["group"] == pytest.approx(0.5)


# ---------------------------------------------------------------- T-12 no "<empty>" candidates (B-12)
def test_bce_dataset_never_offers_the_empty_marker(tmp_path, monkeypatch):
    from beitscore.stage1 import dataset as stage1_dataset
    from beitscore.stage1.dataset import BCEDataset, retrieval_candidates

    assert retrieval_candidates(["<empty>"]) == [] and retrieval_candidates([]) == []
    assert retrieval_candidates(float("nan")) == [] and retrieval_candidates(["a", "<empty>"]) == ["a"]

    root = _stage1_root(tmp_path, ["<empty>"])
    ds = BCEDataset(benchmark="stage1", base_path=root, tokenizer=FakeTokenizer(), split="val")
    offered = []
    real_choice = stage1_dataset.random.choice

    def spy(seq):
        offered.append(list(seq))
        return real_choice(seq)

    monkeypatch.setattr(stage1_dataset.random, "choice", spy)
    for i in range(len(ds)):
        ds[i]
    assert offered and all("<empty>" not in options for options in offered)


# ---------------------------------------------------------------- scripts run from the command line
@pytest.mark.parametrize("path", [
    "scripts/train.py", "scripts/evaluate.py", "scripts/beitscore_v1/evaluate_errors.py",
    "scripts/beitscore_pp/calibrate.py", "scripts/beitscore_v1/caparena.py", "scripts/download_release.py",
    "scripts/get_images.py", "scripts/model_cost.py",
])
def test_script_help_runs(path):
    """The scripts import the package when run as files (the scripts folder must not shadow it)."""
    env = dict(os.environ, BEITSCORE_OUTPUT=os.environ.get("TMPDIR", "/tmp"))
    out = subprocess.run([sys.executable, str(REPO / path), "--help"], capture_output=True, text=True, env=env,
                         cwd=REPO, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]

"""Smoke tests: every script compiles, entry points import, defaults match the paper."""
import py_compile
import sys

import pytest

from conftest import REPO, load_script

SCRIPTS = sorted(p.relative_to(REPO) for p in (REPO / "scripts").rglob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=str)
def test_script_compiles(path):
    py_compile.compile(str(REPO / path), doraise=True)


@pytest.mark.parametrize("path", [
    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/beitscore_pp/calibrate.py",
    "scripts/beitscore_v1/caparena.py",
    "scripts/beitscore_v1/evaluate_errors.py",
    "scripts/download_release.py",
    "scripts/get_images.py",
])
def test_entry_point_imports(path):
    load_script(path, path.replace("/", "_")[:-3])


@pytest.fixture
def train_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    return load_script("scripts/train.py", "train_script").get_args()


@pytest.mark.parametrize("task", ["stage1", "regression", "quant"])
def test_train_accepts_every_task(monkeypatch, task):
    monkeypatch.setattr(sys, "argv", ["train.py", "--benchmark", task])
    assert load_script("scripts/train.py", "train_" + task).get_args().benchmark == task


def test_train_defaults_shared_by_both_metrics(train_defaults):
    assert train_defaults.batch_size == 16
    assert train_defaults.quantiles == [0.9, 0.85, 0.8, 0.75, 0.7]
    assert train_defaults.seed == 0


@pytest.fixture
def quant_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--benchmark", "quant"])
    return load_script("scripts/train.py", "train_quant_defaults").get_args()


def test_beitscorepp_defaults_match_the_paper(quant_defaults):
    """C-01, C-02, C-03, C-23: running --benchmark quant without flags trains the paper model."""
    assert quant_defaults.median_weight == 1.0
    assert quant_defaults.bias == -2.5
    assert quant_defaults.early_stopping == 15
    assert quant_defaults.chained_construction is True
    assert quant_defaults.use_clusters is True


@pytest.mark.parametrize("task", ["stage1", "regression"])
def test_beitscore_defaults_are_unchanged(monkeypatch, task):
    monkeypatch.setattr(sys, "argv", ["train.py", "--benchmark", task])
    args = load_script("scripts/train.py", "train_bs_" + task).get_args()
    assert args.early_stopping == 5
    assert args.use_clusters is False


def test_ablation_flags_switch_the_paper_settings_off(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py", "--benchmark", "quant", "--no-chained_construction",
                                      "--no-use_clusters", "--coef", "0", "--early_stopping", "5"])
    args = load_script("scripts/train.py", "train_ablation").get_args()
    assert (args.chained_construction, args.use_clusters, args.median_weight, args.early_stopping) == (False, False, 0.0, 5)

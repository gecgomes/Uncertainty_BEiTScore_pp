"""BEiTScore Stage 1: model registration, negative augmentation, handler."""
import copy
import random

import pytest
from timm.models import is_model

import beitscore.models  # noqa: F401
from beitscore.engine import TaskHandler
from beitscore.stage1.dataset import BCEDataset, augment_negatives
from beitscore.stage1.handler import BCEHandler
from beitscore.data.datasets import BEiTDataset

WORDS = "a small brown dog runs across the green grass near a red ball".split()
POS = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN", "ADP", "DET", "ADJ", "NOUN"]


@pytest.mark.parametrize("size", ["base", "large"])
@pytest.mark.parametrize("task", ["stage1", "regression", "quant"])
def test_all_model_configs_are_registered(size, task):
    assert is_model(f"beit3_{size}_patch16_480_{task}")


def test_stage1_classes_plug_into_the_shared_code():
    assert issubclass(BCEDataset, BEiTDataset)
    assert issubclass(BCEHandler, TaskHandler)


def test_augment_negatives_returns_a_different_caption():
    changed = 0
    for seed in range(50):
        out = augment_negatives(list(WORDS), list(POS), seed=seed)
        assert isinstance(out, str) and out
        changed += out != " ".join(WORDS)
    assert changed >= 45


def test_augment_negatives_is_reproducible_with_a_seed():
    a = augment_negatives(list(WORDS), list(POS), seed=7)
    b = augment_negatives(list(WORDS), list(POS), seed=7)
    assert a == b


def test_augment_negatives_does_not_modify_its_input():
    words, pos = list(WORDS), list(POS)
    original = copy.deepcopy(words)
    for seed in range(20):
        augment_negatives(words, pos, seed=seed)
    assert words == original


def test_augment_negatives_handles_short_captions():
    random.seed(0)
    for _ in range(20):
        out = augment_negatives(["a", "dog"], ["DET", "NOUN"])
        assert isinstance(out, str)

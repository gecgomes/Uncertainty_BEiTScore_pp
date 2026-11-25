import importlib

import numpy as np
import pytest
import torch

from beitscore.data.collate import merge_batch_tensors_by_dict_key
from beitscore.data.sampler import ClusterBatchSampler


class _ClusteredDataset:
    def __init__(self, clusters):
        self.clusters = clusters


def _sampler(batch_size=4):
    rng = np.random.default_rng(0)
    clusters = rng.integers(0, 5, 103).tolist()
    return clusters, ClusterBatchSampler(_ClusteredDataset(clusters), batch_size=batch_size)


def test_batches_come_from_a_single_cluster_and_cover_everything():
    clusters, sampler = _sampler()
    seen = []
    for batch in sampler:
        assert len({clusters[i] for i in batch}) == 1
        seen.extend(batch)
    assert sorted(seen) == list(range(len(clusters)))


def test_sampler_len_matches_iteration():
    _, sampler = _sampler()
    assert len(sampler) == len(list(iter(sampler)))


def test_collate_pads_variable_length_targets_and_tokens():
    batch = [
        dict(image=torch.zeros(3, 4, 4), language_tokens=[0, 5, 6, 2], padding_mask=[0, 0, 0, 0],
             targets=torch.tensor([0.2, 0.4, 0.6])),
        dict(image=torch.zeros(3, 4, 4), language_tokens=[0, 7, 2], padding_mask=[0, 0, 0],
             targets=torch.tensor([0.9])),
    ]
    out = merge_batch_tensors_by_dict_key(batch)
    assert out["image"].shape == (2, 3, 4, 4)
    assert out["language_tokens"].tolist() == [[0, 5, 6, 2], [0, 7, 2, 1]]
    assert out["padding_mask"].tolist() == [[0, 0, 0, 0], [0, 0, 0, 1]]
    assert out["targets"][1].tolist() == pytest.approx([0.9, -999, -999])
    assert out["targets_mask"].tolist() == [[True, True, True], [True, False, False]]


def test_paths_are_configurable(monkeypatch, tmp_path):
    import beitscore.paths as paths
    monkeypatch.setenv("BEITSCORE_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("BEITSCORE_OUTPUT", str(tmp_path / "o"))
    reloaded = importlib.reload(paths)
    assert reloaded.DATA_ROOT == str(tmp_path / "d")
    assert reloaded.FINETUNE_ROOT == str(tmp_path / "o" / "finetuning")
    monkeypatch.undo()
    importlib.reload(paths)

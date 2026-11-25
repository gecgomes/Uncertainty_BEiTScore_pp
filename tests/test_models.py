import argparse

import numpy as np
import torch
from scipy.stats import norm

import beitscore.models  # noqa: F401
from beitscore.models.checkpoint import load_model_and_may_interpolate_beitscorepp
from beitscore.models.heads import BEiT3ForQuantileRegression, BEiT3ForRegression

CI_LEVELS = [0.9, 0.85, 0.8, 0.75, 0.7]


def test_chained_head_is_monotone_and_bounded(tiny_cfg, tiny_batch):
    torch.manual_seed(0)
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True).eval()
    with torch.no_grad():
        y = model(**tiny_batch)
    assert y.shape == (4, 2 * len(CI_LEVELS) + 1)
    assert torch.all(y.diff(dim=-1) >= 0), "quantiles must be non-decreasing"
    assert torch.all((y >= 0) & (y <= 1))


def test_chained_head_monotone_for_extreme_logits(tiny_cfg, tiny_batch):
    """Monotonicity must hold by construction, whatever the head weights are."""
    torch.manual_seed(1)
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True).eval()
    with torch.no_grad():
        for pool in (model.low_quantile_pool, model.mid_quantile_pool, model.high_quantile_pool):
            pool.weight.normal_(0, 5.0)
            pool.bias.normal_(0, 5.0)
        y = model(**tiny_batch)
    assert torch.all(y.diff(dim=-1) >= 0)


def test_median_is_the_middle_output(tiny_cfg, tiny_batch):
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True).eval()
    with torch.no_grad():
        y = model(**tiny_batch)
        cls = model.pooler(model.beit3(textual_tokens=tiny_batch["text_description"],
                                       visual_tokens=tiny_batch["image"],
                                       text_padding_position=tiny_batch["padding_mask"])["encoder_out"])
        q_mid = torch.sigmoid(model.mid_quantile_pool(model.trunk(cls))).squeeze(-1)
    torch.testing.assert_close(y[:, len(CI_LEVELS)], q_mid)


def test_unconstrained_head_shape(tiny_cfg, tiny_batch):
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=False).eval()
    with torch.no_grad():
        y = model(**tiny_batch)
    assert y.shape == (4, 11)


def test_regression_model_output_range(tiny_cfg, tiny_batch):
    model = BEiT3ForRegression(tiny_cfg, num_classes=1).eval()
    with torch.no_grad():
        y = model(**tiny_batch, only_inference=True)
    assert y.shape == (4, 1)
    assert torch.all((y >= 0) & (y <= 1))


def _beitscore_checkpoint(tiny_cfg, tmp_path):
    torch.manual_seed(2)
    beitscore = BEiT3ForRegression(tiny_cfg, num_classes=1)
    path = tmp_path / "beitscore.pth"
    torch.save({"model": beitscore.state_dict()}, path)
    return beitscore.state_dict(), str(path)


def test_quantile_head_is_initialised_from_beitscore(tiny_cfg, tmp_path):
    src, path = _beitscore_checkpoint(tiny_cfg, tmp_path)
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True)
    load_model_and_may_interpolate_beitscorepp(argparse.Namespace(bias=-2.5), path, model, "model|module", "")
    sd = model.state_dict()
    for i in (0, 1):
        torch.testing.assert_close(sd[f"trunk.{i}.weight"], src[f"head.{i}.weight"])
        torch.testing.assert_close(sd[f"trunk.{i}.bias"], src[f"head.{i}.bias"])
    torch.testing.assert_close(sd["mid_quantile_pool.weight"], src["head.3.weight"])
    torch.testing.assert_close(sd["mid_quantile_pool.bias"], src["head.3.bias"])
    for pool in ("low_quantile_pool", "high_quantile_pool"):
        torch.testing.assert_close(sd[f"{pool}.weight"], src["head.3.weight"].repeat(5, 1))
    for k in src:
        if k.startswith(("beit3.", "pooler.")):
            torch.testing.assert_close(sd[k], src[k])


def test_delta_bias_init_is_probit_scaled(tiny_cfg, tmp_path):
    """Pins down the --bias initialisation used for the released checkpoints (C-02).

    The offset is probit-scaled and has opposite signs on the two sides, so
    with --bias -2.5 the lower deltas are pushed UP (+1.58 .. +2.5) and the
    upper deltas DOWN (-1.58 .. -2.5). This is not a uniform "subtract 2.5".
    If the initialisation is changed, update this test, the docstring of
    quantile_bias_offsets and the README together.
    """
    src, path = _beitscore_checkpoint(tiny_cfg, tmp_path)
    model = BEiT3ForQuantileRegression(tiny_cfg, chained_construction=True)
    load_model_and_may_interpolate_beitscorepp(argparse.Namespace(bias=-2.5), path, model, "model|module", "")
    base = src["head.3.bias"][0]
    low_off = (model.low_quantile_pool.bias.detach() - base).numpy()
    high_off = (model.high_quantile_pool.bias.detach() - base).numpy()

    low_q = sorted((1 - p) / 2 for p in CI_LEVELS)
    high_q = sorted(1 - (1 - p) / 2 for p in CI_LEVELS)
    lp, hp = norm.ppf(low_q), norm.ppf(high_q)
    np.testing.assert_allclose(low_off, np.sort(lp / np.abs(lp).max() * -2.5), atol=1e-5)
    np.testing.assert_allclose(high_off, hp / np.abs(hp).max() * -2.5, atol=1e-5)
    assert np.all(low_off > 0) and np.all(high_off < 0)

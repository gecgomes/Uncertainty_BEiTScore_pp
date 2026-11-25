import numpy as np
import pytest

from beitscore.pp.calibration.bounds import hoeffding_UCB
from beitscore.pp.calibration.intervals import interval_fn3
from beitscore.pp.calibration.risk_control import (
    confromal_risk_control_using_CI_and_WB,
    interval_fn3 as interval_fn3_stacked,
)


def _synthetic(n, seed):
    """Median predictions with noise and intervals that are too narrow."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.15, 0.85, n)
    mid = np.clip(y + rng.normal(0, 0.08, n), 0, 1)
    half = rng.uniform(0.02, 0.08, n)
    low, high = np.clip(mid - half, 0, 1), np.clip(mid + half, 0, 1)
    return low, mid, high, y


def _coverage(low, high, y):
    return float(np.mean((y >= low) & (y <= high)))


@pytest.mark.parametrize("rhat", [0.0, 0.05, 0.2])
def test_ucb_is_above_empirical_risk_and_shrinks_with_n(rhat):
    small = hoeffding_UCB(rhat, 200, 0.05)
    large = hoeffding_UCB(rhat, 20000, 0.05)
    assert small >= rhat and large >= rhat
    assert large <= small


def test_identity_transform():
    low, mid, high, _ = _synthetic(100, 0)
    lo, m, hi = interval_fn3(low, mid, high, b=0.0, l=1.0)
    np.testing.assert_allclose(lo, low)
    np.testing.assert_allclose(hi, high)
    np.testing.assert_allclose(m, mid)


def test_transform_width_is_monotone_in_both_parameters():
    low, mid, high, _ = _synthetic(500, 1)
    widths_l = [np.mean(np.subtract(*interval_fn3(low, mid, high, b=0.0, l=l)[::-2])) for l in (0.5, 1, 2, 4)]
    widths_b = [np.mean(np.subtract(*interval_fn3(low, mid, high, b=b, l=1.0)[::-2])) for b in (0, 0.05, 0.1, 0.2)]
    assert np.all(np.diff(widths_l) >= 0)
    assert np.all(np.diff(widths_b) >= 0)


def test_two_copies_of_the_transform_agree():
    """risk_control.interval_fn3 (stacked input) and intervals.interval_fn3 must match."""
    low, mid, high, y = _synthetic(200, 2)
    stacked = np.stack([low, mid, high], axis=1)
    a = interval_fn3_stacked(stacked, y, 0.03, 1.7)
    b = interval_fn3(low, mid, high, b=0.03, l=1.7)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[2])


@pytest.mark.slow
def test_calibration_reaches_target_on_fresh_data():
    """End to end: calibrate on one sample, check coverage on another."""
    low, mid, high, y = _synthetic(3000, 3)
    assert _coverage(low, high, y) < 0.9  # starts under-covering
    lam, b = confromal_risk_control_using_CI_and_WB(
        np.stack([low, mid, high], axis=1), y, ci=0.9, delta=0.05, n=len(y))

    tl, tm, th, ty = _synthetic(20000, 4)
    lo, _, hi = interval_fn3(tl, tm, th, b=b, l=lam)
    assert _coverage(lo, hi, ty) >= 0.9


def test_already_covering_intervals_are_left_unchanged():
    low, mid, high, y = _synthetic(1000, 5)
    lo, hi = np.zeros_like(low), np.ones_like(high)
    lam, b = confromal_risk_control_using_CI_and_WB(
        np.stack([lo, mid, hi], axis=1), y, ci=0.9, delta=0.05, n=len(y))
    assert (lam, b) == (1, 0)


def test_bound_choice_orders_and_is_recorded():
    import pytest
    from beitscore.pp.calibration.bounds import ucb
    from beitscore.pp.calibration.clustered import calibrate
    h, b, hb = (ucb(0.06, 250, 0.05, k) for k in ("hoeffding", "bentkus", "hb"))
    assert hb <= min(h, b) + 1e-9 and hb >= 0.06
    with pytest.raises(ValueError):
        ucb(0.06, 250, 0.05, "nope")
    rng = np.random.default_rng(0)
    n = 1200
    gt = rng.uniform(0, 1, n)
    mid = np.clip(gt + rng.normal(0, 0.1, n), 0, 1)
    low, high = np.clip(mid - 0.1, 0, 1), np.clip(mid + 0.1, 0, 1)
    clusters = rng.integers(0, 3, n)
    widths = {}
    for k in ("hoeffding", "bentkus", "hb"):
        res = calibrate(low, mid, high, gt, clusters=clusters, n_candidates=40, bound=k)
        assert res.bound == k
        assert all(c.coverage >= 0.9 for c in res.clusters)
        widths[k] = res.width
    assert widths["hb"] <= min(widths["hoeffding"], widths["bentkus"]) + 1e-6
    with pytest.raises(ValueError):
        calibrate(low, mid, high, gt, clusters=clusters, bound="nope")

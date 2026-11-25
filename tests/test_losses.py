import pytest
import torch

from beitscore.pp.handler import QuantHandler
from beitscore.pp.losses import CenteredIntervalLoss, PinBallLoss

PRED = torch.tensor([[0.2, 0.5, 0.8],
                     [0.4, 0.5, 0.6],
                     [0.1, 0.3, 0.35]])
Y = torch.tensor([0.5, 0.9, 0.05])


def _centered_interval(pred, y, alpha_half, median_weight):
    lo, mid, hi = pred[:, 0], pred[:, 1], pred[:, 2]
    width = hi - lo
    below = torch.where(y < lo, (lo - y) / alpha_half, torch.zeros_like(y))
    above = torch.where(y > hi, (y - hi) / alpha_half, torch.zeros_like(y))
    return (width + below + above).mean() + median_weight * (mid - y).abs().mean()


@pytest.mark.parametrize("coef", [0.0, 1.0])
def test_centered_interval_loss_matches_hand_computation(coef):
    loss = CenteredIntervalLoss(CI=[0.9], coef=coef)
    expected = _centered_interval(PRED, Y, alpha_half=0.05, median_weight=coef)
    torch.testing.assert_close(loss(PRED, Y), expected)


def test_weight_equals_two_over_alpha():
    """Paper Eq. 1 uses 2/alpha with alpha = 1 - CI; the code uses 1/(alpha/2)."""
    loss = CenteredIntervalLoss(CI=[0.9])
    assert torch.isclose(1 / loss.alpha[0], torch.tensor(2 / 0.1))


def test_centered_interval_loss_sums_over_intervals():
    pred = torch.tensor([[0.1, 0.2, 0.5, 0.7, 0.9]])
    y = torch.tensor([0.95])
    loss = CenteredIntervalLoss(CI=[0.9, 0.8], coef=0.0)
    # outer pair (0.1, 0.9) for 90%, inner pair (0.2, 0.7) for 80%
    expected = (0.8 + (0.95 - 0.9) / 0.05) + (0.5 + (0.95 - 0.7) / 0.10)
    torch.testing.assert_close(loss(pred, y), torch.tensor(expected))


def test_pinball_loss_matches_hand_computation():
    loss = PinBallLoss(CI=[0.9])
    qs = [0.05, 0.5, 0.95]
    expected = 0
    for i, q in enumerate(qs):
        d = Y - PRED[:, i]
        expected = expected + torch.where(d >= 0, q * d, (q - 1) * d)
    torch.testing.assert_close(loss(PRED, Y), expected.mean())


def test_median_weight_defaults_to_one():
    assert CenteredIntervalLoss(CI=[0.9]).coef == 1


def test_handler_passes_median_weight_to_loss():
    handler = QuantHandler(CI=[0.9], coef=1.0, criterion="CI")
    assert handler.criterion.coef == 1.0

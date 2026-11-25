"""Conformal risk control over the interval parameters (scale and bias)."""
from .bounds import ucb
from scipy.optimize import brentq
import numpy as np
from tqdm import tqdm


def interval_fn2(predictions,Y,l):
    lower = np.maximum(predictions[:, 0] - l,0)
    upper = np.minimum(predictions[:, 2] + l,1)
    return lower, upper

def interval_fn3(predictions,Y,b,l):
    lower = np.maximum((predictions[:, 0] - predictions[:, 1])*l + predictions[:, 1] - b,0)
    upper = np.minimum((predictions[:, 2] - predictions[:, 1])*l + predictions[:, 1] + b,1)
    return lower, upper

def risk_coverage_fn(predictions, Y, l,b):
    lower, upper = interval_fn3(predictions=predictions,Y=Y,l=l, b=b)
    covered = (lower <= Y) & (Y <= upper)
    
    return 1-covered.mean()

def risk_coverage_fn_for_bias(predictions, Y, l):
    lower, upper = interval_fn2(predictions=predictions,Y=Y,l=l)
    covered = (lower <= Y) & (Y <= upper)
    
    return 1-covered.mean()

def _feasible_root(fn, lo, hi, maxiters):
    """Root of a decreasing step function, moved to the side where fn <= 0.

    The empirical risk changes in steps, so brentq returns a point within its tolerance
    of a jump, possibly just before it, where the bound is still above the target.
    Moving right by a few ulps puts the parameter on the side that meets the target.
    """
    x = brentq(fn, lo, hi, maxiter=maxiters)
    step = max(abs(x), 1.0) * 1e-12
    while fn(x) > 0 and x < hi:
        x = min(x + step, hi)
        step *= 2
    return x


def conformal_risk_control_using_concentration_inequalities_l(prediction_low_high,Y,risk_fn,b,ci,delta,n,maxiters =int(1e8),
                                                            bound="hoeffding"):
    def _get_UCB(l):
        Rhat =  risk_fn(prediction_low_high, Y, l, b)
        return ucb(Rhat, n, delta, bound, maxiters) - (1-ci)

    return _feasible_root(_get_UCB, 0, 10, maxiters)

def conformal_risk_control_using_concentration_inequalities_b(prediction_low_high,Y,risk_fn,ci,delta,n,maxiters =int(1e8),
                                                            bound="hoeffding"):
    def _get_UCB(b):
        Rhat =  risk_fn(prediction_low_high, Y, 0, b)
        return ucb(Rhat, n, delta, bound, maxiters) - (1-ci)

    return _feasible_root(_get_UCB, 0, 1, maxiters)

N_BIAS_CANDIDATES = 1000


def confromal_risk_control_using_CI_and_WB(prediction_low_high, Y, ci, delta, n, maxiters=int(1e8), strict=False,
                                           n_candidates=N_BIAS_CANDIDATES, progress=True, bound="hoeffding"):
    """Choose the scale omega (l) and bias beta (b) of interval_fn3 so that the Hoeffding
    upper confidence bound on the miscoverage stays below 1 - ci (Bates et al., 2021).
    bound selects that bound: "hoeffding" (default), "bentkus" or "hb" (the minimum of both).

    prediction_low_high: (n, 3) array of [low, mid, high]; Y: (n,) targets; n: sample size
    used in the bound. Among the candidate biases b in [0, b_high] (n_candidates values),
    each gets the smallest valid scale l, and the pair with the narrowest mean interval is
    kept. Returns (l, b); (1, 0) when the raw intervals already meet the target.

    strict=False reproduces the paper: every candidate uses delta, and the "already
    calibrated" check compares the empirical risk with 1 - ci. strict=True makes the
    guarantee hold after the search: each candidate uses delta / n_candidates (union
    bound), and the check uses the upper confidence bound.
    """
    delta_c = delta / n_candidates if strict else delta
    raw_risk = risk_coverage_fn_for_bias(prediction_low_high, Y, 0)
    if strict:
        needs_calibration = ucb(raw_risk, n, delta_c, bound, maxiters) > (1 - ci)
    else:
        needs_calibration = raw_risk > (1 - ci)
    if not needs_calibration:
        return 1, 0

    def _get_UCB_B(l):
        Rhat = risk_coverage_fn_for_bias(prediction_low_high, Y, l)
        return ucb(Rhat, n, delta_c, bound, maxiters) - (1 - ci)

    # check if there is quantile crossing: then only a bias around the median is used
    if any((prediction_low_high[:, 2] - prediction_low_high[:, 0]) < 0):
        return 0, conformal_risk_control_using_concentration_inequalities_b(
            prediction_low_high, Y, risk_coverage_fn, ci, delta_c, n, maxiters, bound=bound)

    b_high = brentq(_get_UCB_B, 0, 1, maxiter=maxiters)
    best_sharpness, best_b, best_l = np.inf, None, None
    candidates = np.linspace(0, b_high, n_candidates)
    for b in (tqdm(candidates) if progress else candidates):
        try:
            l = conformal_risk_control_using_concentration_inequalities_l(
                prediction_low_high, Y, risk_coverage_fn, b, ci, delta_c, n, maxiters, bound=bound)
        except ValueError:
            # no scale in [0, 10] reaches the target with this bias (no sign change)
            continue
        new_interval = interval_fn3(prediction_low_high, Y, b, l)
        sharpness = (new_interval[1] - new_interval[0]).mean()
        if sharpness < best_sharpness:
            best_sharpness, best_b, best_l = sharpness, b, l
    if best_l is None:
        # every candidate failed: fall back to a bias-only interval around the median
        return 0, conformal_risk_control_using_concentration_inequalities_b(
            prediction_low_high, Y, risk_coverage_fn, ci, delta_c, n, maxiters, bound=bound)
    return best_l, best_b


# Correctly spelled alias.
conformal_risk_control_scale_and_bias = confromal_risk_control_using_CI_and_WB

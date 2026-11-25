"""Evaluation metrics shared by the training handlers and the scripts.

Correlations are reported x100, as in both papers. Human scores may hold several
annotations per instance; correlations then pair every annotation with the
instance's score (aggregation method A of the CLIPScore paper), while interval
metrics use the mean annotation.
"""
import numpy as np
import scipy.stats as stats
from sklearn.metrics import average_precision_score, roc_auc_score


def mean_human_score(human_scores):
    """One value per instance: the mean of its annotations."""
    return np.array([np.mean(h) if isinstance(h, (list, tuple, np.ndarray)) else float(h) for h in human_scores],
                    dtype=float)


def explode_pairs(scores, human_scores):
    """Repeat each score once per annotation; return two aligned 1-D arrays."""
    xs, ys = [], []
    for s, h in zip(scores, human_scores):
        values = list(h) if isinstance(h, (list, tuple, np.ndarray)) else [h]
        xs.extend([float(s)] * len(values))
        ys.extend(float(v) for v in values)
    return np.asarray(xs), np.asarray(ys)


def correlations(scores, human_scores):
    """Kendall tau-b, tau-c and Spearman rho between scores and human judgements, x100."""
    x, y = explode_pairs(scores, human_scores)
    return {
        "tau_b": 100 * stats.kendalltau(x, y, variant="b")[0],
        "tau_c": 100 * stats.kendalltau(x, y, variant="c")[0],
        "rho": 100 * stats.spearmanr(x, y)[0],
    }


def _safe(fn, x, y):
    return 100 * fn(x, y)[0] if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0


def interval_metrics(gt, low, mid, high):
    """Coverage (%), sharpness (mean width), UPS and its rank variants (x100), MAE.

    gt holds one value per instance (use mean_human_score). UPS is the Pearson
    correlation between the absolute error of the median and the interval width.
    """
    gt, low, mid, high = (np.asarray(a, dtype=float) for a in (gt, low, mid, high))
    width = high - low
    error = np.abs(gt - mid)
    return {
        "coverage": 100 * float(np.mean((gt >= low) & (gt <= high))),
        "sharpness": float(np.mean(width)),
        "sharpness_std": float(np.std(width)),
        "ups": _safe(stats.pearsonr, error, width),
        "uss": _safe(stats.spearmanr, error, width),
        "uks": _safe(stats.kendalltau, error, width),
        "mae": float(np.mean(error)),
    }


def quantile_levels(confidence_levels):
    """Quantile levels of the 2k+1 outputs, sorted: [a_1/2, ..., 0.5, ..., 1 - a_1/2]."""
    levels = [0.5]
    for c in confidence_levels:
        levels += [(1 - c) / 2, 1 - (1 - c) / 2]
    return sorted(levels)


def quantile_ece(quantiles, gt, levels):
    """Mean |P(gt < q_i) - level_i| over the quantile outputs (columns sorted by level)."""
    quantiles = np.asarray(quantiles, dtype=float)
    gt = np.asarray(gt, dtype=float)
    return float(np.mean([abs(np.mean(gt < quantiles[:, i]) - q) for i, q in enumerate(levels)]))


def error_detection(correct_scores, incorrect_scores):
    """Pairwise accuracy (correct caption scored above its incorrect version), AUROC, AP."""
    correct = np.asarray(correct_scores, dtype=float)
    incorrect = np.asarray(incorrect_scores, dtype=float)
    target = np.r_[np.ones(len(correct)), np.zeros(len(incorrect))]
    pred = np.r_[correct, incorrect]
    return {
        "accuracy": float(np.mean(correct > incorrect)),
        "auroc": float(roc_auc_score(target, pred)),
        "ap": float(average_precision_score(target, pred)),
    }


def winoground(p00, p01, p10, p11):
    """Winoground text, image and group scores.

    pij is the score of the pair tagged "ij" in winoground.pkl; the text score compares
    p00 with p01 and p11 with p10, the image score p00 with p10 and p11 with p01.
    """
    p00, p01, p10, p11 = (np.asarray(a, dtype=float) for a in (p00, p01, p10, p11))
    text = (p00 > p01) & (p11 > p10)
    image = (p00 > p10) & (p11 > p01)
    return {"text": float(np.mean(text)), "image": float(np.mean(image)), "group": float(np.mean(text & image))}


def bimodality_coefficient(values):
    """
    > 0.555 → bimodal (collapsed to poles)
    < 0.555 → unimodal (spread or single-peaked)
    Max = 1.0 (perfect two-point distribution)
    """
    n = len(values)
    skew = stats.skew(values)
    kurt = stats.kurtosis(values)  # excess kurtosis
    bc = (skew**2 + 1) / (kurt + 3 * ((n - 1)**2 / ((n - 2) * (n - 3))))
    return float(bc)

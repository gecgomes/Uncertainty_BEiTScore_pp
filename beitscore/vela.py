"""LongCap-Arena, the long-caption benchmark of VELA (Matsuda et al., EMNLP 2025).

Every caption of LongCap-Arena is rated on three dimensions: descriptiveness (desc), relevance
(rel) and fluency (flu), each in [0, 1]. The benchmark file ($BEITSCORE_DATA/vela/vela.pkl, written
by scripts/prepare_data/benchmarks.py download vela) holds one row per image-caption pair with
its three ratings and its group (`subset`).

The evaluation groups are those of the VELA paper: dci_val (TestA, 98 pairs) and dci_test
(TestB, 108 pairs), each reported separately. Besides the three dimensions, a metric is compared
with three single human scores that combine them; the principal component they rely on is fitted
on the ratings of the train and val groups only (no metric score is needed there):

    pca                    (x - mean) . c, with c the first principal component of the centred
                           ratings, signed so that its coefficients sum to a positive value
    pca_harmonic_mean      H = 1 / sum_d (w_d / x_d), the harmonic mean of the three ratings
                           weighted by w = |c| / sum |c| (0 when a rating is 0)
    pca_harmonic_minimum   (H + min_d x_d) / 2

Only numpy is needed, so the baselines' environments can import this module too.
"""
import numpy as np

DIMS = ("desc", "rel", "flu")
EVAL_GROUPS = ("dci_val", "dci_test")
FIT_GROUPS = ("train", "val")
GROUP_NAMES = {"dci_val": "TestA", "dci_test": "TestB"}
TARGETS = ("desc", "rel", "flu", "pca", "pca_harmonic_mean", "pca_harmonic_minimum")
DEFAULT_TARGETS = ("pca", "pca_harmonic_mean", "pca_harmonic_minimum")
TARGET_NAMES = {"desc": "Desc.", "rel": "Rel.", "flu": "Flu.", "pca": "PCA",
                "pca_harmonic_mean": "PCA-HM", "pca_harmonic_minimum": "PCA-HMin"}


def ratings(df):
    """The (n, 3) array of (desc, rel, flu) ratings of a frame of vela.pkl."""
    return df[list(DIMS)].to_numpy(dtype=float)


def fit_pca(x):
    """First principal component of the centred (not standardised) ratings x (n, 3), signed so
    that its coefficients sum to a positive value (the largest absolute coefficient positive on a
    tie). Returns the mean, the component and the weights |c| / sum |c|."""
    x = np.asarray(x, dtype=float)
    mean = x.mean(axis=0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    c = vt[0].copy()
    s = float(c.sum())
    if s < -1e-12 or (abs(s) <= 1e-12 and c[np.argmax(np.abs(c))] < 0):
        c *= -1
    return {"mean": mean, "component": c, "weights": np.abs(c) / np.abs(c).sum()}


def weighted_harmonic(x, w):
    """Weighted harmonic mean 1 / sum_d (w_d / x_d) of each row of x; 0 when a dimension with a
    positive weight is rated 0."""
    x = np.asarray(x, dtype=float)
    out = np.zeros(len(x))
    active = np.asarray(w) > 0
    ok = np.all(x[:, active] > 0, axis=1)
    out[ok] = 1.0 / np.sum(np.asarray(w)[active] / x[ok][:, active], axis=1)
    return out


def targets(x, fit):
    """{target: one human score per row} for the (n, 3) ratings x, with the PCA fit of fit_pca."""
    x = np.asarray(x, dtype=float)
    hm = weighted_harmonic(x, fit["weights"])
    return {"desc": x[:, 0], "rel": x[:, 1], "flu": x[:, 2],
            "pca": (x - fit["mean"]) @ fit["component"],
            "pca_harmonic_mean": hm,
            "pca_harmonic_minimum": (hm + x.min(axis=1)) / 2}


def fit_from_frame(df, fit_groups=FIT_GROUPS):
    """PCA fit on the rows of vela.pkl whose subset is one of fit_groups."""
    fit_rows = df[df["subset"].isin(fit_groups)]
    if fit_rows.empty:
        raise ValueError(f"no rows of {fit_groups} in the VELA file; prepare it again with "
                         f"scripts/prepare_data/benchmarks.py download vela")
    return fit_pca(ratings(fit_rows))


def describe_fit(fit, fit_groups=FIT_GROUPS):
    return {"fit_groups": list(fit_groups),
            "component": dict(zip(DIMS, np.round(fit["component"], 4).tolist())),
            "weights": dict(zip(DIMS, np.round(fit["weights"], 4).tolist()))}

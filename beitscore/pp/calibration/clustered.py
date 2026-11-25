"""Calibration of the BEiTScore++ intervals, with or without image clusters, and the
calibration.json file that stores the result.

Used by scripts/beitscore_pp/calibrate.py, scripts/evaluate.py and the scoring API, so that all of them apply the same rule.

Clustered calibration (default): the risk control runs inside every cluster of the
calibration set that has at least `min_cluster_size` instances, using the cluster size
as n in the bound. Each cluster yields its own pair (omega, beta). The pair kept is the
most conservative one: the pair whose intervals are widest on average when applied to
the whole calibration set. One pair is then applied to every instance.
"""
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from .bounds import BOUNDS
from .intervals import interval_fn3
from .risk_control import N_BIAS_CANDIDATES, confromal_risk_control_using_CI_and_WB

CALIBRATION_FORMAT = 1


@dataclass
class ClusterResult:
    """One cluster of the calibration set.

    Coverage and width are measured on the cluster's own instances: `raw_*` for the
    uncalibrated interval, `coverage`/`width` with the cluster's own (omega, beta), and
    `coverage_selected`/`width_selected` with the pair finally kept. The bound guarantees
    that `coverage` is at least the target with probability 1 - delta, so it can sit
    slightly above or below the target on these same instances. `width_on_calibration_set`
    is the mean width of the cluster's pair on the whole calibration set, the quantity
    compared by the most-conservative selection.
    """
    cluster: int
    size: int
    omega: float
    beta: float
    width_on_calibration_set: float
    raw_coverage: Optional[float] = None
    raw_width: Optional[float] = None
    coverage: Optional[float] = None
    width: Optional[float] = None
    coverage_selected: Optional[float] = None
    width_selected: Optional[float] = None


@dataclass
class Calibration:
    omega: float
    beta: float
    target_coverage: float = 0.90
    delta: float = 0.05
    bound: str = "hoeffding"
    strict: bool = False
    clustered: bool = True
    n_clusters: Optional[int] = None
    selected_cluster: Optional[int] = None
    min_cluster_size: int = 30
    n_calibration: int = 0
    selection: str = "most_conservative"
    raw_coverage: Optional[float] = None
    raw_width: Optional[float] = None
    coverage: Optional[float] = None
    width: Optional[float] = None
    clusters: List[ClusterResult] = field(default_factory=list)
    model: str = ""
    cluster_file: str = ""
    cluster_file_sha256: str = ""
    format: int = CALIBRATION_FORMAT

    def apply(self, low, mid, high):
        """Calibrated (low, mid, high), Equations 3 and 4 of the paper."""
        return interval_fn3(low=np.asarray(low, dtype=float), mid=np.asarray(mid, dtype=float),
                            high=np.asarray(high, dtype=float), b=self.beta, l=self.omega)

    def save(self, path):
        folder = os.path.dirname(os.path.abspath(path))
        os.makedirs(folder, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        if data.get("format", 1) != CALIBRATION_FORMAT:
            raise ValueError(f"{path}: calibration format {data.get('format')} is not supported")
        data["clusters"] = [ClusterResult(**c) for c in data.get("clusters", [])]
        return cls(**data)


IDENTITY = Calibration(omega=1.0, beta=0.0, clustered=False)


def coverage_and_width(low, high, gt):
    """Empirical coverage and mean width of the intervals [low, high]."""
    if len(gt) == 0:
        return float("nan"), float("nan")
    return float(np.mean((gt >= low) & (gt <= high))), float(np.mean(high - low))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _calibrate_once(low, mid, high, gt, n, ci, delta, strict, progress, n_candidates, bound):
    low_high = np.stack((low, mid, high), axis=1)
    return confromal_risk_control_using_CI_and_WB(prediction_low_high=low_high, Y=gt, ci=ci, delta=delta, n=n,
                                                  strict=strict, progress=progress, n_candidates=n_candidates,
                                                  bound=bound)


def calibrate(low, mid, high, gt, clusters=None, ci=0.90, delta=0.05, min_cluster_size=30, strict=False,
              progress=False, n_candidates=N_BIAS_CANDIDATES, bound="hoeffding"):
    """Return a Calibration for the given calibration-set predictions.

    low, mid, high: raw model outputs (the 90% interval and the median); gt: one target
    per instance on [0, 1]; clusters: one label per instance, or None for a single
    calibration over the whole set. Inside every cluster the bound uses the cluster
    size as n, the number of instances the cluster's risk is estimated from. bound is the
    upper confidence bound: "hoeffding" (default), "bentkus" or "hb" (the minimum of both).
    """
    if bound not in BOUNDS:
        raise ValueError(f"Unknown bound {bound!r}; choose one of {sorted(BOUNDS)}")
    low, mid, high, gt = (np.asarray(a, dtype=float) for a in (low, mid, high, gt))
    n_total = len(gt)
    if clusters is None:
        omega, beta = _calibrate_once(low, mid, high, gt, n_total, ci, delta, strict, progress, n_candidates, bound)
        result = Calibration(omega=float(omega), beta=float(beta), target_coverage=ci, delta=delta, bound=bound,
                             strict=strict,
                             clustered=False, n_calibration=n_total, min_cluster_size=min_cluster_size)
        _add_summary(result, low, mid, high, gt)
        return result

    clusters = np.asarray(clusters)
    results = []
    for c in np.unique(clusters):
        mask = clusters == c
        size = int(mask.sum())
        if size < min_cluster_size:
            continue
        try:
            omega, beta = _calibrate_once(low[mask], mid[mask], high[mask], gt[mask], size, ci, delta, strict, progress,
                                          n_candidates, bound)
        except ValueError as err:  # a degenerate cluster, e.g. no valid bias
            print(f"cluster {c} ({size} instances) could not be calibrated ({err}), skipped")
            continue
        if not (np.isfinite(omega) and np.isfinite(beta)):
            continue
        cal_low, _, cal_high = interval_fn3(low, mid, high, b=beta, l=omega)
        raw_cov, raw_w = coverage_and_width(low[mask], high[mask], gt[mask])
        cov, w = coverage_and_width(cal_low[mask], cal_high[mask], gt[mask])
        results.append(ClusterResult(cluster=int(c), size=size, omega=float(omega), beta=float(beta),
                                     width_on_calibration_set=float(np.mean(cal_high - cal_low)),
                                     raw_coverage=raw_cov, raw_width=raw_w, coverage=cov, width=w))
    if not results:
        raise ValueError(f"No cluster with at least {min_cluster_size} instances could be calibrated")
    best = max(results, key=lambda r: r.width_on_calibration_set)
    sel_low, _, sel_high = interval_fn3(low, mid, high, b=best.beta, l=best.omega)
    for r in results:
        mask = clusters == r.cluster
        r.coverage_selected, r.width_selected = coverage_and_width(sel_low[mask], sel_high[mask], gt[mask])
    result = Calibration(omega=best.omega, beta=best.beta, target_coverage=ci, delta=delta, bound=bound,
                         strict=strict,
                         clustered=True, n_clusters=int(len(np.unique(clusters))), selected_cluster=best.cluster,
                         min_cluster_size=min_cluster_size, n_calibration=n_total, clusters=results)
    _add_summary(result, low, mid, high, gt)
    return result


def _add_summary(result, low, mid, high, gt):
    """Coverage and mean width on the whole calibration set, before and after calibration."""
    result.raw_coverage, result.raw_width = coverage_and_width(low, high, gt)
    cal_low, _, cal_high = result.apply(low, mid, high)
    result.coverage, result.width = coverage_and_width(cal_low, cal_high, gt)


__all__ = ["Calibration", "ClusterResult", "IDENTITY", "coverage_and_width", "N_BIAS_CANDIDATES", "calibrate", "file_sha256"]

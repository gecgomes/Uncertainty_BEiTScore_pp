"""Metrics of one model on one benchmark, from its items and its scores.

correlation  Kendall tau-b, tau-c and Spearman rho (x100) with every human rating, the metric score
             repeated for each rating of a pair (CLIPScore protocol)
pairwise     accuracy (correct caption scored strictly above the incorrect one), micro and macro
             over data_type, and AUROC (x100)
winoground   text, image and group scores (x100)
vela         tau-b, tau-c and rho (x100) against each single human score of vela.TARGETS, on each
             evaluation group separately (dci_val = TestA, dci_test = TestB), with the PCA fitted
             on the fit groups
Pairs whose score is NaN (an LLM judge that gave no number) are left out and counted in n_failed.
"""
import numpy as np
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score

from beitscore import vela


def _corr(score, human):
    ok = np.isfinite(score) & np.isfinite(human)
    s, h = score[ok], human[ok]
    return {"tau_b": 100 * kendalltau(s, h, variant="b").statistic,
            "tau_c": 100 * kendalltau(s, h, variant="c").statistic,
            "rho": 100 * spearmanr(s, h).statistic, "n": int(ok.sum()), "n_failed": int((~ok).sum())}


def evaluate(items, lookup):
    rows, task = items["rows"], items["task"]
    score = np.array([lookup.get((r["image"], r["caption"]), np.nan) for r in rows], dtype=float)  # VELA fit rows have none

    if task == "correlation":
        s = np.concatenate([[v] * len(r["human_scores"]) for v, r in zip(score, rows)])
        h = np.concatenate([r["human_scores"] for r in rows])
        return _corr(s, h)

    if task == "pairwise":
        pairs = {}
        for v, r in zip(score, rows):
            pairs.setdefault(r["pair_id"], {"data_type": r["data_type"]})["bad" if r["incorrect"] else "good"] = v
        good = np.array([p["good"] for p in pairs.values()])
        bad = np.array([p["bad"] for p in pairs.values()])
        types = np.array([p["data_type"] for p in pairs.values()])
        ok = np.isfinite(good) & np.isfinite(bad)
        acc = good[ok] > bad[ok]
        out = {"acc": 100 * acc.mean(),
               "acc_macro": 100 * np.mean([acc[types[ok] == t].mean() for t in np.unique(types[ok])]),
               "auroc": 100 * roc_auc_score([1] * ok.sum() + [0] * ok.sum(), np.r_[good[ok], bad[ok]]),
               "n": int(ok.sum()), "n_failed": int((~ok).sum())}
        for t in np.unique(types[ok]):
            out[f"acc/{t}"] = 100 * acc[types[ok] == t].mean()
        return out

    if task == "winoground":
        by = {}
        for v, r in zip(score, rows):
            by.setdefault(r["pair_id"], {})[r["tag"]] = v
        ex = [b for b in by.values() if all(k in b and np.isfinite(b[k]) for k in ("00", "01", "10", "11"))]
        text = np.array([(b["00"] > b["01"]) and (b["11"] > b["10"]) for b in ex])
        image = np.array([(b["00"] > b["10"]) and (b["11"] > b["01"]) for b in ex])
        return {"text": 100 * text.mean(), "image": 100 * image.mean(), "group": 100 * (text & image).mean(),
                "n": len(ex), "n_failed": len(by) - len(ex)}

    if task == "vela":
        groups = np.array([r["group"] for r in rows])
        x = np.array([[r[d] for d in vela.DIMS] for r in rows], dtype=float)
        fit = vela.fit_pca(x[np.isin(groups, items["fit_groups"])])
        out = {"fit": {"weights": dict(zip(vela.DIMS, fit["weights"].round(4).tolist())),
                       "component": dict(zip(vela.DIMS, fit["component"].round(4).tolist())),
                       "fit_groups": items["fit_groups"], "groups": items["groups"]}}
        for g in items["groups"]:
            ev = groups == g
            out[g] = {name: _corr(score[ev], human) for name, human in vela.targets(x[ev], fit).items()}
        return out
    raise ValueError(task)


def headline(result, task, corr_metric="tau_c", pair_metric="acc", wino_metric="group", vela_targets=None):
    """The numbers shown in the table: {column suffix: value}."""
    if task == "correlation":
        return {"": result[corr_metric]}
    if task == "pairwise":
        return {"": result[pair_metric]}
    if task == "winoground":
        return {"": result[wino_metric]}
    groups = result["fit"]["groups"]
    return {f"{t}|{g}": result[g][t][corr_metric] for t in (vela_targets or vela.TARGETS) for g in groups}

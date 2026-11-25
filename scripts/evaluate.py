"""Correlation with human judgements, for BEiTScore and BEiTScore++.

    # BEiTScore (released checkpoint)
    python scripts/evaluate.py --model_size large --benchmark vicr
    # BEiTScore++ with its calibration (written by scripts/beitscore_pp/calibrate.py)
    python scripts/evaluate.py --model_size large --quant --benchmark vicr --calibration auto

Prints tau-b, tau-c and rho for each subset (data_type) of the benchmark and, for
BEiTScore++, coverage, sharpness, UPS (and its rank variants) and the quantile ECE.
Writes the raw scores to $BEITSCORE_OUTPUT/scores/<model id>/<benchmark>_test.pkl and
the metrics to $BEITSCORE_OUTPUT/results/<model id>/<benchmark>.csv.

Error detection (nocaps-FOIL, VALSE, SugarCrepe, Winoground, LongCapVLCP) is in
scripts/beitscore_v1/evaluate_errors.py, and CapArena in scripts/beitscore_v1/caparena.py.
"""
import argparse
import os

import pandas as pd

from beitscore import metrics
from beitscore.data.datasets import CORRELATION_BENCHMARKS, BEiTDataset, canonical_benchmark, get_sentencepiece_model_for_beit3
from beitscore.inference import (SENTENCEPIECE_MODEL, add_model_args, build_model, output_dir, predict,
                                 print_timing, resolve_model, save_scores, score_frame, select_interval)
from beitscore.paths import DATA_ROOT
from beitscore.pp.calibration.clustered import IDENTITY, Calibration
from beitscore.utils import set_seed




def get_args(argv=None):
    parser = argparse.ArgumentParser("Correlation with human judgements")
    add_model_args(parser)
    parser.add_argument("--benchmark", type=canonical_benchmark, choices=CORRELATION_BENCHMARKS, default="vicr",
                        help="vicr, polaris, nebula, flickr8k-expert, flickr8k-cf, composite, genai")
    parser.add_argument("--base_path", default=DATA_ROOT, help="Root directory of the datasets")
    parser.add_argument("--calibration", default=None,
                        help="BEiTScore++ only: a calibration.json, or 'auto' for the one written by "
                             "scripts/beitscore_pp/calibrate.py for this model. Without it the raw intervals are used.")
    parser.add_argument("--normalization", default="nominal", choices=["nominal", "observed"],
                        help="Map human scores to [0, 1] with the benchmark's rating scale (nominal) or with the "
                             "split's own min and max (observed, as for the published BEiTScore++ tables)")
    return parser.parse_args(argv)


def load_calibration(args, mid, checkpoint=None):
    """--calibration auto: the calibration.json written by scripts/beitscore_pp/calibrate.py for
    this model, or else the one released next to the downloaded checkpoint."""
    if not args.calibration:
        return IDENTITY
    if args.calibration == "auto":
        candidates = [os.path.join(output_dir("calibration", mid), "calibration.json")]
        if checkpoint:
            candidates.append(os.path.join(os.path.dirname(str(checkpoint)), "calibration.json"))
        path = next((c for c in candidates if os.path.exists(c)), candidates[0])
    else:
        path = args.calibration
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; download the released model (scripts/download_release.py models) "
                                f"or run scripts/beitscore_pp/calibrate.py first")
    calibration = Calibration.load(path)
    print(f"Calibration from {path}: omega={calibration.omega:.4f}, beta={calibration.beta:.4f}")
    return calibration


def evaluate_subset(df, quant, calibration, confidence_levels):
    """Metrics for one subset of the score frame (C-10: everything on the subset)."""
    if not quant:
        return metrics.correlations(df["score"].to_numpy(), df["human_scores"].tolist())
    qcols = sorted(c for c in df.columns if c.startswith("q") and c[1:].isdigit())
    raw = df[qcols].to_numpy(dtype=float)
    low, mid, high = calibration.apply(*select_interval(raw))
    gt = metrics.mean_human_score(df["human_scores"].tolist())
    out = metrics.correlations(mid, df["human_scores"].tolist())
    out.update(metrics.interval_metrics(gt, low, mid, high))
    if raw.shape[1] == 2 * len(confidence_levels) + 1:
        out["ece"] = metrics.quantile_ece(raw, gt, metrics.quantile_levels(confidence_levels))
    out["bimodality_low"] = metrics.bimodality_coefficient(low)
    out["bimodality_high"] = metrics.bimodality_coefficient(high)
    return out


def main(args):
    set_seed(args.seed)
    checkpoint, mid, quant = resolve_model(args)
    model = build_model(args.model_size, quant, checkpoint, args.chained_construction, device=args.device)
    calibration = load_calibration(args, mid, checkpoint) if quant else IDENTITY
    tokenizer = get_sentencepiece_model_for_beit3(SENTENCEPIECE_MODEL)

    dataset = BEiTDataset(benchmark=args.benchmark, base_path=args.base_path, tokenizer=tokenizer,
                          min_max_scale=args.normalization, seed=args.seed)
    scores, times = predict(model, dataset, args.device, desc=args.benchmark)
    df = score_frame(dataset, scores, gt=dataset.gt_scores)
    save_scores(df, mid, f"{args.benchmark}_test")

    levels = list(getattr(model, "quantiles", []))
    rows = []
    for d_type, sub in df.groupby("data_type", sort=False):
        res = evaluate_subset(sub, quant, calibration, levels)
        rows.append({"benchmark": args.benchmark, "data_type": d_type, "n": len(sub), **res})
        print(f"=== {args.benchmark} / {d_type} ({len(sub)} instances) ===")
        for k, v in res.items():
            print(f"{k}: {v:.3f}")
    print_timing(times)

    path = os.path.join(output_dir("results", mid), f"{args.benchmark}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Metrics written to {path}")
    return rows


if __name__ == "__main__":
    main(get_args())

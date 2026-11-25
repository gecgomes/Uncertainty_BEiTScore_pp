"""Post-hoc conformal calibration of BEiTScore++ (Section 3.3 of the paper).

    python scripts/beitscore_pp/calibrate.py --model_size large              # released model
    python scripts/beitscore_pp/calibrate.py --model_size large --use_run 96 # a local run

Scores the validation split (subset == "val") of $BEITSCORE_DATA/stage2/stage2.pkl, the
Stage-2 data of VICR, Polaris, Nebula and EvalMuse-40K (scripts/download_release.py stage2),
then chooses the scale omega and bias beta of the 90% interval with an upper confidence
bound (delta = 0.05) on the miscoverage: Hoeffding by default, or Bentkus or the minimum
of both (HB) with --bound. Inside each cluster, n in the bound is the cluster size. By default the calibration runs per
image cluster (the `cluster` column of the mixture, K = 63) and keeps the most
conservative pair (beitscore.pp.calibration.clustered).

Writes $BEITSCORE_OUTPUT/calibration/<model id>/calibration.json (or the path given by
--out), which scripts/evaluate.py --calibration auto and the scoring API read, and caches
the validation predictions next to it (calibration_scores.pkl) with image_path,
candidates and cluster, so that the calibration can be redone without a GPU.
"""
import argparse
import os

import numpy as np
import pandas as pd

from beitscore import metrics
from beitscore.data.datasets import BEiTDataset, get_sentencepiece_model_for_beit3
from beitscore.inference import (SENTENCEPIECE_MODEL, add_model_args, build_model, output_dir, predict,
                                 resolve_model, score_frame, select_interval)
from beitscore.paths import DATA_ROOT
from beitscore.pp.calibration.clustered import calibrate, file_sha256
from beitscore.utils import set_seed

MIXTURE = "stage2/stage2.pkl"  # read by get_data("regression"); scripts/download_release.py stage2


def get_args(argv=None):
    parser = argparse.ArgumentParser("Calibrate the BEiTScore++ intervals")
    add_model_args(parser, quant_only=True)
    parser.add_argument("--base_path", default=DATA_ROOT)
    parser.add_argument("--target", default=0.90, type=float, help="Target coverage")
    parser.add_argument("--delta", default=0.05, type=float, help="Confidence level of the bound")
    parser.add_argument("--bound", default="hoeffding", choices=["hoeffding", "bentkus", "hb"],
                        help="Upper confidence bound on the miscoverage: hoeffding (default, as in the paper), "
                             "bentkus, or hb, the minimum of the two (Bates et al., 2021)")
    parser.add_argument("--no_clustering", action="store_true", help="One calibration over the whole set")
    parser.add_argument("--min_cluster_size", default=30, type=int)
    parser.add_argument("--strict", action="store_true",
                        help="Divide delta by the number of bias candidates so the guarantee holds after the search")
    parser.add_argument("--recompute", action="store_true", help="Ignore the cached validation predictions")
    parser.add_argument("--out", default=None, help="Output calibration.json (default: under $BEITSCORE_OUTPUT)")
    return parser.parse_args(argv)


def calibration_predictions(args, checkpoint, mid, cache):
    if os.path.exists(cache) and not args.recompute:
        print(f"Reusing the cached predictions in {cache}")
        return pd.read_pickle(cache)
    model = build_model(args.model_size, True, checkpoint, args.chained_construction, device=args.device)
    tokenizer = get_sentencepiece_model_for_beit3(SENTENCEPIECE_MODEL)
    # Human scores of the mixture are already on [0, 1]; they are used as stored.
    dataset = BEiTDataset(benchmark="regression", base_path=args.base_path, tokenizer=tokenizer, split="val",
                          _task="correlation", seed=args.seed)
    scores, _ = predict(model, dataset, args.device, desc="Calibration set")
    df = score_frame(dataset, scores, gt=dataset.items["human_scores"].tolist())
    df.to_pickle(cache)
    return df


def report(df, qcols, low, mid_q, high, cal_low, cal_high, gt, result, target):
    """Print what the calibration did: raw and calibrated coverage and width, overall,
    per dataset and per cluster, plus checks on the raw model outputs."""
    from scipy.stats import kendalltau

    print("\nRaw model outputs on the calibration set")
    print("  mean of each output: " + " ".join(f"{df[c].mean():.3f}" for c in qcols))
    print(f"  median vs target: Kendall tau-b {kendalltau(mid_q, gt).statistic:.3f}")
    print(f"  raw 90% interval: coverage {100 * result.raw_coverage:.2f}%, mean width {result.raw_width:.4f}, "
          f"width quartiles {np.percentile(high - low, [25, 50, 75]).round(3).tolist()}")
    print(f"  lower end <= 0.01: {100 * np.mean(low <= 0.01):.1f}%, upper end >= 0.99: {100 * np.mean(high >= 0.99):.1f}%")
    if np.any(np.diff(df[qcols].to_numpy(), axis=1) < -1e-9):
        print("  WARNING: the outputs are not ordered (quantile crossing)")
    if result.raw_width > 0.5:
        print("  WARNING: the raw intervals cover more than half of [0, 1]. A trained BEiTScore++ gives much "
              "narrower intervals; check that the checkpoint is a BEiTScore++ checkpoint, that its head was "
              "loaded (no 'not initialized' or 'Removing key' lines above) and that --chained_construction "
              "matches the run.")

    print(f"\nCalibrated (omega = {result.omega:.4f}, beta = {result.beta:.4f}): coverage "
          f"{100 * result.coverage:.2f}% (target {100 * target:.0f}%), mean width {result.width:.4f}, {len(gt)} instances")
    if "dataset" in df.columns:
        rows = []
        for name, idx in df.groupby("dataset").indices.items():
            rows.append({"dataset": name, "n": len(idx),
                         "raw_cov": np.mean((gt[idx] >= low[idx]) & (gt[idx] <= high[idx])),
                         "raw_width": np.mean(high[idx] - low[idx]),
                         "cov": np.mean((gt[idx] >= cal_low[idx]) & (gt[idx] <= cal_high[idx])),
                         "width": np.mean(cal_high[idx] - cal_low[idx])})
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if result.clustered:
        table = pd.DataFrame([vars(c) for c in result.clusters])
        cols = ["cluster", "size", "omega", "beta", "raw_coverage", "raw_width", "coverage", "width",
                "coverage_selected", "width_selected"]
        print(f"\nPer cluster ({len(table)} calibrated): coverage on the cluster's own instances")
        print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        below = table[table["coverage_selected"] < target]
        print(f"Clusters below the target with the selected pair: {len(below)} of {len(table)}"
              + (f" (lowest {below['coverage_selected'].min():.3f})" if len(below) else ""))
        print(f"Clusters already at the target without calibration: {(table['raw_coverage'] >= target).sum()}")


def main(args):
    set_seed(args.seed)
    checkpoint, mid, _ = resolve_model(args, quant=True)
    folder = output_dir("calibration", mid)
    df = calibration_predictions(args, checkpoint, mid, os.path.join(folder, "calibration_scores.pkl"))

    qcols = sorted(c for c in df.columns if c.startswith("q") and c[1:].isdigit())
    low, mid_q, high = select_interval(df[qcols].to_numpy())
    gt = metrics.mean_human_score(df["human_scores"].tolist())
    clusters = None
    if not args.no_clustering:
        if "cluster" not in df.columns:
            raise ValueError("The calibration set has no `cluster` column; use the training mixture "
                             "with image clusters (stage2/stage2.pkl, from scripts/download_release.py stage2) or pass --no_clustering")
        clusters = df["cluster"].to_numpy()

    result = calibrate(low, mid_q, high, gt, clusters=clusters, ci=args.target, delta=args.delta,
                       min_cluster_size=args.min_cluster_size, strict=args.strict, progress=True,
                       bound=args.bound)
    result.model = mid
    mixture = os.path.join(args.base_path, MIXTURE)
    if clusters is not None and os.path.exists(mixture):
        result.cluster_file, result.cluster_file_sha256 = MIXTURE, file_sha256(mixture)

    cal_low, _, cal_high = result.apply(low, mid_q, high)
    print(f"omega = {result.omega:.4f}, beta = {result.beta:.4f}, {result.bound} bound, delta = {result.delta}"
          + (f" (cluster {result.selected_cluster} of {result.n_clusters})" if result.clustered else ""))
    report(df, qcols, low, mid_q, high, cal_low, cal_high, gt, result, args.target)

    name = "calibration.json" if clusters is not None else "calibration_unclustered.json"
    path = args.out or os.path.join(folder, name)
    result.save(path)
    print(f"Calibration written to {path}")
    return result


if __name__ == "__main__":
    main(get_args())

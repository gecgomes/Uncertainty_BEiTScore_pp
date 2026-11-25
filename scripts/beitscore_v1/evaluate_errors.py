"""Detection of incorrect captions (BEiTScore paper): nocaps-FOIL, VALSE,
SugarCrepe, LongCapVLCP and Winoground.

    python scripts/beitscore_v1/evaluate_errors.py --model_size large --benchmark valse

For each image, the correct caption should score above its incorrect version. Prints
pairwise accuracy, AUROC and average precision, overall (micro average) and per subset;
for Winoground, the text, image and group scores. With --quant, the BEiTScore++ median
is used as the score. Scores go to $BEITSCORE_OUTPUT/scores/<model id>/<benchmark>_test.pkl
and metrics to $BEITSCORE_OUTPUT/results/<model id>/<benchmark>.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

from beitscore import metrics
from beitscore.data.datasets import ERROR_BENCHMARKS, BEiTDataset, canonical_benchmark, get_sentencepiece_model_for_beit3
from beitscore.inference import (SENTENCEPIECE_MODEL, add_model_args, build_model, output_dir, predict,
                                 print_timing, resolve_model, save_scores, score_frame, select_interval)
from beitscore.paths import DATA_ROOT
from beitscore.utils import set_seed




def get_args(argv=None):
    parser = argparse.ArgumentParser("Detection of incorrect captions")
    add_model_args(parser)
    parser.add_argument("--benchmark", type=canonical_benchmark, choices=ERROR_BENCHMARKS, default="valse",
                        help="nocaps-foil, valse, sugarcrepe, winoground, longcapvlcp")
    parser.add_argument("--base_path", default=DATA_ROOT)
    return parser.parse_args(argv)


def pair_scores(df):
    """Split a score frame into aligned (correct, incorrect) score arrays per pair_id."""
    correct = df[df["foil"] == False].sort_values("pair_id")  # noqa: E712
    incorrect = df[df["foil"] == True].sort_values("pair_id")  # noqa: E712
    if correct["pair_id"].tolist() != incorrect["pair_id"].tolist():
        raise ValueError("Correct and incorrect captions are not paired one to one")
    return correct, incorrect


def error_results(df):
    correct, incorrect = pair_scores(df)
    rows = [{"data_type": "all (micro)", "n_pairs": len(correct),
             **metrics.error_detection(correct["score"], incorrect["score"])}]
    for d_type in correct["data_type"].unique():
        c = correct[correct["data_type"] == d_type]
        i = incorrect.set_index("pair_id").loc[c["pair_id"]]
        rows.append({"data_type": d_type, "n_pairs": len(c), **metrics.error_detection(c["score"], i["score"])})
    if len(rows) > 2:
        macro = np.mean([r["accuracy"] for r in rows[1:]])
        rows.append({"data_type": "all (macro accuracy)", "n_pairs": len(correct), "accuracy": macro})
    return rows


def winoground_results(df):
    by_tag = {tag: g.sort_values("pair_id")["score"].to_numpy() for tag, g in df.groupby("tag_combination")}
    return [{"data_type": "winoground", **metrics.winoground(by_tag["00"], by_tag["01"], by_tag["10"], by_tag["11"])}]


def main(args):
    set_seed(args.seed)
    checkpoint, mid, quant = resolve_model(args)
    model = build_model(args.model_size, quant, checkpoint, args.chained_construction, device=args.device)
    tokenizer = get_sentencepiece_model_for_beit3(SENTENCEPIECE_MODEL)
    dataset = BEiTDataset(benchmark=args.benchmark, base_path=args.base_path, tokenizer=tokenizer, seed=args.seed)
    scores, times = predict(model, dataset, args.device, desc=args.benchmark)
    df = score_frame(dataset, scores)
    if quant:
        df["score"] = select_interval(scores)[1]  # the median
    save_scores(df, mid, f"{args.benchmark}_test")

    rows = winoground_results(df) if dataset.wino_tag else error_results(df)
    for r in rows:
        print(", ".join(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in r.items()))
    print_timing(times)
    path = os.path.join(output_dir("results", mid), f"{args.benchmark}.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Metrics written to {path}")
    return rows


if __name__ == "__main__":
    main(get_args())

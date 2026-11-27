#!/usr/bin/env python
"""Compute the metrics from the saved scores of all the runners and write a LaTeX table.

Nothing is scored here: a (model, benchmark) without saved scores is reported and left as "--".
Any environment with numpy, scipy and scikit-learn can run it, from the repository root.

  python baselines/make_table.py --benchmarks vela --models all
  python baselines/make_table.py --benchmarks vela vicr polaris valse --models beitscore-b beitscore-l specs fleur expert
  python baselines/make_table.py --benchmarks vela --vela_targets desc rel flu --corr_metric tau_b

Writes <results>/metrics/<model>/<benchmark>.json (every metric) and the table (--out).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import benchmarks as B  # noqa: E402
from common import store  # noqa: E402
from common.evaluate import evaluate, headline  # noqa: E402
from common.names import MODELS, REPORTED_VELA  # noqa: E402
from common.runner import RESULTS  # noqa: E402
from common.table import latex  # noqa: E402
from beitscore.vela import DEFAULT_TARGETS, TARGETS  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["all"], help=f"'all' or any of: {', '.join(MODELS)}")
    p.add_argument("--benchmarks", nargs="+", default=["vela"], help=f"'all' or any of: {', '.join(B.ALL)}")
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--out", default=None, help="Default: <results>/table.tex")
    p.add_argument("--corr_metric", default="tau_c", choices=["tau_b", "tau_c", "rho"])
    p.add_argument("--pair_metric", default="acc", choices=["acc", "acc_macro", "auroc"])
    p.add_argument("--wino_metric", default="group", choices=["text", "image", "group"])
    p.add_argument("--vela_targets", nargs="+", default=list(DEFAULT_TARGETS),
                   choices=list(TARGETS))
    p.add_argument("--vela_paper", nargs="*", default=[], choices=list(REPORTED_VELA),
                   help="Add VELA's values as reported in its paper (not run here): without_refs, with_refs; "
                        "only in the desc, rel and flu columns, never ranked")
    args = p.parse_args()
    scored = [m for m in MODELS if MODELS[m][1] != "reported"]
    models = scored if args.models == ["all"] else args.models
    benches = B.ALL if args.benchmarks == ["all"] else args.benchmarks
    for m in models:
        if m not in MODELS:
            p.error(f"unknown model {m}; choose from {', '.join(MODELS)}")

    cells, columns, metric_names, missing = {}, [], {}, []
    for bench in benches:
        items = store.load_items(args.results, bench)
        if items is None:
            print(f"[{bench}] not prepared; skipped")
            continue
        task = items["task"]
        metric_names[bench] = {
            "correlation": {"tau_b": "$\\tau_b$", "tau_c": "$\\tau_c$", "rho": "$\\rho$"}[args.corr_metric],
            "pairwise": {"acc": "Acc.", "acc_macro": "Macro Acc.", "auroc": "AUROC"}[args.pair_metric],
            "winoground": args.wino_metric.capitalize(), "vela": ""}[task]
        columns += ([(bench, f"{t}|{g}") for t in args.vela_targets for g in items["groups"]] if task == "vela"
                    else [(bench, "")])
        pairs = store.scored_pairs(items)
        all_pairs = store.unique_pairs(items["rows"])
        for model in models:
            if MODELS[model][1] == "reported":  # values quoted from a paper, filled below
                continue
            lookup = store.load_scores(args.results, model, bench, pairs, all_pairs)
            if lookup is None:
                missing.append(f"{model} / {bench} (run {MODELS[model][2]}/run_{MODELS[model][2]}.py)")
                continue
            result = evaluate(items, lookup)
            path = os.path.join(args.results, "metrics", model, f"{bench}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(result, f, indent=1)
            shown = headline(result, task, args.corr_metric, args.pair_metric, args.wino_metric, args.vela_targets)
            failed = result.get("n_failed") if task != "vela" else result[items["groups"][0]][args.vela_targets[0]]["n_failed"]
            print(f"{model:14s} {bench:16s} " + "  ".join(f"{k or 'score'}={v:.1f}" for k, v in shown.items())
                  + (f"  [{failed} without a score]" if failed else ""))
            for sub, v in shown.items():
                cells[(model, (bench, sub))] = v
        if task == "vela":
            for variant in args.vela_paper:
                for target in args.vela_targets:
                    for group in items["groups"]:
                        value = REPORTED_VELA[variant].get(group, {}).get(target)
                        if value is not None:
                            cells[(f"vela-paper-{variant}", (bench, f"{target}|{group}"))] = value
    models = [f"vela-paper-{v}" for v in args.vela_paper] + [m for m in models if not m.startswith("vela-paper-")]

    note = ""
    if any(b == "vela" for b, _ in columns):
        dims = [t for t in args.vela_targets if t in ("desc", "rel", "flu")]
        combined = [t for t in args.vela_targets if t not in dims]
        named = {"desc": "descriptiveness", "rel": "relevance", "flu": "fluency"}
        parts = []
        if dims:
            parts.append("the " + " and ".join(named[t] for t in dims) + " ratings")
        if combined:
            parts.append("single human scores built from the descriptiveness, relevance and fluency ratings, "
                         "with the principal component fitted on its training and validation sets")
        note = "LongCap-Arena columns report the correlation with " + ", and with ".join(parts) + \
               ", on TestA and TestB of LongCap-Arena. "
        if args.vela_paper:
            note += ("The VELA rows are the values reported by \\citet{matsuda2025vela} on the same test sets "
                     "(not computed here); VELA was fine-tuned on the training split of LongCap-Arena for each "
                     "dimension, and its main model also reads the reference captions. They are not ranked. ")
    out = args.out or os.path.join(args.results, "table.tex")
    with open(out, "w") as f:
        f.write(latex(cells, models, columns, metric_names, note))
    print(f"Table written to {out}")
    if missing:
        print("No saved scores (left as --):\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()

"""The scoring loop shared by the runners (baselines/encoders, fleur, expert and vqascore).

A runner scores its models on the selected benchmarks and saves one score per image-caption pair
in <results>/scores/<model>/<benchmark>.json, with <results> = $BEITSCORE_OUTPUT/baselines by
default. Saved scores are reused when they exist for the same benchmark items; --recompute scores
again and overwrites them. The benchmark items are prepared by baselines/encoders/run_encoders.py,
the only runner whose environment has the beitscore package.
"""
import argparse
import os
import time
import traceback

import numpy as np

from . import benchmarks as B
from . import store

# The scorers fork data-loader workers after a Hugging Face tokenizer has run (CLIPScore, then PAC-S):
# turning the tokenizer's own threads off avoids its fork warning and any deadlock.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import REPO  # noqa: E402

DATA = os.environ.get("BEITSCORE_DATA", os.path.join(REPO, "data"))
RESULTS = os.path.join(os.environ.get("BEITSCORE_OUTPUT", os.path.join(REPO, "outputs")), "baselines")


def base_parser(description, models):
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(models), help=f"'all' or any of: {', '.join(models)} (default: all)")
    p.add_argument("--benchmarks", nargs="+", default=["vela"], help=f"'all' or any of: {', '.join(B.ALL)}")
    p.add_argument("--data", default=DATA,
                   help="Benchmark data (default: $BEITSCORE_DATA, see configs/paths.sh)")
    p.add_argument("--results", default=RESULTS,
                   help="Folder shared by all the runners and make_table.py (default: $BEITSCORE_OUTPUT/baselines)")
    p.add_argument("--recompute", action="store_true", help="Score again and overwrite the saved scores")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p


def check_args(p, args, models):
    if not args.data:
        p.error("BEITSCORE_DATA is not set: source configs/paths.sh or pass --data")
    args.data = os.path.abspath(args.data)
    args.benchmarks = B.ALL if args.benchmarks == ["all"] else args.benchmarks
    args.models = list(models) if args.models == ["all"] else args.models
    for m in args.models:
        if m not in models:
            p.error(f"unknown model {m}; this runner scores: {', '.join(models)}")
    for b in args.benchmarks:
        if b not in B.ALL:
            p.error(f"unknown benchmark {b}; choose from {', '.join(B.ALL)}")


def load_items(args, prepare=None):
    """{benchmark: items}. Missing items are prepared with `prepare` (encoders runner only)."""
    out = {}
    for bench in args.benchmarks:
        items = store.load_items(args.results, bench)
        if items is not None:
            print(f"[{bench}] prepared items: {store.items_path(args.results, bench)}")
        if items is None and prepare is not None:
            try:
                prepare(bench)
                items = store.load_items(args.results, bench)
            except Exception as e:
                print(f"[{bench}] could not be prepared, skipped: {type(e).__name__}: {e}")
                continue
        if items is None:
            print(f"[{bench}] not prepared yet, skipped: run baselines/encoders/run_encoders.py --benchmarks {bench} first")
            continue
        out[bench] = items
    return out


def data_root(args):
    if not args.data:
        raise SystemExit("BEITSCORE_DATA is not set: source configs/paths.sh or pass --data")
    return args.data


def image_file(args, path):
    """Relative image paths in the items are relative to the data folder ($BEITSCORE_DATA)."""
    return path if os.path.isabs(path) else os.path.join(data_root(args), path)


def _check_images(args, pairs):
    files = {image_file(args, p[0]) for p in pairs}
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(files)} images not found, e.g. {missing[0]}")


def _score_with_progress(args, scorer, pairs, batch_size, desc):
    """Scores the pairs in chunks of a few batches, so a progress bar can follow them. Each pair is
    scored independently, so the chunking does not change the scores."""
    from tqdm import tqdm

    chunk = getattr(scorer, "chunk_size", None) or max(1, batch_size) * 8
    out = []
    with tqdm(total=len(pairs), desc=desc, unit="pair", dynamic_ncols=True) as bar:
        for i in range(0, len(pairs), chunk):
            part = pairs[i:i + chunk]
            out.extend(scorer.score([image_file(args, p[0]) for p in part], [p[1] for p in part]))
            bar.update(len(part))
    return np.asarray(out, dtype=float)


def score_all(args, factories, items_by_bench):
    """Score every (model, benchmark) without saved scores; returns the models that failed."""
    failed = []
    for model in args.models:
        todo = []
        for bench, items in items_by_bench.items():
            pairs = store.scored_pairs(items)
            if args.recompute or store.load_scores(args.results, model, bench, pairs,
                                                    store.unique_pairs(items["rows"])) is None:
                todo.append((bench, pairs))
            else:
                print(f"[{model} / {bench}] saved scores found, skipped")
        ready = []
        for bench, pairs in todo:
            try:
                _check_images(args, pairs)
                ready.append((bench, pairs))
            except FileNotFoundError as e:
                print(f"[{model} / {bench}] skipped: {e}")
                failed.append(model)
        todo = ready
        if not todo:
            continue
        try:
            print(f"[{model}] loading")
            scorer = factories[model](args)
            for bench, pairs in todo:
                print(f"[{model} / {bench}] scoring {len(pairs)} image-caption pairs")
                start = time.time()
                scores = _score_with_progress(args, scorer, pairs, args.batch_size, f"{model} / {bench}")
                info = {**getattr(scorer, "info", {}), "seconds": round(time.time() - start, 1),
                        "n_nan": int(np.isnan(scores).sum())}
                if info["n_nan"] == len(scores):
                    print(f"[{model} / {bench}] every score failed; nothing saved")
                    failed.append(model)
                    continue
                store.save_scores(args.results, model, bench, pairs, scores, info)
                print(f"[{model} / {bench}] saved ({info['seconds']} s, {info['n_nan']} without a score)")
            del scorer
        except Exception:
            traceback.print_exc()
            failed.append(model)
            print(f"[{model}] failed")
    return failed

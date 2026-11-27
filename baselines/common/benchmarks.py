"""The benchmark items: one row per image-caption pair to score, prepared from the benchmark files
of the repository ($BEITSCORE_DATA/<name>/<name>.pkl, see docs/benchmarks.md) by
baselines/encoders/run_encoders.py, the runner whose environment has the beitscore package. Once saved, any
environment can score and evaluate them. Image paths are stored relative to $BEITSCORE_DATA.
"""
import os

from .store import save_items

CORRELATION = ["vicr", "polaris", "nebula", "flickr8k-expert", "flickr8k-cf", "composite", "genai"]
PAIRWISE = ["nocaps-foil", "valse", "sugarcrepe", "longcapvlcp"]
WINOGROUND = ["winoground"]
ALL = ["vela"] + CORRELATION + PAIRWISE + WINOGROUND


def _relative(path, data_root):
    path = os.path.abspath(str(path))
    root = os.path.abspath(data_root)
    return os.path.relpath(path, root) if path.startswith(root + os.sep) else path


def prepare(benchmark, out, data_root):
    from beitscore.data.datasets import get_data

    if benchmark == "vela":
        return _prepare_vela(out, data_root)
    df, task, wino = get_data(benchmark, data_root, "test")
    img = [_relative(p, data_root) for p in df["image_path"]]
    if benchmark in CORRELATION:
        rows = [{"image": i, "caption": str(r.caption),
                 "human_scores": [float(h) for h in (r.human_scores if isinstance(r.human_scores, (list, tuple))
                                                     else list(r.human_scores))]}
                for i, r in zip(img, df.itertuples())]
        save_items(out, benchmark, "correlation", rows)
    elif wino:
        rows = [{"image": i, "caption": str(r.caption), "pair_id": str(r.pair_id), "tag": str(r.tag_combination)}
                for i, r in zip(img, df.itertuples())]
        save_items(out, benchmark, "winoground", rows)
    elif task == "foil":
        rows = []
        for k, (i, r) in enumerate(zip(img, df.itertuples())):
            dt = str(getattr(r, "data_type", benchmark))
            rows.append({"image": i, "caption": str(r.caption), "pair_id": k, "incorrect": False, "data_type": dt})
            rows.append({"image": i, "caption": str(r.foil), "pair_id": k, "incorrect": True, "data_type": dt})
        save_items(out, benchmark, "pairwise", rows)
    else:
        raise ValueError(f"{benchmark}: unexpected task {task}")
    print(f"  prepared {benchmark}: {len(rows)} rows")


def _prepare_vela(out, data_root):
    """Every pair of vela.pkl with its three ratings: the evaluation groups are scored, the train
    and val groups only provide the ratings the principal component is fitted on."""
    import joblib

    from beitscore.data.datasets import benchmark_file
    from beitscore.vela import DIMS, EVAL_GROUPS, FIT_GROUPS

    path = benchmark_file("vela", data_root)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; prepare it with: python scripts/prepare_data/benchmarks.py download vela")
    df = joblib.load(path)
    df = df[df["subset"].isin(list(EVAL_GROUPS) + list(FIT_GROUPS))]
    rows = [{"image": str(r.image_path), "caption": str(r.caption), "group": str(r.subset),
             **{d: float(getattr(r, d)) for d in DIMS}} for r in df.itertuples()]
    save_items(out, "vela", "vela", rows, {"groups": list(EVAL_GROUPS), "fit_groups": list(FIT_GROUPS)})
    sizes = df.groupby("subset").size().to_dict()
    print(f"  prepared vela: {len(rows)} rows {sizes}; scored: {', '.join(EVAL_GROUPS)}")

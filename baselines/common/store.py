"""Benchmark items and model scores, stored as JSON so that every environment can read them.

<out>/items/<benchmark>.json   the (image, caption) rows of a benchmark and what is needed to evaluate it
<out>/scores/<model>/<benchmark>.json   one score per distinct (image, caption) of the benchmark
"""
import hashlib
import json
import os
import time


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def items_path(out, benchmark):
    return os.path.join(out, "items", f"{benchmark}.json")


def scores_path(out, model, benchmark):
    return os.path.join(out, "scores", model, f"{benchmark}.json")


def save_items(out, benchmark, task, rows, extra=None):
    _write(items_path(out, benchmark), {"benchmark": benchmark, "task": task, "rows": rows, **(extra or {})})


def load_items(out, benchmark):
    path = items_path(out, benchmark)
    return _read(path) if os.path.exists(path) else None


def unique_pairs(rows):
    """The distinct (image, caption) pairs of a benchmark, in first-seen order."""
    seen, pairs = set(), []
    for r in rows:
        key = (r["image"], r["caption"])
        if key not in seen:
            seen.add(key)
            pairs.append(key)
    return pairs


def scored_pairs(items):
    """The pairs that need a metric score. For VELA, only the evaluation groups: the fit groups only
    provide human ratings for the PCA."""
    rows = items["rows"]
    if items.get("task") == "vela":
        rows = [r for r in rows if r["group"] in items["groups"]]
    return unique_pairs(rows)


def fingerprint(pairs):
    h = hashlib.sha256()
    for image, caption in pairs:
        h.update(os.path.basename(image).encode())
        h.update(b"\t")
        h.update(caption.encode())
        h.update(b"\n")
    return h.hexdigest()


def save_scores(out, model, benchmark, pairs, scores, info=None):
    _write(scores_path(out, model, benchmark), {
        "model": model, "benchmark": benchmark, "fingerprint": fingerprint(pairs),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "info": info or {},
        "scores": [None if s is None or s != s else float(s) for s in scores]})


def load_scores(out, model, benchmark, pairs, all_pairs=None):
    """{(image, caption): score} when saved scores exist for exactly these pairs, else None. Scores
    saved for a larger set of pairs (all_pairs, e.g. every VELA group) are used for their subset."""
    path = scores_path(out, model, benchmark)
    if not os.path.exists(path):
        return None
    saved = _read(path)
    if (all_pairs is not None and all_pairs != pairs and saved.get("fingerprint") == fingerprint(all_pairs)
            and len(saved["scores"]) == len(all_pairs)):
        wanted = set(pairs)
        saved["scores"] = [s for p, s in zip(all_pairs, saved["scores"]) if p in wanted]
        saved["fingerprint"] = fingerprint(pairs)
    if saved.get("fingerprint") != fingerprint(pairs) or len(saved["scores"]) != len(pairs):
        print(f"  [{model} / {benchmark}] saved scores do not match the current benchmark items; ignoring them")
        return None
    if all(s is None for s in saved["scores"]):
        print(f"  [{model} / {benchmark}] saved scores are all missing; scoring again")
        return None
    return {p: (float("nan") if s is None else s) for p, s in zip(pairs, saved["scores"])}

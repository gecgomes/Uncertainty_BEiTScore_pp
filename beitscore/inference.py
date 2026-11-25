"""Model loading, batched prediction and score files, shared by the evaluation scripts.

Checkpoint layout under $BEITSCORE_MODELS (the same for every script):

    beit3.spm
    <vqa|nlvr2>/<base|large>_<with_stage1|without_stage1|without_stage2>.pth   (BEiTScore)
    beitscorepp/<base|large>/model.safetensors, config.json, calibration.json  (BEiTScore++)

Checkpoints trained locally are read from $BEITSCORE_OUTPUT/finetuning/<task>/run-<n>/.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timm.models import create_model
from tqdm import tqdm

import beitscore.models  # noqa: F401  (registers the model configs)

from .metrics import mean_human_score
from .models.checkpoint import load_model_and_may_interpolate_beitscore
from .paths import FINETUNE_ROOT, MODELS_ROOT, OUTPUT_ROOT

SENTENCEPIECE_MODEL = os.path.join(MODELS_ROOT, "beit3.spm")
BEITSCORE_VARIANTS = ("with_stage1", "without_stage1", "without_stage2")


# ---------------------------------------------------------------------------- paths
def beitscore_checkpoint(model_size, pretrained="vqa", variant="with_stage1"):
    """BEiTScore checkpoint. The downloaded release lives in
    $BEITSCORE_MODELS/beitscore/<size>[_<variant>][_nlvr2]/model.safetensors (see
    scripts/download_release.py); otherwise $BEITSCORE_MODELS/<pretrained>/<size>_<variant>.pth
    (checkpoints from your own runs) is used."""
    if variant not in BEITSCORE_VARIANTS:
        raise ValueError(f"variant must be one of {BEITSCORE_VARIANTS}")
    folder = model_size + ("" if variant == "with_stage1" else f"_{variant}") + ("" if pretrained == "vqa" else f"_{pretrained}")
    released = os.path.join(MODELS_ROOT, "beitscore", folder, "model.safetensors")
    if os.path.exists(released):
        return released
    return os.path.join(MODELS_ROOT, pretrained, f"{model_size}_{variant}.pth")


def beitscorepp_checkpoint(model_size):
    """Released BEiTScore++ checkpoint: $BEITSCORE_MODELS/beitscorepp/<size>/model.safetensors."""
    return os.path.join(MODELS_ROOT, "beitscorepp", model_size, "model.safetensors")


def finetuned_checkpoint(task, run, checkpoint="best"):
    """Checkpoint written by scripts/train.py: finetuning/<task>/run-<run>/checkpoint-<n|best>.pth."""
    checkpoint = "best" if checkpoint in (-1, "-1", None) else checkpoint
    return os.path.join(FINETUNE_ROOT, task, f"run-{run}", f"checkpoint-{checkpoint}.pth")


def model_id(quant, model_size, run=None, checkpoint=None, pretrained="vqa", variant="with_stage1", path=None):
    """Short name used for the folders of score files and calibration files."""
    if path is not None:
        return f"{Path(path).parent.name}_{Path(path).stem}"
    if quant:
        suffix = "" if checkpoint in (-1, "-1", None, "best") else f"_ckpt-{checkpoint}"
        return f"beitscorepp_{model_size}_run-{run}{suffix}"
    return f"beitscore_{model_size}_{pretrained}_{variant}"


def output_dir(kind, mid):
    path = os.path.join(OUTPUT_ROOT, kind, mid)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------- models
def run_config(ckpt_path):
    """Settings of the run that produced a checkpoint.

    Read from config.json next to the checkpoint (released models), else from the
    `args` saved by scripts/train.py. Returns a dict, possibly empty.
    """
    config = Path(ckpt_path).with_name("config.json")
    if config.exists():
        return json.loads(config.read_text())
    if str(ckpt_path).endswith(".pth") and os.path.exists(ckpt_path):
        args = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("args")
        if args is not None:
            return {k: v for k, v in vars(args).items() if isinstance(v, (bool, int, float, str, list))}
    return {}


def head_settings(checkpoint, chained_construction=None, quantiles=None):
    """BEiTScore++ head settings: explicit values first, then those recorded with the
    checkpoint (run_config), then the paper model's."""
    cfg = run_config(checkpoint)
    if chained_construction is None:
        chained_construction = cfg.get("chained_construction")
        if chained_construction is None:
            chained_construction = True
            print("Chained construction not recorded with the checkpoint, assuming the chained head")
    return dict(chained_construction=bool(chained_construction),
                quantiles=list(quantiles or cfg.get("quantiles") or [0.9, 0.85, 0.8, 0.75, 0.7]))


def build_model(model_size, quant, checkpoint, chained_construction=None, quantiles=None, device="cuda",
                drop_path_rate=0.0):
    """Create the model, load `checkpoint` and move it to `device` in eval mode.

    For BEiTScore++, `chained_construction` and `quantiles` default to the values of the
    run that produced the checkpoint (see run_config), and to the paper model otherwise
    (chained head, confidence levels [0.9, 0.85, 0.8, 0.75, 0.7]).
    """
    kwargs = {}
    if quant:
        kwargs = head_settings(checkpoint, chained_construction, quantiles)
    config = "beit3_{}_patch16_480_{}".format(model_size, "quant" if quant else "regression")
    model = create_model(config, pretrained=False, dropout=0, drop_path_rate=drop_path_rate, vocab_size=64010,
                         checkpoint_activations=False, **kwargs)
    load_model_and_may_interpolate_beitscore(checkpoint, model, "model|module", "", strict=True)
    return model.to(device).eval()



# ---------------------------------------------------------------------------- prediction
def select_interval(scores):
    """Return (low, mid, high) of the 90% interval from an (n, n_outputs) array.

    The 11-output model is ordered [q_0.05, ..., q_0.5, ..., q_0.95]; models with more
    than 25 outputs (the 1%-spaced ablation) use columns 9 and -10.
    """
    scores = np.asarray(scores, dtype=float)
    n_quantiles = scores.shape[1]
    id_mid = (n_quantiles - 1) // 2
    id_low, id_high = (9, -10) if n_quantiles > 25 else (0, -1)
    return scores[:, id_low], scores[:, id_mid], scores[:, id_high]


@torch.no_grad()
def predict(model, dataset, device, desc=None):
    """Score every item of `dataset` one at a time.

    Returns (scores, times_ms): scores is (n,) for BEiTScore and (n, n_outputs) for
    BEiTScore++; times_ms is the per-instance inference time.
    """
    device = torch.device(device)
    scores, times = [], []
    for row in tqdm(dataset, total=len(dataset), desc=desc):
        batch = {k: v.unsqueeze(0).to(device) for k, v in row.items()}
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        out = model(image=batch["image"], text_description=batch["language_tokens"],
                    padding_mask=batch["padding_mask"], only_inference=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        out = out.squeeze(0).float().cpu().numpy()
        scores.append(out if out.size > 1 else float(out))
    return np.stack(scores) if np.ndim(scores[0]) else np.asarray(scores), np.asarray(times)


def score_frame(dataset, scores, gt=None):
    """One row per instance: image_path, candidates, data_type, human_scores and the raw
    outputs (`score` for BEiTScore, `q00`..`q10` for BEiTScore++)."""
    items = dataset.items
    df = pd.DataFrame({
        "image_path": items["image_path"].tolist(),
        "candidates": items["candidates"].tolist(),
        "data_type": items["data_type"].tolist(),
    })
    for col in ("pair_id", "foil", "tag_combination", "cluster", "dataset"):
        if col in items.columns:
            df[col] = items[col].tolist()
    if gt is not None:
        df["human_scores"] = list(gt)
    scores = np.asarray(scores)
    if scores.ndim == 1:
        df["score"] = scores
    else:
        for i in range(scores.shape[1]):
            df[f"q{i:02d}"] = scores[:, i]
    return df



def save_scores(df, mid, name):
    path = os.path.join(output_dir("scores", mid), f"{name}.pkl")
    df.to_pickle(path)
    print(f"Scores written to {path}")
    return path


def print_timing(times_ms):
    t = np.asarray(times_ms)
    print(f"Inference time per instance: mean {t.mean():.2f} ms, median {np.median(t):.2f} ms, "
          f"P25 {np.percentile(t, 25):.2f} ms, P75 {np.percentile(t, 75):.2f} ms, "
          f"min/max {t.min():.2f} / {t.max():.2f} ms")


# ---------------------------------------------------------------------------- command line
def add_model_args(parser, quant_only=False):
    """Arguments that pick the model, shared by the evaluation scripts."""
    import argparse

    parser.add_argument("--model_size", required=True, choices=["base", "large"])
    if not quant_only:
        parser.add_argument("--quant", action="store_true", help="BEiTScore++ (interval) model instead of BEiTScore")
        parser.add_argument("--pretrained", default="vqa", choices=["vqa", "nlvr2"],
                            help="BEiTScore initialisation (released checkpoints)")
        parser.add_argument("--without_stage1", action="store_true", help="BEiTScore ablation without Stage 1")
        parser.add_argument("--without_stage2", action="store_true", help="BEiTScore ablation without Stage 2")
    parser.add_argument("--use_run", default=None, type=int,
                        help="Read finetuning/<task>/run-<n>/ instead of the released checkpoint")
    parser.add_argument("--checkpoint_run", default=-1, type=int, help="Checkpoint number in that run (-1: best)")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (overrides the two above)")
    parser.add_argument("--chained_construction", action=argparse.BooleanOptionalAction, default=None,
                        help="BEiTScore++ head type (default: as recorded with the checkpoint, else chained)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    return parser


def resolve_model(args, quant=None):
    """Return (checkpoint path, model id, quant) for parsed add_model_args arguments."""
    quant = getattr(args, "quant", True) if quant is None else quant
    if args.checkpoint:
        return args.checkpoint, model_id(quant, args.model_size, path=args.checkpoint), quant
    if quant:
        if args.use_run is None:
            ckpt = beitscorepp_checkpoint(args.model_size)
            return ckpt, f"beitscorepp_{args.model_size}", quant
        ckpt = finetuned_checkpoint("quant", args.use_run, args.checkpoint_run)
        return ckpt, model_id(True, args.model_size, run=args.use_run, checkpoint=args.checkpoint_run), quant
    variant = "without_stage1" if args.without_stage1 else "without_stage2" if args.without_stage2 else "with_stage1"
    if args.use_run is not None:
        ckpt = finetuned_checkpoint("regression", args.use_run, args.checkpoint_run)
        return ckpt, f"beitscore_{args.model_size}_run-{args.use_run}", quant
    return (beitscore_checkpoint(args.model_size, args.pretrained, variant),
            model_id(False, args.model_size, pretrained=args.pretrained, variant=variant), quant)


__all__ = ["beitscore_checkpoint", "beitscorepp_checkpoint", "finetuned_checkpoint", "model_id", "run_config", "build_model", "select_interval",
           "predict", "score_frame", "add_model_args", "resolve_model", "save_scores", "print_timing", "mean_human_score", "SENTENCEPIECE_MODEL"]

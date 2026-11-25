"""CapArena evaluation of BEiTScore with the competition's scoring protocol."""
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau

from timm.models import create_model
import beitscore.models  # noqa: F401  (registers the model configs)
import logging
# build_transform / default_loader are the same helpers the original BEiTDataset used.
from beitscore.data.datasets import get_sentencepiece_model_for_beit3, build_transform, default_loader
from beitscore.models.checkpoint import load_model_and_may_interpolate_beitscore
from tqdm import tqdm
from torch.utils.data import DataLoader
from beitscore.inference import beitscore_checkpoint
from beitscore.paths import DATA_ROOT, MODELS_ROOT, OUTPUT_ROOT


# ---- CapArena paths (under --base_path) --------------------------------------
# written by `python scripts/prepare_data/benchmarks.py download caparena`
CAPARENA_SUBDIR = "caparena"
CAPARENA_IMAGES_REL = os.path.join(CAPARENA_SUBDIR, "images")
CAPARENA_ANNOTS_REL = os.path.join(CAPARENA_SUBDIR, "caparena_annots_eval.json")

CAPARENA_HUMAN_RANKING = [
    "GPT-4o-0806", "human", "Gemini-2.0-flash-exp", "InternVL2-26B", "Gemini-1.5-pro-002",
    "Claude-3.5-Sonnet-0620", "GPT-4o-mini-0718", "LLama-3.2-90B", "Qwen2-VL-72B-Instruct",
    "CogVLM2-llama3-chat-19B", "MiniCPM-V2.6-8B", "Qwen2-VL-7B-Instruct", "Qwen2-VL-2B-Instruct",
    "LLaVA-1.6-34B", "LLaVA-1.5-7B",
]

def get_args():
    parser = argparse.ArgumentParser('BEiTScore - CapArena evaluation')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--base_path', type=str, default=DATA_ROOT, required=False,
                        help='Path to the root directory of the datasets')
    parser.add_argument("--model_size", type=str, required=True, choices=["base", "large"],
                        help="Model size: base or large")
    parser.add_argument("--without_stage1", action="store_true", help="Disable stage1")
    parser.add_argument("--without_stage2", action="store_true", help="Disable stage2")
    parser.add_argument("--pretrained", default="vqa", choices=["vqa", "nlvr2"], help="BEiTScore initialisation")
    # ---- CapArena-specific options ----
    parser.add_argument('--num_max_bpe_tokens', type=int, default=256,
                        help="Caption token cap. CapArena captions are long (100+ words); watch for truncation")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--out_dir', type=str, default=os.path.join(OUTPUT_ROOT, "caparena"))

    return parser.parse_args()


def setup_logger(rank=0, log_level=logging.INFO):
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if rank != 0:
        logger.disabled = True
    if not logger.hasHandlers() and rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


# ============================================================================
#  CapArena dataset
# ============================================================================
class CapArenaPairDataset(torch.utils.data.Dataset):
    """Evaluate BEiTScore on CapArena's human pairwise battles.

    caparena_annots_eval.json holds the battles; each has an image, two candidate
    captions (from two models) and the human preference (`winner` = winning model
    name, or a tie/skip marker). BEiTScore is a per-(image, caption) score, so we
    flatten to the UNIQUE (image, caption) instances, score each once in a single
    pass, then reorganize the scores back into battles for caption-level and
    model-level agreement. Non-decisive battles (tie/skip/bad) are dropped.

    __getitem__ yields {image, language_tokens, padding_mask, inst_id}.
    """

    KEYS = {
        "image":   "img",
        "model_a": "source1",
        "model_b": "source2",
        "cap_a":   "caption1",
        "cap_b":   "caption2",
        "label":   "winner",     # winning SOURCE NAME, or a tie/skip marker
    }
    # markers that are NOT a decisive win for either side -> excluded
    NONDECISIVE = {"tie", "draw", "equal", "both", "same", "bad", "skip", "none", 0.5, "0.5"}

    def __init__(self, annots_path, base_path, tokenizer, num_max_bpe_tokens=256, keys=None):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.num_max_bpe_tokens = num_max_bpe_tokens
        self.base_path = base_path
        self.loader = default_loader
        self.transform = build_transform()
        if keys:
            self.KEYS = {**self.KEYS, **keys}

        with open(annots_path) as f:
            self._raw = json.load(f)

        self._all = []                            # every battle, aligned 1:1 with self._raw
        self.battles = []                         # decisive battles only (label in {0,1})
        self._inst_index = {}                     # (image_path, caption) -> inst_id
        self.instances = []                       # list of (image_path, caption)
        self.models = set()
        self.n_nondecisive = 0                    # ties / skips (still scored, for export)

        # Score EVERY caption (BEiTScore always yields a value), so the exported
        # judge file covers all 6523 battles for the official caparena_metrics.py.
        # Keep only decisive battles in self.battles for the in-script previews.
        for rec in self._raw:
            b = self._parse_record(rec)
            b["inst_a"] = self._intern(b["image_path"], b["cap_a"])
            b["inst_b"] = self._intern(b["image_path"], b["cap_b"])
            self._all.append(b)
            if b["label"] == -1:
                self.n_nondecisive += 1
            else:
                self.models.update((b["model_a"], b["model_b"]))
                self.battles.append(b)

        self.models = sorted(self.models)

    # ---- record parsing -----------------------------------------------------
    def _parse_record(self, rec):
        k = self.KEYS
        model_a = rec[k["model_a"]]
        model_b = rec[k["model_b"]]
        return {
            "image_path": self._resolve_image(rec[k["image"]]),
            "model_a": model_a,
            "model_b": model_b,
            "cap_a": rec[k["cap_a"]],
            "cap_b": rec[k["cap_b"]],
            "label": self._norm_label(rec[k["label"]], model_a, model_b),
            "cluster": rec.get("cluster", "unknown"),
            "in_400": rec.get("in-400", False),
        }

    def _resolve_image(self, img):
        return img if os.path.isabs(img) else os.path.join(self.base_path, img)

    def _norm_label(self, label, model_a, model_b):
        """Map `winner` to {0: model_a wins, 1: model_b wins, -1: non-decisive}."""
        if isinstance(label, str) and label.strip().lower() in self.NONDECISIVE:
            return -1
        if label in self.NONDECISIVE:
            return -1
        if label == model_a:
            return 0
        if label == model_b:
            return 1
        generic = {0: 0, 1: 1, "0": 0, "1": 1, "A": 0, "B": 1}
        if label in generic:
            return generic[label]
        raise ValueError(
            f"Unrecognized winner {label!r} (model_a={model_a!r}, model_b={model_b!r}); "
            "add it to NONDECISIVE or extend _norm_label."
        )

    def _intern(self, image_path, caption):
        key = (image_path, caption)
        if key not in self._inst_index:
            self._inst_index[key] = len(self.instances)
            self.instances.append(key)
        return self._inst_index[key]

    # ---- image / text (mirrors the original BEiTDataset) --------------------
    def _get_image(self, image_path):
        return self.transform(self.loader(image_path))

    def _get_text_segment(self, text):
        tokens = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contain at least one token!")
        max_len = self.num_max_bpe_tokens
        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]
        tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        padding_mask = [0] * len(tokens)          # 0 = real token, 1 = padding
        return tokens, padding_mask

    # ---- torch Dataset API: iterate over UNIQUE instances ------------------
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        image_path, caption = self.instances[index]
        tokens, padding_mask = self._get_text_segment(caption)
        return {
            "image": self._get_image(image_path),
            "language_tokens": torch.tensor(tokens),
            "padding_mask": torch.tensor(padding_mask),
            "inst_id": index,
        }

    def collate_fn(self, batch):
        max_len = max(x["language_tokens"].size(0) for x in batch)
        images, toks, masks, ids = [], [], [], []
        for x in batch:
            pad = max_len - x["language_tokens"].size(0)
            toks.append(torch.cat([x["language_tokens"],
                                   torch.full((pad,), self.pad_token_id, dtype=torch.long)]))
            masks.append(torch.cat([x["padding_mask"],
                                    torch.ones(pad, dtype=torch.long)]))   # 1 = padding
            images.append(x["image"])
            ids.append(x["inst_id"])
        return {
            "image": torch.stack(images),
            "language_tokens": torch.stack(toks),
            "padding_mask": torch.stack(masks),
            "inst_id": torch.tensor(ids),
        }

    # ---- metrics computed exactly as the official repo ---------------------
    # BEiTScore's per-battle "judge": higher-scoring caption wins. Written in the
    # repo's sentence form so it satisfies BOTH cal_agreement (caption-level) and
    # calculate_elo_rankings (model-level, which tests `"1"/"2" in judge`).
    @staticmethod
    def _judge_str(sa, sb):
        return "Caption 1 is better." if sa >= sb else "Caption 2 is better."

    def caption_level_agreement(self, scores, include_tie=True, in_400=False):
        """Reproduces vlm_as_a_judge.cal_agreement(include_tie, in_400).
        Excludes battles involving `human`; overall + per level. With
        include_tie=True, human ties stay in the denominator (BEiTScore never
        ties, so it scores 0 on them)."""
        scores = np.asarray(scores, dtype=float)
        levels = {"overall": [], "level 1": [], "level 2": [], "level 3": [], "level 4": []}
        for b in self._all:
            if b["model_a"] == "human" or b["model_b"] == "human":
                continue
            if in_400 and not b["in_400"]:
                continue
            sa, sb = scores[b["inst_a"]], scores[b["inst_b"]]
            judge = self._judge_str(sa, sb)
            if b["label"] == 0:
                judge_human = "Caption 1 is better."
            elif b["label"] == 1:
                judge_human = "Caption 2 is better."
            else:
                judge_human = "Tie."
            cl = b["cluster"] if b["cluster"] in levels else None
            if not include_tie and judge_human == "Tie.":
                continue
            agree = 1 if judge in judge_human else 0
            levels["overall"].append(agree)
            if cl is not None:
                levels[cl].append(agree)

        def acc(lst):
            return sum(lst) / len(lst) if lst else None
        return {
            "accuracy": acc(levels["overall"]),
            "n_pairs": len(levels["overall"]),
            "per_cluster": {
                cl: {"accuracy": acc(levels[cl]), "n_pairs": len(levels[cl])}
                for cl in ("level 1", "level 2", "level 3", "level 4")
            },
        }

    @staticmethod
    def _compute_elo(battles, K=4, SCALE=400, BASE=10, INIT=1000):
        """Online Elo over a sequence of (source1, source2, winner_name)."""
        rating = defaultdict(lambda: INIT)
        for s1, s2, winner in battles:
            ra, rb = rating[s1], rating[s2]
            ea = 1.0 / (1 + BASE ** ((rb - ra) / SCALE))
            eb = 1.0 / (1 + BASE ** ((ra - rb) / SCALE))
            sa = 1.0 if winner == s1 else (0.0 if winner == s2 else 0.5)
            rating[s1] += K * (sa - ea)
            rating[s2] += K * (1 - sa - eb)
        return rating

    def model_level_agreement(self, scores, human_ranking=CAPARENA_HUMAN_RANKING,
                              bootstrap_rounds=1000, seed=42):
        """Reproduces cal_ranking.calculate_elo_rankings + caparena_metrics.
        cal_model_level_agreement: bootstrap Elo (K=4, 1000 rounds, median) over
        non-human battles using BEiTScore's pick, then Spearman/Kendall of that
        model order against the fixed human ranking."""
        scores = np.asarray(scores, dtype=float)
        battles = []
        for b in self._all:
            s1, s2 = b["model_a"], b["model_b"]
            if s1 == "human" or s2 == "human":
                continue
            sa, sb = scores[b["inst_a"]], scores[b["inst_b"]]
            winner = s1 if sa >= sb else s2
            battles.append((s1, s2, winner))

        rng = np.random.default_rng(seed)
        n = len(battles)
        acc = defaultdict(list)
        for _ in range(bootstrap_rounds):
            idx = rng.integers(0, n, size=n)
            r = self._compute_elo([battles[i] for i in idx])
            for m, v in r.items():
                acc[m].append(v)
        median = {m: float(np.median(v)) for m, v in acc.items()}

        sorted_models = sorted(median, key=median.get, reverse=True)
        if "human" in sorted_models:
            sorted_models.remove("human")
        sorted_models = [m for m in sorted_models if m in human_ranking]
        metric_rank = list(range(1, len(sorted_models) + 1))
        human_rank = [human_ranking.index(m) + 1 for m in sorted_models]
        n_ok = len(sorted_models) > 1
        return {
            "spearman": float(spearmanr(human_rank, metric_rank).correlation) if n_ok else float("nan"),
            "kendall": float(kendalltau(human_rank, metric_rank).correlation) if n_ok else float("nan"),
            "metric_order": sorted_models,
            "models_compared": sorted_models,
            "elo": median,
        }

    def export_judge_file(self, scores, out_path, judge_key="judge"):
        """Write every record back with a `judge` key in the repo's sentence form
        ("Caption 1 is better." / "Caption 2 is better."), which both cal_agreement
        and calculate_elo_rankings accept. Feed this straight to caparena_metrics.py."""
        scores = np.asarray(scores, dtype=float)
        out = []
        for rec, b in zip(self._raw, self._all):
            sa, sb = scores[b["inst_a"]], scores[b["inst_b"]]
            item = dict(rec)
            item[judge_key] = self._judge_str(sa, sb)
            item["beitscore_source1"] = float(sa)
            item["beitscore_source2"] = float(sb)
            out.append(item)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        return out_path


def _forward_score(model, batch, device):
    """Return a (B,) numpy array of BEiTScores for the batch.

    NOTE: match this call to however BEiTScore is invoked in the rest of your
    code (the regression head forward). Adjust kwargs / unpacking if needed.
    """
    image = batch["image"].to(device, non_blocking=True)
    tokens = batch["language_tokens"].to(device, non_blocking=True)
    pad = batch["padding_mask"].to(device, non_blocking=True)

    out = model(image=image, text_description=tokens, padding_mask=pad)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.float().squeeze(-1).detach().cpu().numpy()


@torch.no_grad()
def run_caparena(args, model, tokenizer, device, logger):
    annots = os.path.join(args.base_path, CAPARENA_ANNOTS_REL)
    images = os.path.join(args.base_path, CAPARENA_IMAGES_REL)

    logger.info(f"[CapArena] annotations: {annots}")
    logger.info(f"[CapArena] images:      {images}")

    ds = CapArenaPairDataset(
        annots_path=annots,
        base_path=images,
        tokenizer=tokenizer,
        num_max_bpe_tokens=args.num_max_bpe_tokens,
    )
    logger.info(f"[CapArena] {len(ds._all)} battles ({len(ds.battles)} decisive, "
                f"{ds.n_nondecisive} tie/skip) -> {len(ds)} unique (image, caption) "
                f"instances across {len(ds.models)} models")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=ds.collate_fn, num_workers=args.num_workers, pin_memory=True)

    # ---- single pass: score every unique instance once ----
    scores = np.empty(len(ds), dtype=np.float64)
    for batch in tqdm(dl, desc="Scoring CapArena instances"):
        out = _forward_score(model, batch, device)
        scores[batch["inst_id"].numpy()] = out

    # ---- caption-level (official cal_agreement) + model-level (bootstrap Elo) ----
    cap = ds.caption_level_agreement(scores, include_tie=True, in_400=False)
    acc = cap["accuracy"]
    logger.info(f"[CapArena] caption-level agreement: "
                f"{acc:.4f} over {cap['n_pairs']} pairs" if acc is not None
                else "[CapArena] caption-level agreement: n/a")
    for cl, v in cap["per_cluster"].items():
        a = v["accuracy"]
        logger.info(f"[CapArena]   {cl}: {a:.4f} ({v['n_pairs']} pairs)" if a is not None
                    else f"[CapArena]   {cl}: n/a (0 pairs)")

    logger.info("[CapArena] computing bootstrap Elo model-level agreement (1000 rounds)...")
    ml = ds.model_level_agreement(scores)
    if len(ml["models_compared"]) > 1:
        logger.info(f"[CapArena] model-level Spearman: {ml['spearman']:.4f}  "
                    f"Kendall: {ml['kendall']:.4f}  (n={len(ml['models_compared'])} models)")
        logger.info(f"[CapArena] metric ranking: {ml['metric_order']}")
    else:
        logger.warning("[CapArena] model-level: fewer than 2 models overlap CAPARENA_HUMAN_RANKING - "
                       "check the model-name strings in the annotations.")

    # ---- persist results ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"caparena_scores_{args.model_size}.npy", scores)
    judge_path = out_dir / f"caparena_annots_eval_beitscore_{args.model_size}.json"
    ds.export_judge_file(scores, str(judge_path))
    with open(out_dir / f"caparena_results_{args.model_size}.json", "w") as f:
        json.dump({
            "caption_level": cap,
            "model_level": {k: ml[k] for k in ("spearman", "kendall", "models_compared")},
        }, f, indent=2)
    logger.info(f"[CapArena] wrote scores, judge file, and summary to {out_dir}")
    logger.info(f"[CapArena] For official numbers run: "
                f"python caparena_metrics.py --eval_dir {judge_path}")

    return {"caption_level": cap, "model_level": ml}


def main(args):
    device = torch.device(args.device)
    logger = setup_logger(rank=0)

    model_config = "beit3_{}_patch16_480_regression".format(args.model_size)
    if args.without_stage1:
        suffix = "without_stage1"
    elif args.without_stage2:
        suffix = "without_stage2"
    else:
        suffix = "with_stage1"
    checkpoint = beitscore_checkpoint(args.model_size, args.pretrained, suffix)
    sentencepiece_model = MODELS_ROOT + "/beit3.spm"

    model = create_model(
        model_config,
        pretrained=False,
        dropout=0,
        drop_path_rate=0.15,
        vocab_size=64010,
        checkpoint_activations=False,
    )

    logger.info("Start Interpolate")
    load_model_and_may_interpolate_beitscore(checkpoint, model, 'model|module', '')
    logger.info("Send Model to GPU")
    tokenizer = get_sentencepiece_model_for_beit3(sentencepiece_model)
    model.to(device)
    model.eval()

    run_caparena(args, model, tokenizer, device, logger)


if __name__ == "__main__":
    args = get_args()
    main(args)

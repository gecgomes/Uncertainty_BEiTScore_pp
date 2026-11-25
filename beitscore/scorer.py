"""Scoring API for BEiTScore and BEiTScore++.

    from beitscore import BEiTScore, BEiTScorePP

    metric = BEiTScorePP.from_pretrained("large")          # $BEITSCORE_MODELS/beitscorepp/large/
    out = metric.score(["cat.jpg"], ["A cat sleeping on a sofa."])
    out["median"], out["low"], out["high"]                  # calibrated 90% interval

    BEiTScore.from_pretrained("large").score(images, captions)   # one score per pair, in [0, 1]

Images can be paths or PIL images. Captions are tokenised with the BEiT-3 SentencePiece
model and truncated to 254 tokens, as during training. Scores are computed in batches
with padding; this matches scoring one pair at a time up to floating-point noise.
"""
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.folder import default_loader

from .data.collate import merge_batch_tensors_by_dict_key
from .data.datasets import build_transform, get_sentencepiece_model_for_beit3
from .inference import (SENTENCEPIECE_MODEL, beitscore_checkpoint, beitscorepp_checkpoint, build_model,
                        select_interval)
from .pp.calibration.clustered import IDENTITY, Calibration

ImageLike = Union[str, os.PathLike, Image.Image]


class _Scorer:
    quant = False

    def __init__(self, model, tokenizer, device, max_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.transform = build_transform()

    @staticmethod
    def _device(device):
        return device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _tokens(self, caption):
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(caption))
        if not ids:
            raise ValueError("Empty caption")
        ids = ids[: self.max_tokens - 2]
        ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
        return ids, [0] * len(ids)

    def _image(self, image):
        if not isinstance(image, Image.Image):
            image = default_loader(str(image))
        return self.transform(image.convert("RGB"))

    @torch.no_grad()
    def _raw(self, images: Sequence[ImageLike], captions: Sequence[str], batch_size: int):
        if len(images) != len(captions):
            raise ValueError(f"{len(images)} images but {len(captions)} captions")
        outputs = []
        for start in range(0, len(images), batch_size):
            batch = []
            for image, caption in zip(images[start:start + batch_size], captions[start:start + batch_size]):
                tokens, mask = self._tokens(caption)
                batch.append({"image": self._image(image), "language_tokens": tokens, "padding_mask": mask})
            batch = merge_batch_tensors_by_dict_key(batch)
            out = self.model(image=batch["image"].to(self.device),
                             text_description=batch["language_tokens"].to(self.device),
                             padding_mask=batch["padding_mask"].to(self.device), only_inference=True)
            outputs.append(out.float().cpu().numpy())
        return np.concatenate(outputs, axis=0)


class BEiTScore(_Scorer):
    """Reference-free caption score in [0, 1]."""

    @classmethod
    def from_pretrained(cls, model_size="large", checkpoint: Optional[str] = None, pretrained="vqa",
                        variant="with_stage1", device: Optional[str] = None):
        device = cls._device(device)
        checkpoint = checkpoint or beitscore_checkpoint(model_size, pretrained, variant)
        model = build_model(model_size, False, checkpoint, device=device)
        return cls(model, get_sentencepiece_model_for_beit3(SENTENCEPIECE_MODEL), device)

    def score(self, images: Sequence[ImageLike], captions: Sequence[str], batch_size: int = 16) -> np.ndarray:
        return self._raw(images, captions, batch_size)[:, 0]


class BEiTScorePP(_Scorer):
    """Median score and a 90% interval, calibrated with the model's calibration.json."""
    quant = True

    def __init__(self, model, tokenizer, device, calibration: Calibration = IDENTITY, **kwargs):
        super().__init__(model, tokenizer, device, **kwargs)
        self.calibration = calibration

    @classmethod
    def from_pretrained(cls, model_size="large", checkpoint: Optional[str] = None,
                        calibration: Optional[str] = "auto", device: Optional[str] = None):
        """calibration: "auto" (calibration.json next to the checkpoint), a path, or None
        for the raw intervals."""
        device = cls._device(device)
        checkpoint = checkpoint or beitscorepp_checkpoint(model_size)
        model = build_model(model_size, True, checkpoint, device=device)
        if calibration == "auto":
            path = os.path.join(os.path.dirname(checkpoint), "calibration.json")
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} not found; pass calibration=None for the raw intervals")
            calibration = path
        cal = Calibration.load(calibration) if calibration else IDENTITY
        return cls(model, get_sentencepiece_model_for_beit3(SENTENCEPIECE_MODEL), device, calibration=cal)

    def score(self, images: Sequence[ImageLike], captions: Sequence[str], batch_size: int = 16,
              calibrated: bool = True, return_quantiles: bool = False) -> dict:
        """Return {"median", "low", "high"} (arrays of len(images)); with return_quantiles,
        also "quantiles": the raw (n, 11) outputs, sorted by quantile level."""
        raw = self._raw(images, captions, batch_size)
        low, mid, high = select_interval(raw)
        if calibrated:
            low, mid, high = self.calibration.apply(low, mid, high)
        out = {"median": mid, "low": low, "high": high}
        if return_quantiles:
            out["quantiles"] = raw
        return out


__all__: List[str] = ["BEiTScore", "BEiTScorePP"]

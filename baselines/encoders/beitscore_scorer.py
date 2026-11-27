"""BEiTScore and BEiTScore++ (base and large) through the scoring API of the beitscore package.

Weights: the downloaded release (scripts/download_release.py models) unless a checkpoint is passed
with --beitscore_<size>_checkpoint or --beitscorepp_<size>_checkpoint. For BEiTScore++ the score is
the calibrated median. Captions longer than 256 BPE tokens are truncated, as in the papers."""
import os

from beitscore import BEiTScore, BEiTScorePP
from beitscore.inference import beitscore_checkpoint, beitscorepp_checkpoint


def _checkpoint(path, kind):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python scripts/download_release.py models` "
                                f"or pass --{kind}_checkpoint <file>")
    return path


class _BEiTScore:
    size = None

    def __init__(self, opts):
        ckpt = _checkpoint(getattr(opts, f"beitscore_{self.size}_checkpoint", None)
                           or beitscore_checkpoint(self.size), f"beitscore_{self.size}")
        print(f"  BEiTScore {self.size}: {ckpt}")
        self.model = BEiTScore.from_pretrained(self.size, checkpoint=ckpt, device=opts.device)
        self.batch_size = opts.batch_size
        self.info = {"size": self.size, "checkpoint": ckpt}

    def score(self, images, captions):
        return self.model.score(list(images), list(captions), batch_size=self.batch_size)


class _BEiTScorePP:
    size = None

    def __init__(self, opts):
        ckpt = _checkpoint(getattr(opts, f"beitscorepp_{self.size}_checkpoint", None)
                           or beitscorepp_checkpoint(self.size), f"beitscorepp_{self.size}")
        print(f"  BEiTScore++ {self.size}: {ckpt}")
        self.model = BEiTScorePP.from_pretrained(self.size, checkpoint=ckpt, device=opts.device)
        self.batch_size = opts.batch_size
        self.info = {"size": self.size, "checkpoint": ckpt, "score": "calibrated median"}

    def score(self, images, captions):
        return self.model.score(list(images), list(captions), batch_size=self.batch_size)["median"]


class BEiTScoreB(_BEiTScore):
    size = "base"


class BEiTScoreL(_BEiTScore):
    size = "large"


class BEiTScorePPB(_BEiTScorePP):
    size = "base"


class BEiTScorePPL(_BEiTScorePP):
    size = "large"

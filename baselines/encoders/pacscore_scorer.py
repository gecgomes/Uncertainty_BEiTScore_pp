"""PAC-S++ (Sarto et al., 2025), with the original code of github.com/aimagelab/pacscore
(encoders/repos/pacscore, from encoders/setup.sh): CLIP with LoRA (r = 4), the PAC++ checkpoints, w = 2.0, and
the "A photo depicts" prompt of PAC-S. Captions are truncated to CLIP's 77 tokens by the PAC-S
code itself."""
import importlib.util
import os
import sys
import types

import numpy as np
import torch

CHECKPOINTS = {"ViT-B/32": "PAC++_clip_ViT-B-32.pth", "ViT-L/14": "PAC++_clip_ViT-L-14.pth"}


class _PACScorePP:
    backbone = None

    def __init__(self, opts):
        repo = os.path.abspath(opts.pacscore_repo)
        ckpt = os.path.join(repo, "checkpoints", CHECKPOINTS[self.backbone])
        if not os.path.isdir(os.path.join(repo, "models", "clip_lora")) or not os.path.exists(ckpt):
            raise FileNotFoundError(f"PAC-S++ code or {os.path.basename(ckpt)} not found in {repo}; run encoders/setup.sh")
        sys.path.insert(0, repo)  # its modules import "models.clip" from the repository root
        from compat import pkg_resources_packaging
        pkg_resources_packaging()
        from models.clip_lora import clip_lora

        spec = importlib.util.spec_from_file_location("pac_score", os.path.join(repo, "evaluation", "pac_score", "pac_score.py"))
        pac = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pac)
        pac.tqdm = types.SimpleNamespace(tqdm=lambda it, *a, **k: it)  # the runner shows the progress
        self.PACScore = pac.PACScore
        self.device = opts.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, self.preprocess = clip_lora.load(self.backbone, device=self.device, lora=4)
        model = model.to(self.device).float()
        model.load_state_dict(torch.load(ckpt, map_location=self.device)["state_dict"])
        self.model = model.eval()
        self.chunk_size = 2048  # each PAC-S call starts its own data loaders, so fewer, larger calls
        self.info = {"backbone": self.backbone, "checkpoint": CHECKPOINTS[self.backbone], "w": 2.0, "lora": 4}

    @torch.no_grad()
    def score(self, images, captions):
        _, per_instance, _, _ = self.PACScore(self.model, self.preprocess, list(images), list(captions), self.device, w=2.0)
        return np.asarray(per_instance, dtype=float)


class PACScorePPB(_PACScorePP):
    backbone = "ViT-B/32"


class PACScorePPL(_PACScorePP):
    backbone = "ViT-L/14"

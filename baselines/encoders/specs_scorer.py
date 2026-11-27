"""SPECS (github.com/mbzuai-nlp/SPECS): LongCLIP fine-tuned for detailed captions; the score is
(cos(image, caption) + 1) / 2, clipped at 0. Captions longer than its 248-token context are
truncated. --specs_repo is the clone made by encoders/setup.sh; the checkpoint is --specs_checkpoint,
else <specs_repo>/checkpoints/spec.pt, else spec.pt from Hugging Face (Xiaohud/SPECS, the link of
the SPECS model zoo)."""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SPECS:
    def __init__(self, opts):
        repo = os.path.abspath(opts.specs_repo)
        if not os.path.isfile(os.path.join(repo, "model", "longclip.py")):
            raise FileNotFoundError(f"SPECS code not found in {repo}; run encoders/setup.sh or pass --specs_repo")
        sys.path.insert(0, repo)
        from compat import pkg_resources_packaging
        pkg_resources_packaging()
        from model import longclip

        ckpt = opts.specs_checkpoint or os.path.join(repo, "checkpoints", "spec.pt")
        if not os.path.exists(ckpt):
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download("Xiaohud/SPECS", "spec.pt")
        self.longclip = longclip
        self.device = opts.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = longclip.load(ckpt, device=self.device)
        self.model.eval()
        self.context = int(getattr(self.model, "context_length", 248))
        self.batch_size = opts.batch_size
        self.info = {"checkpoint": os.path.basename(ckpt), "context": self.context}

    @torch.no_grad()
    def score(self, images, captions):
        out = []
        for i in range(0, len(images), self.batch_size):
            px = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in images[i:i + self.batch_size]])
            tok = self.longclip.tokenize(list(captions[i:i + self.batch_size]), context_length=self.context,
                                         truncate=True)
            a = self.model.encode_image(px.to(self.device)).float()
            b = self.model.encode_text(tok.to(self.device)).float()
            out.append(((F.cosine_similarity(a, b, dim=-1) + 1) / 2).clamp(min=0).cpu().numpy())
        return np.concatenate(out)

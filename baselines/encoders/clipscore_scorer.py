"""CLIPScore (Hessel et al., 2021): 2.5 * max(cos(image, caption), 0), with the LAION CLIP
models of the BEiTScore evaluation (ViT-B/32 and ViT-L/14, laion2B). CLIP reads at most 77
tokens, so longer captions are truncated."""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

W = 2.5


class _CLIPScore:
    name = None

    def __init__(self, opts):
        self.device = opts.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(self.name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.name)
        self.batch_size = opts.batch_size
        self.info = {"model": self.name, "w": W, "max_tokens": 77}

    @torch.no_grad()
    def score(self, images, captions):
        out = []
        for i in range(0, len(images), self.batch_size):
            ims = [Image.open(p).convert("RGB") for p in images[i:i + self.batch_size]]
            px = self.processor(images=ims, return_tensors="pt")["pixel_values"].to(self.device)
            tok = self.processor.tokenizer(list(captions[i:i + self.batch_size]), padding=True, truncation=True,
                                           max_length=77, return_tensors="pt").to(self.device)
            a = torch.nn.functional.normalize(self.model.get_image_features(pixel_values=px), dim=-1)
            b = torch.nn.functional.normalize(self.model.get_text_features(**tok), dim=-1)
            out.append((W * (a * b).sum(-1).clamp(min=0)).float().cpu().numpy())
        return np.concatenate(out)


class CLIPScoreB(_CLIPScore):
    name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"


class CLIPScoreL(_CLIPScore):
    name = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"

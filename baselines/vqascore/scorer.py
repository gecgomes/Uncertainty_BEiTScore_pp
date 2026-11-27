"""VQAScore (Lin et al., 2024) with CLIP-FlanT5-XXL, through the t2v_metrics package
(pip install t2v_metrics): the probability of "Yes" to "Does this figure show {caption}?"."""
import numpy as np
from tqdm import tqdm


class VQAScore:
    def __init__(self, opts):
        import t2v_metrics

        self.model = t2v_metrics.VQAScore(model="clip-flant5-xxl")
        self.info = {"model": "clip-flant5-xxl"}

    def score(self, images, captions):
        out = []
        for image, caption in tqdm(list(zip(images, captions)), desc="VQAScore", leave=False):
            try:
                out.append(float(self.model(images=[image], texts=[caption]).cpu().item()))
            except Exception as e:  # e.g. a caption longer than the model accepts
                print(f"  VQAScore failed on {image}: {e}")
                out.append(float("nan"))
        return np.asarray(out)

"""Training and evaluation handler for the BEiTScore++ quantile model."""
import numpy as np
import torch

from ..engine import TaskHandler
from .. import metrics
from .losses import CenteredIntervalLoss, PinBallLoss


class QuantHandler(TaskHandler):
    def __init__(self, CI = [0.90], calibration=False, coef=1, criterion="CI") -> None:
        super().__init__()
        self.scores = []       
        self.human_scores = []
        if criterion == "CI":
            self.criterion = CenteredIntervalLoss(reduction='mean', CI=CI, coef = coef)
        elif criterion == "pinball":
            self.criterion = PinBallLoss(reduction='mean', CI=CI)
        self.alpha = self.criterion.alpha
        self.calibration = calibration

    def train_batch(self, model, image, language_tokens, padding_mask, targets, targets_mask):
        logits = model(
            image=image, text_description=language_tokens,
            padding_mask=padding_mask)
        
        loss = self.criterion(input=logits.float(), target=targets.float())
        targets.detach().cpu()
        return {"loss": loss}

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        self.split = data_loader.dataset.split
        self.scores = []
        self.human_scores = []

    def eval_batch(self, model, image, language_tokens, padding_mask, targets,targets_mask):
        logits = model(
            image=image, text_description=language_tokens,
            padding_mask=padding_mask)

        self.scores.append(logits.cpu())
        
        decoded_targets = []
        for target_row, mask_row in zip(targets, targets_mask):
            valid_values = target_row[mask_row].tolist()
            decoded_targets.append(valid_values)
        self.human_scores += decoded_targets

    def after_eval(self, return_results, **kwargs):
        scores = torch.cat(self.scores).float().numpy()  # (N, 2k + 1), sorted by quantile level
        gt_mean = metrics.mean_human_score(self.human_scores)
        alphas = [float(a) for a in self.alpha]
        k = len(alphas)
        mid = scores[:, k]

        # Annotation-level arrays (one row per annotation), as used for the loss and correlations.
        repeat = np.array([len(h) if isinstance(h, (list, tuple, np.ndarray)) else 1 for h in self.human_scores])
        _, gt = metrics.explode_pairs(mid, self.human_scores)
        mid_e = np.repeat(mid, repeat)

        results = {}
        loss_lower_upper_quantile = 0
        for i, alpha in enumerate(alphas):
            ci = round((1 - alpha * 2) * 100, 4)  # key suffix, e.g. "ups_90.0"
            low_e = np.repeat(scores[:, i], repeat)
            high_e = np.repeat(scores[:, -(i + 1)], repeat)
            loss_lower_upper_quantile += (high_e - low_e) \
                + np.where(gt < low_e, (1 / alpha) * (low_e - gt), 0) \
                + np.where(gt > high_e, (1 / alpha) * (gt - high_e), 0)
            m = metrics.interval_metrics(gt, low_e, mid_e, high_e)
            results[f"sharpness_mean_{ci}"] = m["sharpness"]
            results[f"sharpness_std_{ci}"] = m["sharpness_std"]
            results[f"coverage_{ci}"] = m["coverage"]
            results[f"ups_{ci}"] = m["ups"]
            results[f"bimodality_low_{ci}"] = metrics.bimodality_coefficient(low_e)
            results[f"bimodality_high_{ci}"] = metrics.bimodality_coefficient(high_e)

        mae = float(np.mean(np.abs(mid_e - gt)))  # on the [0, 1] scale of the targets
        results["loss"] = np.mean(loss_lower_upper_quantile) + mae
        results["mae"] = mae
        results.update(metrics.correlations(mid, self.human_scores))
        results["binomiality_mid"] = metrics.bimodality_coefficient(mid_e)
        results["ece"] = metrics.quantile_ece(scores, gt_mean, sorted(alphas + [1 - a for a in alphas] + [0.5]))
        if return_results:
            return results

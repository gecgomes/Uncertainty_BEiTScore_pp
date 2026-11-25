"""Training objectives for BEiTScore++."""
import torch
import torch.nn as nn


class CenteredIntervalLoss(nn.Module):
    """Centered interval loss (Chung et al., 2021) for the k intervals of confidence levels CI,
    plus an L1 term on the median weighted by `coef`. CI=[0.9] gives the interval between the
    0.05 and 0.95 quantiles; the model outputs the 2k + 1 quantiles in increasing order."""
    def __init__(self, reduction='mean', CI = [0.9], coef = 1):
        super().__init__()
        self.CI = sorted(CI, reverse=True)
        self.alpha = (1 - torch.tensor(self.CI)) / 2
        self.reduction = reduction
        self.map_loss = nn.L1Loss(reduction="mean")
        self.coef = coef

    def forward(self, input, target):
        # input: (B, 2k + 1), the k lower quantiles, the median and the k upper quantiles
        # target: (B,)
        assert input.shape[1] == len(self.CI)*2+1, "Expected 2k + 1 outputs: k lower quantiles, the median, k upper quantiles"
        
        loss_lower_upper_quantile = torch.zeros_like(target)
        for a in range(len(self.alpha)): 
            alpha = self.alpha[a]
            qm = input[:,a]
            qM = input[:,-(a+1)]
            
            loss_calibration_lower_quantile = torch.where(target < qm, (1/alpha)*(qm-target), 0)
            loss_calibration_upper_quantile = torch.where(target > qM, (1/alpha)*(target-qM), 0)
            
            loss_lower_upper_quantile+= (qM - qm) + loss_calibration_lower_quantile + loss_calibration_upper_quantile
            
        # Median term: weight `coef` (--median_weight, default 1) on the L1 error of the median.
        loss_mid_quantile = self.coef*self.map_loss(input[:,len(self.alpha)],target)

        loss = loss_lower_upper_quantile.mean() + loss_mid_quantile.mean() 
        return loss


class PinBallLoss(nn.Module):
    """Pinball loss summed over the 2k + 1 quantiles of the confidence levels CI."""
    def __init__(self, reduction='mean', CI=[0.9]):
        super().__init__()
        self.CI = sorted(CI, reverse=True)
        self.alpha = (1 - torch.tensor(self.CI)) / 2
        self.quantiles = self.alpha.tolist() + [0.5] + (1 - self.alpha).tolist()
        self.quantiles = sorted(self.quantiles, reverse=False)
        self.reduction = reduction

    def pinball(self, pred, target, q):
        diff = target - pred
        return torch.where(diff >= 0, q * diff, (q - 1) * diff)

    def forward(self, input, target):
        # input: (B, N), the quantiles in increasing order
        # target: (B,)
        
        loss = sum(
            self.pinball(input[:, i], target, q)
            for i, q in enumerate(self.quantiles)
        )  # (B,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


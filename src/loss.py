import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=pos_weight
        )
        if mask is not None:
            loss = (loss * mask).mean() * 100
        else:
            loss = loss.mean() * 100
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(
                f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore."
            )

        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=pos_weight
        )
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        if mask is not None:
            loss = (loss * mask).mean() * 100
        else:
            loss = loss.mean() * 100

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, square_denominator: bool = True):
        super().__init__()
        self.smooth = smooth
        self.square_denominator = square_denominator

    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        p = torch.sigmoid(logits)
        if mask is not None:
            p = p * mask
            labels = labels * mask

        if self.square_denominator:
            numerator = 2 * (p * labels).sum() + self.smooth
            denominator = (p * p).sum() + (labels * labels).sum() + self.smooth
        else:
            numerator = 2 * (p * labels).sum() + self.smooth
            denominator = p.sum() + labels.sum() + self.smooth

        return (1 - numerator / denominator) * 100


class DiceFocalLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 1.0,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        dice_loss = self.dice(logits, labels, mask=mask)
        focal_loss = self.focal(logits, labels, mask=mask, pos_weight=pos_weight)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        scores: torch.tensor,
        positions: list[int],
        mask: torch.tensor,
        prob_mask: torch.tensor = None,
    ) -> torch.tensor:
        batch_size, seq_length = scores.size(0), scores.size(1)
        scores = scores / self.tau
        if len(scores.shape) == 3:
            scores = scores.view(batch_size, -1)
            mask = mask.view(batch_size, -1)
            log_probs = self.masked_log_softmax(scores, mask)
            log_probs = log_probs.view(batch_size, seq_length, seq_length)
            start_positions, end_positions = positions
            batch_indices = list(range(batch_size))
            log_probs = log_probs[batch_indices, start_positions, end_positions]
        else:
            log_probs = self.masked_log_softmax(scores, mask)
            batch_indices = list(range(batch_size))
            log_probs = log_probs[batch_indices, positions]
        if prob_mask is not None:
            log_probs = log_probs * prob_mask
        return -log_probs.mean()

    def masked_log_softmax(
        self, vector: torch.tensor, mask: torch.tensor, dim: int = -1
    ) -> torch.tensor:
        if mask is not None:
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            vector = vector + (mask + self.tiny_value_of_dtype(vector.dtype)).log()
        return torch.nn.functional.log_softmax(vector, dim=dim)

    def tiny_value_of_dtype(self, dtype: torch.dtype) -> float:
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double or dtype == torch.bfloat16:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

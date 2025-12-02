import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        loss = F.binary_cross_entropy_with_logits(
            logits, 
            labels,
            reduction="none",
            pos_weight=pos_weight
        )
        if mask is not None:
            loss = (loss * mask).sum()
        else:
            loss = loss.sum()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, labels, mask=None, pos_weight=None, **kwargs):
        if not (0 <= self.alpha <= 1) and self.alpha != -1:
            raise ValueError(f"Invalid alpha value: {self.alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
        p_t = p * labels + (1 - p) * (1 - labels)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            loss = alpha_t * loss

        if mask is not None:
            loss = (loss * mask).sum()
        else:
            loss = loss.sum()

        return loss


class ContrastiveLoss(nn.Module):
    def forward(
        self, 
        scores: torch.tensor, 
        positions: list[int], 
        mask: torch.tensor, 
        prob_mask: torch.tensor = None
    ) -> torch.tensor:
        batch_size, seq_length = scores.size(0), scores.size(1)
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
        return - log_probs.mean()

    def masked_log_softmax(self, vector: torch.tensor, mask: torch.tensor, dim: int = -1) -> torch.tensor:
        if mask is not None:
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            vector = vector + (mask + self.tiny_value_of_dtype(vector.dtype)).log()
        return torch.nn.functional.log_softmax(vector, dim=dim)

    def tiny_value_of_dtype(self, dtype: torch.dtype) -> float:
        if not dtype.is_floating_point:
            raise TypeError("Only supports floating point dtypes.")
        if dtype == torch.float or dtype == torch.double:
            return 1e-13
        elif dtype == torch.half:
            return 1e-4
        else:
            raise TypeError("Does not support dtype " + str(dtype))

class TokenizationAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask=None, pos_weight=None, **kwargs):
        base_loss = F.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )

        pos = labels.float().sum(dim=tuple(range(1, labels.dim()))).view(-1)
        total = mask.float().sum(dim=tuple(range(1, mask.dim()))).view(-1)
        neg = total - pos
        pos_weight = neg / (pos + 1e-9)

        loss = torch.where(
            labels.bool(),
            base_loss * pos_weight.unsqueeze(1).unsqueeze(1),
            base_loss,
        )

        if mask is not None:
            loss = (loss * mask).sum()
        else:
            loss = loss.sum()
        return loss


class JGMakerLoss(nn.Module):
    def __init__(self, total_steps: int, k: float = 0.01, threshold: float = 0.5):
        super().__init__()
        self.total_steps = int(total_steps)
        self.k = max(1.0, float(k) * float(total_steps))
        self.threshold = threshold
        self.step = 0

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask=None, pos_weight=None, **kwargs):
        if self.training:
            self.step += 1/3
        ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = probs > self.threshold
            pos = labels.bool()
            neg = ~pos
            tp = pos & preds
            fn = pos & (~preds)
            fp = neg & preds
            tn = neg & (~preds)

        # Create scalars directly instead of tensors to avoid accumulation
        denom = torch.log1p(torch.tensor(self.total_steps / self.k, device=logits.device, dtype=logits.dtype))
        alpha = torch.log1p(torch.tensor(self.step, device=logits.device, dtype=logits.dtype) / self.k) / denom
        beta = 1.0 - alpha

        types_mask = tp | fp | fn
        weights = types_mask.to(logits.dtype) * alpha + tn.to(logits.dtype) * beta

        loss = (ce * weights)
        if mask is not None:
            loss = (loss * mask).sum()
        else:
            loss = loss.sum()

        return loss
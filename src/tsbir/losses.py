from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.nn.functional import all_gather


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_negative: float = 4.0, gamma_positive: float = 1.0, clip: float = 0.05):
        super().__init__()
        self.gamma_negative = gamma_negative
        self.gamma_positive = gamma_positive
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        positive = torch.sigmoid(logits)
        negative = 1.0 - positive
        if self.clip > 0:
            negative = (negative + self.clip).clamp(max=1.0)

        loss = targets * torch.log(positive.clamp(min=1e-8))
        loss += (1.0 - targets) * torch.log(negative.clamp(min=1e-8))

        if self.gamma_negative > 0 or self.gamma_positive > 0:
            positive_probability = positive * targets
            negative_probability = negative * (1.0 - targets)
            asymmetric_weight = torch.pow(
                1.0 - positive_probability - negative_probability,
                self.gamma_positive * targets + self.gamma_negative * (1.0 - targets),
            )
            loss *= asymmetric_weight
        return -loss.sum() / logits.shape[0]


def _gather(features: torch.Tensor) -> torch.Tensor:
    if not dist.is_available() or not dist.is_initialized():
        return features
    return torch.cat(all_gather(features), dim=0)


def distributed_contrastive_loss(
    image_features: torch.Tensor,
    query_features: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    all_images = _gather(image_features)
    all_queries = _gather(query_features)
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0
    targets = rank * image_features.shape[0] + torch.arange(image_features.shape[0], device=image_features.device)
    query_logits = logit_scale * query_features @ all_images.t()
    image_logits = logit_scale * image_features @ all_queries.t()
    return 0.5 * (F.cross_entropy(query_logits, targets) + F.cross_entropy(image_logits, targets))

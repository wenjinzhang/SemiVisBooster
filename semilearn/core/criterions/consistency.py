
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F

from .cross_entropy import ce_loss



def consistency_loss(logits, targets, name='ce', mask=None):
    """
    consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagation, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()



class ConsistencyLoss(nn.Module):
    """
    Wrapper for consistency loss
    """
    def forward(self, logits, targets, name='ce', mask=None):
        return consistency_loss(logits, targets, name, mask)
    


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # You can provide a list or None for automatic balancing
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target, name = "ce", mask=None):
        ce_loss = consistency_loss(logits, target, name, mask)

        if self.alpha is not None:
            if mask is None:
                alpha_c = self.alpha[target]
            else:
                alpha_c = self.alpha[target[mask!=0]]
            focal_loss = alpha_c * (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - torch.exp(-ce_loss)) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
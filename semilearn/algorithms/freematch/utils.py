# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np

from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class FreeMatchThresholdingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.999, use_onethres = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        if use_onethres:
            self.p_model = torch.ones((self.num_classes))
        else:
            self.p_model = torch.ones((self.num_classes))/self.num_classes
            
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        # self.label_hist_clean = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb, mask = None):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        # if mask is not None:
        #     temp_idx = max_idx[mask!= 0]
        #     if mask.sum() > 0:
        #         hist_clean = torch.bincount(temp_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype)
        #         # print("hist_clean", hist_clean)
        #         self.label_hist_clean = self.label_hist_clean * self.m + (1 - self.m) * (hist_clean / hist_clean.sum())
                # print("update_hist_clean", self.label_hist_clean)
                # exit()

        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p
        # algorithm.label_hist_clean = self.label_hist_clean
    

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, softmax_x_ulb=True, update = True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        
        # if not self.label_hist_clean.is_cuda:
        #     self.label_hist_clean = self.label_hist_clean.to(logits_x_ulb.device)

        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        mask = max_probs.ge(self.time_p * mod[max_idx]).to(max_probs.dtype)
        if update:
            self.update(algorithm, probs_x_ulb)
        return mask


def mixup_data(x, y, alpha=1.0):
    """Applies mixup to the input data and labels."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Applies cutmix to the input data and labels."""
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()

    # Generate the bounding box for cutmix
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[index]
    
    return x, y_a, y_b, lam
        

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Uniformly sample the bounding box center
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        print(f"[DEBUG] features.shape: {features.shape}")
        print(f"[DEBUG] labels: {labels}")

        if torch.isnan(features).any():
            print("[ERROR] NaN detected in features before processing!")
            return torch.tensor(float('nan'), device=device)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        print(f"[DEBUG] anchor_feature min: {anchor_feature.min()}, max: {anchor_feature.max()}")

        if torch.isnan(anchor_feature).any():
            print("[ERROR] NaN detected in anchor_feature!")
            return torch.tensor(float('nan'), device=device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        print(f"[DEBUG] anchor_dot_contrast min: {anchor_dot_contrast.min()}, max: {anchor_dot_contrast.max()}")

        if torch.isnan(anchor_dot_contrast).any():
            print("[ERROR] NaN detected in anchor_dot_contrast!")
            return torch.tensor(float('nan'), device=device)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if torch.isnan(logits).any():
            print("[ERROR] NaN detected in logits!")
            return torch.tensor(float('nan'), device=device)

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        if torch.isnan(exp_logits).any():
            print("[ERROR] NaN detected in exp_logits!")
            return torch.tensor(float('nan'), device=device)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        if torch.isnan(log_prob).any():
            print("[ERROR] NaN detected in log_prob!")
            return torch.tensor(float('nan'), device=device)

        # compute mean of log-likelihood over positive pairs
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        if torch.isnan(mean_log_prob_pos).any():
            print("[ERROR] NaN detected in mean_log_prob_pos!")
            return torch.tensor(float('nan'), device=device)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        if torch.isnan(loss).any():
            print("[ERROR] NaN detected in loss!")
            return torch.tensor(float('nan'), device=device)

        return loss

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

import pdb

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorchvideo.losses.soft_target_cross_entropy import (
    SoftTargetCrossEntropyLoss,
)

class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean", *args, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long).cuda()
        loss = nn.CrossEntropyLoss(reduction=self.reduction).cuda()(
            inputs, targets
        )
        return loss


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean", *args, **kwargs):
        """
        Args:
            reduction (str): specifies reduction to apply to the output. It can be
                "mean" (default) or "none".
        """
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss

class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self, reduction='mean', ignore_index=-100, *args, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        self.ignore_index = ignore_index 

    def forward(self, input, target):

        if self.ignore_index >= 0:
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index
            ]
            
            return self.criterion(input[:, notice_index], target[:, notice_index])

        else:

            return self.criterion(input, target)

class MultipCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean", ignore_index=-100, *args, **kwargs):
        super(MultipCrossEntropyLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logsoftmax = nn.LogSoftmax(dim=1).to(input.device)

        if self.ignore_index >= 0:
            
            notice_index = [
                i for i in range(target.shape[-1]) if i != self.ignore_index
            ]
            
            output = torch.sum(
                -target[:, notice_index] * logsoftmax(input[:, notice_index]),
                dim=1,
            ) 

            if self.reduction == "mean":
                return torch.mean(output[target[:, self.ignore_index] != 1])
            elif self.reduction == "sum":
                return torch.sum(output[target[:, self.ignore_index] != 1])
            else:
                return output[target[:, self.ignore_index] != 1]
        else:
            
            output = torch.sum(-target * logsoftmax(input), dim=1)

            if self.reduction == "mean":
                
                return torch.mean(output)
            elif self.reduction == "sum":
                return torch.sum(output)
            else:
                return output

class EQLv2Loss(nn.Module):

    def __init__(self, gamma=12, mu=8, alpha=4.0, reduction='mean', ignore_index=None,
                 num_classes=3806, *args, **kwargs):
        super(EQLv2Loss, self).__init__()
        self.num_classes = num_classes - 1

        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        self.ignore_index = ignore_index

        self._pos_grad = None
        self._neg_grad = None
        self.pos_neg = None

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)

    def forward(self, input, target):
        self.n_i, self.n_c = input.size()
        # print(self.n_i, self.n_c)
        pos_w, neg_w = self.get_weight(input)

        weight = pos_w * target + neg_w * (1 - target)

        cls_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        cls_loss = torch.sum(cls_loss * weight) / self.n_i

        self.collect_grad(input.detach(), target.detach(), weight.detach())

        return cls_loss

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob # prob - target
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        if self.ignore_index is not None:
            grad[:, self.ignore_index] = 0
        pos_grad = torch.sum(grad * target * weight, dim=0)[:-1]
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)[:-1]

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self._pos_grad += pos_grad
        self._neg_grad += neg_grad
        self.pos_neg = self._pos_grad / (self._neg_grad + 1e-10)

    def get_weight(self, cls_score):
        if self._pos_grad is None:
            self._pos_grad = cls_score.new_zeros(self.num_classes)
            self._neg_grad = cls_score.new_zeros(self.num_classes)
            neg_w = cls_score.new_zeros((self.n_i, self.n_c))
            pos_w = cls_score.new_zeros((self.n_i, self.n_c))
        else:
            neg_w = torch.cat([self.map_func(self.pos_neg),cls_score.new_ones(1)])
            pos_w = 1 + self.alpha * (1 - neg_w)
            neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
            pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', ignore_index=-100, *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, input, target):
        return sigmoid_focal_loss(input, target, self.alpha, self.gamma, self.reduction)
    
class Pro2Loss(nn.Module):
    def __init__(self, cfg, reduction='none', ignore_index=-100):
        super().__init__()
        
        if cfg is not None:
            self.margin = getattr(cfg.MODEL, 'LOSS_MARGIN', 0.1)
            self.lambda_proc = getattr(cfg.MODEL, 'LOSS_LAMBDA', 0.1)
            self.num_classes = cfg.MODEL.NUM_CLASSES
        else:
            self.margin = 0.1
            self.lambda_proc = 0.1
            self.num_classes = 7
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

    def forward(self, output_dict, targets, boundary_mask=None):
        """
        output_dict: {'logits': (B, T, C), 'dynamic_z': (B, T, D), 'prototypes': (C, D)}
        targets: (B, T) Long Tensor (Indices) or (B, T, C)
        boundary_mask: (B, T) or None
        """
        logits = output_dict['logits']
        dynamic_z = output_dict['dynamic_z']
        prototypes = output_dict['prototypes']
        
        # Ensure targets match logits shape (B, T)
        if targets.dim() > 2: # If targets are one-hot (B, T, C), convert to indices
            targets = targets.argmax(dim=-1)
        
        # Reshape for CE Loss: (N, C) vs (N)
        B, T, C = logits.shape
        flat_logits = logits.reshape(-1, C)
        flat_targets = targets.reshape(-1)
        
        # 1. Boundary-Aware CE Loss
        ce_loss_raw = self.ce_loss(flat_logits, flat_targets).view(B, T)
        
        if boundary_mask is not None:
            weights = 1.0 + boundary_mask * 1.0 
            loss_cls = (ce_loss_raw * weights)
        else:
            loss_cls = ce_loss_raw

        # Apply reduction
        if self.reduction == 'mean':
            if self.ignore_index >= 0:
                valid_mask = flat_targets != self.ignore_index
                loss_cls = loss_cls.view(-1)[valid_mask].mean()
            else:
                loss_cls = loss_cls.mean()
        elif self.reduction == 'sum':
            loss_cls = loss_cls.sum()

        # 2. Procedural Contrastive Loss (Triplet-like)
        # Flatten features
        z = dynamic_z.reshape(-1, dynamic_z.shape[-1]) # (N, D)
        p = prototypes # (C, D)
        y = flat_targets # (N,)

        # Filter ignore_index for contrastive loss
        if self.ignore_index >= 0:
            valid_mask = y != self.ignore_index
            z = z[valid_mask]
            y = y[valid_mask]

        if z.shape[0] > 0:
            # Cosine Similarity
            sim = torch.matmul(z, p.t()) # (N_valid, C)
            
            # Pos Sim
            pos_mask = F.one_hot(y, num_classes=self.num_classes).bool()
            pos_sim = sim[pos_mask] # (N_valid,)
            
            # Neg Sim (Hard Negative Mining)
            neg_sim = sim.clone()
            neg_sim[pos_mask] = -float('inf')
            hard_neg_sim, _ = neg_sim.max(dim=1) # (N_valid,)
            
            loss_proc = F.relu(self.margin + hard_neg_sim - pos_sim).mean()
        else:
            loss_proc = torch.tensor(0.0, device=z.device)

        total_loss = loss_cls + self.lambda_proc * loss_proc
        
        # Return tuple to match expectation of unpacking in train loop
        return total_loss, {"cls": loss_cls, "proc": loss_proc}

@torch.jit.script
def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Taken from
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = 0.25.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


_LOSSES = {
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCELoss,
    # "bce_logit": nn.BCEWithLogitsLoss, 
    "bce_logit": BinaryCrossEntropyLoss,
    "soft_cross_entropy": partial(
        SoftTargetCrossEntropyLoss, normalize_targets=False
    ),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "multi_ce": MultipCrossEntropyLoss,
    "pro2": Pro2Loss,
}


        

def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]


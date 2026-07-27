# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS


def cpd_contrastive_loss(
    pred: Tensor,
    target: Tensor,
    labels: Tensor,
    teacher_labels: Tensor,
    temperature: float = 0.07,
) -> Tensor:
    """Core computation function for Class-Prototype Distillation Loss.

    Args:
        pred (Tensor): Student proposal features, shape [M, D].
        target (Tensor): Teacher class prototypes buffer, shape [K, D].
        labels (Tensor): Ground-truth labels for student proposals, shape [M].
        teacher_labels (Tensor): Corresponding labels for teacher prototypes, shape [K].
        temperature (float): Temperature hyperparameter for scaling logits. 
            Defaults to 0.07.

    Returns:
        Tensor: Element-wise loss tensor before reduction, shape [M].
    """
    # Normalization
    student_norm = F.normalize(pred, p=2, dim=1)
    teacher_norm = F.normalize(target, p=2, dim=1)

    # 2. Similarity
    logits = torch.matmul(student_norm, teacher_norm.t()) / temperature  # [M, K]

    # 3. Positive Mask
    labels = labels.view(-1, 1)                  # [M, 1]
    teacher_labels = teacher_labels.view(1, -1)  # [1, K]
    pos_mask = (labels == teacher_labels).float()  # [M, K]


    # 4. Log-Softmax
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits_stable = logits - logits_max.detach()

    log_prob = logits_stable - torch.logsumexp(logits_stable, dim=1, keepdim=True)
    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    #5. loss
    loss = -mean_log_prob_pos
    invalid_mask = (pos_count == 0)
    loss[invalid_mask] = 0.0

    return loss

@MODELS.register_module()
class CPDLoss(nn.Module):
    """Class-Prototype Distillation (CPD) Contrastive Loss Module.

    This loss enforces alignment between high-quality student proposal embeddings 
    and the dynamically updated historical teacher class prototypes.

    Args:
        temperature (float): Scaling factor for logit evaluation. Defaults to 0.07.
        reduction (str): The method used to reduce the loss. 
            Options are "none", "mean" and "sum". Defaults to 'mean'.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(self,
                 temperature: float = 0.07,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred: Tensor,
                target: Tensor,
                labels: Tensor,
                teacher_labels: Optional[Tensor] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward pass for CPDLoss.

        Args:
            pred (Tensor): Student representations, shape [M, D].
            target (Tensor): Teacher representations, shape [K, D].
            labels (Tensor): Target labels for student instances, shape [M].
            teacher_labels (Tensor, optional): Identity vectors for the target buffer. 
                If None, auto-generates sequential index markers. Defaults to None.

        Returns:
            Tensor: Reduced loss scalar.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')

        if pred.numel() == 0:
            return pred.new_zeros([])
        
        if teacher_labels is None:
            teacher_labels = torch.arange(
                target.size(0), 
                device=target.device
            )

        loss_unreduced = cpd_contrastive_loss(
            pred,
            target ,
            labels,
            teacher_labels,
            temperature=self.temperature
        )

        if self.reduction == 'mean':
            loss = loss_unreduced.mean()
        elif self.reduction == 'sum':
            loss = loss_unreduced.sum()
        else:
            loss = loss_unreduced

        return loss
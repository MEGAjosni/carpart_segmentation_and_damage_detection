# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:08:16 2022

@author: Marcus
"""

from typing import List

import torch
import torch.nn.functional as F
from pytorch_toolbelt.utils.torch_utils import to_tensor
from torch import Tensor
from torch.nn.modules.loss import _Loss

from pytorch_toolbelt.losses.functional import soft_dice_score
#%%
__all__ = ["DiceLoss"]



class DiceLossJAM(_Loss):
    """
    Implementation of Dice loss for image segmentation task.
    It supports binary, multiclass and multilabel cases
    """

    def __init__(
        self,
        classes: List[int] = None,
        log_loss=False,
        from_logits=False,
        smooth: float = 0.0,
        ignore_index=None,
        weights = None,
        eps=1e-7,
    ):
        """

        :param classes: Optional list of classes that contribute in loss computation;
        By default, all channels are included.
        :param log_loss: If True, loss computed as `-log(jaccard)`; otherwise `1 - jaccard`
        :param from_logits: If True assumes input is raw logits
        :param smooth:
        :param ignore_index: Label that indicates ignored pixels (does not contribute to loss)
        :param eps: Small epsilon for numerical stability
        """
        super(DiceLossJAM, self).__init__()

        self.weights = weights
        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index
        self.log_loss = log_loss

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        :param y_pred: NxCxHxW
        :param y_true: NxHxW
        :return: scalar
        """
        assert y_true.shape == y_pred.shape

        bs,num_classes,_,_ = y_true.shape
        
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred.double()).exp()
        


            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask
    
        y_true = y_true.view(bs, num_classes, -1)
        y_pred = y_pred.view(bs, num_classes, -1)
        dims = 2
        scores = soft_dice_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.weights is not None:
            scores *=weights

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores



        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        return loss.mean()
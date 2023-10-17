#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from .box import GIOU

from .configurator import Configurator


class GIOULoss(nn.Module):
    def __init__(self, reduction = 'mean'):
        super().__init__()

        self.reduce_fn = {
            'none' : lambda x: x,
            'mean' : torch.mean,
            'sum'  : torch.sum,
        }[reduction]


    def forward(self, source_boxes, target_boxes):
        giou      = GIOU.calculate_giou(source_boxes, target_boxes)
        loss_giou = 1 - giou

        loss_giou_reduced = self.reduce_fn(loss_giou)

        return loss_giou_reduced




class BoxLoss(nn.Module):
    def __init__(self, lambda_giou = 10, lambda_l1 = 1, reduction = 'mean'):
        super().__init__()

        self.lambda_giou = lambda_giou
        self.lambda_l1   = lambda_l1

        self.GIOULoss = GIOULoss (reduction = reduction)
        self.L1Loss   = nn.L1Loss(reduction = reduction)


    def forward(self, source_boxes, target_boxes):
        loss_giou = self.GIOULoss(source_boxes, target_boxes)
        loss_l1   = self.L1Loss  (source_boxes, target_boxes).mean(dim = -1)

        return self.lambda_giou * loss_giou + self.lambda_l1 * loss_l1




class HungarianLoss(nn.Module):

    @staticmethod
    def get_default_config():
        CONFIG = Configurator()
        with CONFIG.enable_auto_create():
            CONFIG.BOXLOSS.LAMBDA_GIOU = 10.0
            CONFIG.BOXLOSS.LAMBDA_L1   = 1.0

            CONFIG.ENABLES_TRACK_VAR = False

        return CONFIG


    def __init__(self, config = None):
        super().__init__()

        self.config = HungarianLoss.get_default_config() if config is None else config

        self.BoxLoss = BoxLoss(self.config.BOXLOSS.LAMBDA_GIOU,
                               self.config.BOXLOSS.LAMBDA_L1,
                               reduction = 'none')

        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction = 'none')

        self.track_var_dict = {} if self.config.ENABLES_TRACK_VAR else None


    def track_var(self, k, v):
        if self.config.ENABLES_TRACK_VAR:
            if not k in self.track_var_dict: self.track_var_dict[k] = []
            self.track_var_dict[k].append(v)


    def forward(self, source_class_logits, target_classes, source_boxes, target_boxes):
        """
        Arguments:
            source_class_logits: Tensor with shape of (B, Ns, LOGIT_DIM).
            target_classes     : Tensor with shape of (B, Nt,          ).
            source_boxes       : Tensor with shape of (B, Ns, 4        ).
            target_boxes       : Tensor with shape of (B, Nt, 4        ).

        Returns:
            source_idx: Tensor with shape of (Nm).
            target_idx: Tensor with shape of (Nm).
        """
        # Retrieve metadata...
        B, Ns, _ = source_class_logits.shape
        _, Nt    = target_classes.shape

        # Calculate Hungarian loss...
        loss_hungarian_list = []
        for batch_idx in range(B):
            # Calculate pairwise cross entropy loss...
            extend_source_class_logits = source_class_logits[batch_idx,    :, None,    :].expand(-1, Nt, -1)
            extend_target_classes      = target_classes     [batch_idx, None,    :, None].expand(Ns, -1, -1)

            flat_extend_source_class_logits = extend_source_class_logits.flatten(start_dim = 0, end_dim = -2)    # ...Not including feature dimension (dim=-1)
            flat_extend_target_classes      = extend_target_classes.flatten     (start_dim = 0, end_dim = -1)

            loss_cross_entropy = self.CrossEntropyLoss(flat_extend_source_class_logits,
                                                       flat_extend_target_classes,).view(Ns, Nt)

            # Calculate pairwise box loss...
            extend_source_boxes = source_boxes[batch_idx,    :, None].expand(-1, Nt, -1)
            extend_target_boxes = target_boxes[batch_idx, None,    :].expand(Ns, -1, -1)

            flat_extend_source_boxes = extend_source_boxes.flatten(start_dim = 0, end_dim = -2)
            flat_extend_target_boxes = extend_target_boxes.flatten(start_dim = 0, end_dim = -2)

            loss_box = self.BoxLoss(flat_extend_source_boxes, flat_extend_target_boxes).view(Ns, Nt)

            # Figure out bipartite matching...
            loss_matrix      = loss_cross_entropy + loss_box
            row_idx, col_idx = linear_sum_assignment(loss_matrix)
            loss_hungarian   = loss_matrix[row_idx, col_idx].mean()

            loss_hungarian_list.append(loss_hungarian)

            self.track_var(k = 'lsa', v = (batch_idx, (row_idx, col_idx)))

        return torch.stack(loss_hungarian_list)

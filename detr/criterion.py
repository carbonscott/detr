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
            CONFIG.REDUCTION           = 'mean'

        return CONFIG


    def __init__(self, config = None):
        super().__init__()

        self.config = HungarianLoss.get_default_config() if config is None else config

        self.BoxLoss = BoxLoss(self.config.BOXLOSS.LAMBDA_GIOU,
                               self.config.BOXLOSS.LAMBDA_L1,
                               self.config.REDUCTION)

        self.CrossEntropyLoss = nn.CrossEntropyLoss(reduction = self.config.REDUCTION)



    def calculate_pairwis_loss(self, source_class_logits, target_classes, source_boxes, target_boxes):
        """
        Arguments:
            source_class_logits: Tensor with shape of (Ns, LOGIT_DIM).
            target_classes     : Tensor with shape of (Nt, ).
            source_boxes       : Tensor with shape of (Ns, 4).
            target_boxes       : yensor with shape of (Nt, 4).

        Returns:
            source_idx: Tensor with shape of (Nm).
            target_idx: Tensor with shape of (Nm).
        """
        ## # ___/ DRAFT \___
        ## B, Ns, Nt = 10, 12, 12
        ## 
        ## source_class_logits = torch.randn(B, Ns, 2) 
        ## target_classes = torch.randint(0, 2, (B, Nt))
        ## 
        ## source_boxes = torch.randn(B, Ns, 4)
        ## target_boxes = torch.randn(B, Nt, 4)
        ## boxloss_fn = BoxLoss(reduction = 'none')
        ## giouloss_fn = GIOULoss(reduction = 'none')
        ## for batch_idx in range(B):
        ##     extend_source_class_logits = source_class_logits[batch_idx,     :, None,     :].expand(-1, Ns, -1)
        ##     extend_target_classes      = target_classes     [batch_idx, None,      :, None].expand(Ns, -1, -1)
        ##
        ##     flat_extend_source_class_logits   = extend_source_class_logits.flatten(start_dim = 0, end_dim = -2)
        ##     flat_extend_target_classes = extend_target_classes.flatten(start_dim = 0, end_dim = -1)
        ##
        ##     loss_cross_entropy = F.cross_entropy(flat_extend_source_class_logits, 
        ##                                          flat_extend_target_classes, reduction = 'none').view(Ns, Ns)
        ##
        ##     extend_source_boxes = source_boxes[batch_idx, :, None].expand( -1, Ns, -1)
        ##     extend_target_boxes = target_boxes[batch_idx, None, :].expand(Ns, -1, -1)
        ##
        ##     flat_extend_source_boxes = extend_source_boxes.flatten(start_dim = 0, end_dim = -2)
        ##     flat_extend_target_boxes = extend_target_boxes.flatten(start_dim = 0, end_dim = -2)
        ##
        ##     loss_l1 = F.l1_loss(flat_extend_source_boxes, flat_extend_target_boxes, reduction = 'none').view(Ns, Ns, -1).mean(dim = -1)
        ##     loss_giou = giouloss_fn(flat_extend_source_boxes, flat_extend_target_boxes).view(Ns, Ns)
        ##
        ##     break

        # Calculate pairwise cross_entropy...
        # (Ns, LOGIT_DIM) -> (Ns, 1, LOGIT_DIM)
        # (Nt,          ) -> (1, Nt, 1)
        # output: (Ns, Nt)
        loss_cross_entropy = self.CrossEntropyLoss(source_class_logits[:, None, :], target_classes[None, :, None])

        # Calculate pairwise box loss...
        # (Ns, 4) -> (Ns, 1, 4)
        # (Nt, 4) -> (1, Nt, 4)
        # (Ns, Nt)
        loss_box = self.BoxLoss(source_boxes[:, None], target_boxes[None, ])

        return loss_cross_entropy + loss_box


    @staticmethod
    def assign_matching_idx(self, source, target):
        """
        Arguments:
            source: Tensor with shape of (B, Ns, DETR_OUTPUT_DIM).
            target: Tensor with shape of (B, Nt, DETR_OUTPUT_DIM).

        Returns:
            source_idx: Tensor with shape of (B, Nm).
            target_idx: Tensor with shape of (B, Nm).
        """
        source_class_logits = source[...,   :-4]    # (B, Ns, LOGIT_DIM)
        source_boxes        = source[..., -4:  ]    # (B, Ns, 4)

        target_class_logits = target[...,   :-4]    # (B, Nt, LOGIT_DIM)
        target_boxes        = target[..., -4:  ]    # (B, Nt, 4)

        target_class = torch.argmax(target_class_logits, dim = -1)    # ...Feature dim; (B, Nt)


        ## linear_sum_assignment()

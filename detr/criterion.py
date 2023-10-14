#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, lambda_giou, lambda_l1, reduction = 'mean'):
        super().__init__()

        self.lambda_giou = lambda_giou
        self.lambda_l1   = lambda_l1

        self.GIOULoss = GIOULoss (reduction = reduction)
        self.L1Loss   = nn.L1Loss(reduction = reduction)

    def forward(self, source_boxes, target_boxes):
        loss_giou = self.GIOULoss(source_boxes, target_boxes)
        loss_l1   = self.L1Loss  (source_boxes, target_boxes)

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


    def forward(self, source_class_logits, target_classes, source_boxes, target_boxes):
        loss_cross_entropy = self.CrossEntropyLoss(source_class_logits, target_classes)
        loss_box           = self.BoxLoss(source_boxes, target_boxes)

        return loss_cross_entropy + loss_box

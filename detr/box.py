#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GIOU(nn.Module):
    @staticmethod
    def yxyx_to_yxhw(yxyx):
        y_min, x_min, y_max, x_max = yxyx

        h = y_max - y_min
        w = x_max - x_min

        y_c = (y_max + y_min)/2
        x_c = (x_max + x_min)/2

        return y_c, x_c, h, w


    @staticmethod
    def yxhw_to_yxyx(yxhw):
        y_c, x_c, h, w = yxhw

        y_min = y_c - h/2
        y_max = y_c + h/2
        x_min = x_c - w/2
        x_max = x_c + w/2

        return y_min, x_min, y_max, x_max


    @staticmethod
    def get_pairwise_superbox_coordinates(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bs, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bt, 4)

        Returns:
            super_boxes: shape of (Bs, Bt, 4)
        """
        min_coords = torch.min(source_boxes[:, None,  :2], target_boxes[None, :,  :2])    # (Bs, Bt, 4)
        max_coords = torch.max(source_boxes[:, None, 2: ], target_boxes[None, :, 2: ])    # (Bs, Bt, 4)

        super_boxes = torch.cat([min_coords, max_coords], dim = -1)

        return super_boxes


    @staticmethod
    def calculate_pairwise_superbox_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bs, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bt, 4)

        Returns:
            area: shape of (Bs, Bt, 4)
        """
        super_boxes = GIOU.get_pairwise_superbox_coordinates(source_boxes, target_boxes)

        y_min, x_min, y_max, x_max = super_boxes.permute(2, 0, 1)    # (Bs, Bt, 4) -> (4, Bs, Bt)

        h = y_max - y_min
        w = x_max - x_min

        area = h * w

        return area


    @staticmethod
    def get_pairwise_intersection_coordinates(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...]
            target_boxes: [(y_min, x_min, y_max, x_max), ...]

        Returns:
            super_boxes: shape of (Bs, Bt, 4)
        """
        min_coords = torch.max(source_boxes[:, None,  :2], target_boxes[None, :,  :2])    # (Bs, Bt, 4)
        max_coords = torch.min(source_boxes[:, None, 2: ], target_boxes[None, :, 2: ])    # (Bs, Bt, 4)

        intersection_boxes = torch.cat([min_coords, max_coords], dim = -1)

        return intersection_boxes


    @staticmethod
    def calculate_pairwise_intersection_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bs, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bt, 4)

        Returns:
            area: shape of (Bs, Bt, 4)
        """
        intersection_boxes = GIOU.get_pairwise_intersection_coordinates(source_boxes, target_boxes)

        y_min, x_min, y_max, x_max = intersection_boxes.permute(2, 0, 1)    # (Bs, Bt, 4) -> (4, Bs, Bt)

        h = y_max - y_min
        w = x_max - x_min

        area = h * w

        return area


    def __init__(self):
        super().__init__()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class GIOU(nn.Module):
    @staticmethod
    def yxyx_to_yxhw(yxyx):
        instance_axes = range(yxyx.ndim - 1)
        y_min, x_min, y_max, x_max = yxyx.permute(-1, *instance_axes)

        h = y_max - y_min
        w = x_max - x_min

        y_c = (y_max + y_min)/2
        x_c = (x_max + x_min)/2

        yxhw = torch.stack([y_c, x_c, h, w])
        instance_axes = range(1, yxhw.ndim)

        return yxhw.permute(*instance_axes, 0)


    @staticmethod
    def yxhw_to_yxyx(yxhw):
        instance_axes = range(yxhw.ndim - 1)
        y_c, x_c, h, w = yxhw.permute(-1, *instance_axes)

        y_min = y_c - h/2
        y_max = y_c + h/2
        x_min = x_c - w/2
        x_max = x_c + w/2

        yxyx = torch.stack([y_min, x_min, y_max, x_max])
        instance_axes = range(1, yxyx.ndim)

        return yxyx.permute(*instance_axes, 0)


    @staticmethod
    def get_superbox_coordinates(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)

        Returns:
            super_boxes: shape of (B, 4)
        """
        min_coords = torch.min(source_boxes[:,  :2], target_boxes[:,  :2])    # (B, 2)
        max_coords = torch.max(source_boxes[:, 2: ], target_boxes[:, 2: ])    # (B, 2)

        super_boxes = torch.cat([min_coords, max_coords], dim = -1)    # (B, 4)

        return super_boxes


    @staticmethod
    def calculate_superbox_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)

        Returns:
            area: shape of (B,)
        """
        super_boxes = GIOU.get_superbox_coordinates(source_boxes, target_boxes)    # (B, 4)

        area = GIOU.calculate_area(super_boxes)

        return area


    @staticmethod
    def get_intersection_coordinates(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...]
            target_boxes: [(y_min, x_min, y_max, x_max), ...]

        Returns:
            super_boxes: shape of (B, 4)
        """
        min_coords = torch.max(source_boxes[:,  :2], target_boxes[:,  :2])    # (B, 2)
        max_coords = torch.min(source_boxes[:, 2: ], target_boxes[:, 2: ])    # (B, 2)

        intersection_boxes = torch.cat([min_coords, max_coords], dim = -1)    # (B, 4)

        return intersection_boxes


    @staticmethod
    def calculate_intersection_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)

        Returns:
            area: shape of (B, )
        """
        intersection_boxes = GIOU.get_intersection_coordinates(source_boxes, target_boxes)
        area               = GIOU.calculate_area(intersection_boxes)

        return area


    @staticmethod
    def calculate_union_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)

        Returns:
            area: shape of (B, )
        """
        area_source_boxes = GIOU.calculate_area(source_boxes)
        area_target_boxes = GIOU.calculate_area(target_boxes)

        # Calculate the area of the intersection boxes...
        area_intersection_boxes = GIOU.calculate_intersection_area(source_boxes, target_boxes)

        # Calculate the area of the union (A + B - Intersection(A, B))...
        area_union_boxes = area_source_boxes + area_target_boxes - area_intersection_boxes

        return area_union_boxes


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
            area: shape of (Bs, Bt)
        """
        super_boxes = GIOU.get_pairwise_superbox_coordinates(source_boxes, target_boxes)
        area        = GIOU.calculate_area(super_boxes)

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
            area: shape of (Bs, Bt)
        """
        intersection_boxes = GIOU.get_pairwise_intersection_coordinates(source_boxes, target_boxes)
        area               = GIOU.calculate_area(intersection_boxes)

        return area


    @staticmethod
    def calculate_pairwise_union_area(source_boxes, target_boxes):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bs, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bt, 4)

        Returns:
            area: shape of (Bs, Bt)
        """
        area_source_boxes = GIOU.calculate_area(source_boxes)
        area_target_boxes = GIOU.calculate_area(target_boxes)

        # Calculate the area of the intersection boxes...
        area_intersection_boxes = GIOU.calculate_pairwise_intersection_area(source_boxes, target_boxes)

        # Calculate the area of the union (A + B - Intersection(A, B))...
        area_union_boxes = area_source_boxes[:, None] + area_target_boxes[None, :] - area_intersection_boxes

        return area_union_boxes


    @staticmethod
    def calculate_area(boxes):
        """
        Arguments:
        boxes: Tensor with shape (..., 4).
               - The last dimension represents coordinates: (y_min, x_min, y_max, x_max).
               - '...' denotes any number of preceding dimensions, each containing instances.

        Returns:
            area: shape of (...,)
        """
        instance_axes = range(boxes.ndim - 1)
        y_min, x_min, y_max, x_max = boxes.permute(-1, *instance_axes)

        h = y_max - y_min
        w = x_max - x_min

        area_boxes = h * w

        return area_boxes


    @staticmethod
    def calculate_giou(source_boxes, target_boxes, returns_intermediate = False):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (B, 4)

        Returns:
            giou: shape of (B, )
        """
        intersection_area = GIOU.calculate_intersection_area(source_boxes, target_boxes)
        union_area        = GIOU.calculate_union_area       (source_boxes, target_boxes)
        superbox_area     = GIOU.calculate_superbox_area    (source_boxes, target_boxes)

        margin_area = superbox_area - union_area
        mos         = margin_area / superbox_area    # ...Margin over superbox
        iou         = intersection_area / union_area
        giou        = iou - mos

        return giou if not returns_intermediate else \
               (giou, intersection_area, union_area, superbox_area)


    @staticmethod
    def calculate_pairwise_giou(source_boxes, target_boxes, returns_intermediate = False):
        """
        Arguments:
            source_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bs, 4)
            target_boxes: [(y_min, x_min, y_max, x_max), ...], shape of (Bt, 4)

        Returns:
            giou: shape of (Bs, Bt)
        """
        intersection_area = GIOU.calculate_pairwise_intersection_area(source_boxes, target_boxes)
        union_area        = GIOU.calculate_pairwise_union_area       (source_boxes, target_boxes)
        superbox_area     = GIOU.calculate_pairwise_superbox_area    (source_boxes, target_boxes)

        margin_area = superbox_area - union_area
        mos         = margin_area / superbox_area    # ...Margin over superbox
        iou         = intersection_area / union_area
        giou        = iou - mos

        return giou if not returns_intermediate else \
               (giou, intersection_area, union_area, superbox_area)


    def __init__(self):
        super().__init__()

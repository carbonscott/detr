import torch
import torch.nn            as nn
import torch.nn.functional as F

from detr.box import GIOU

def test_superbox():
    Bs, Bt, N = 10, 12, 4

    source_boxes = torch.randn(Bs, N)
    target_boxes = torch.randn(Bt, N)

    # Calculate manually...
    super_y_min = torch.min(source_boxes[:, None, 0], target_boxes[None, :, 0])
    super_x_min = torch.min(source_boxes[:, None, 1], target_boxes[None, :, 1])
    super_y_max = torch.max(source_boxes[:, None, 2], target_boxes[None, :, 2])
    super_x_max = torch.max(source_boxes[:, None, 3], target_boxes[None, :, 3])
    super_boxes_manual = torch.cat([ i[:, :, None] for i in (super_y_min,
                                                             super_x_min,
                                                             super_y_max,
                                                             super_x_max)],  dim = -1)

    # Calculate by the package...
    super_boxes = GIOU.get_pairwise_superbox_coordinates(source_boxes, target_boxes)

    assert torch.equal(super_boxes, super_boxes_manual)


def test_intersection():
    Bs, Bt, N = 10, 12, 4

    source_boxes = torch.randn(Bs, N)
    target_boxes = torch.randn(Bt, N)

    # Calculate manually...
    intersection_y_min = torch.max(source_boxes[:, None, 0], target_boxes[None, :, 0])
    intersection_x_min = torch.max(source_boxes[:, None, 1], target_boxes[None, :, 1])
    intersection_y_max = torch.min(source_boxes[:, None, 2], target_boxes[None, :, 2])
    intersection_x_max = torch.min(source_boxes[:, None, 3], target_boxes[None, :, 3])
    intersection_boxes_manual = torch.cat([ i[:, :, None,] for i in (intersection_y_min,
                                                                     intersection_x_min,
                                                                     intersection_y_max,
                                                                     intersection_x_max)],  dim = -1)

    # Calculate by the package...
    intersection_boxes = GIOU.get_pairwise_intersection_coordinates(source_boxes, target_boxes)

    assert torch.equal(intersection_boxes, intersection_boxes_manual)


def test_superbox_area():
    Bs, Bt, N = 10, 12, 4

    source_boxes = torch.randn(Bs, N)
    target_boxes = torch.randn(Bt, N)

    # Calculate manually...
    super_y_min = torch.min(source_boxes[:, None, 0], target_boxes[None, :, 0])
    super_x_min = torch.min(source_boxes[:, None, 1], target_boxes[None, :, 1])
    super_y_max = torch.max(source_boxes[:, None, 2], target_boxes[None, :, 2])
    super_x_max = torch.max(source_boxes[:, None, 3], target_boxes[None, :, 3])

    h_manual = super_y_max - super_y_min
    w_manual = super_x_max - super_x_min
    area_manual = h_manual * w_manual

    # Calculate by the package...
    area = GIOU.calculate_pairwise_superbox_area(source_boxes, target_boxes)

    assert area.shape == area_manual.shape, f"Expected same shape, but got area.shape = {area.shape}; area_manual.shape = {area_manual.shape}."
    assert torch.equal(area, area_manual), f"Expected same value, but got area = {area}; area_manual = {area_manual}."

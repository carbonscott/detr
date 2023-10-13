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
    assert torch.equal(area, area_manual) , f"Expected same value, but got area = {area}; area_manual = {area_manual}."


def test_intersection_area():
    Bs, Bt, N = 10, 12, 4

    source_boxes = torch.randn(Bs, N)
    target_boxes = torch.randn(Bt, N)

    # Calculate manually...
    intersection_y_min = torch.max(source_boxes[:, None, 0], target_boxes[None, :, 0])
    intersection_x_min = torch.max(source_boxes[:, None, 1], target_boxes[None, :, 1])
    intersection_y_max = torch.min(source_boxes[:, None, 2], target_boxes[None, :, 2])
    intersection_x_max = torch.min(source_boxes[:, None, 3], target_boxes[None, :, 3])

    h_manual = intersection_y_max - intersection_y_min
    w_manual = intersection_x_max - intersection_x_min
    area_manual = h_manual * w_manual

    # Calculate by the package...
    area = GIOU.calculate_pairwise_intersection_area(source_boxes, target_boxes)

    assert area.shape == area_manual.shape, f"Expected same shape, but got area.shape = {area.shape}; area_manual.shape = {area_manual.shape}."
    assert torch.equal(area, area_manual) , f"Expected same value, but got area = {area}; area_manual = {area_manual}."


def test_union_area():
    source_boxes = torch.tensor([0, 0, 10, 10])[None,]

    h_offset, w_offset = 8, 8
    target_boxes = source_boxes + torch.tensor([h_offset, h_offset, w_offset, w_offset])

    area_gt = 196

    # Calculate by the package...
    area = GIOU.calculate_pairwise_union_area(source_boxes, target_boxes)

    assert area.item() == area_gt , f"Expected same value, but got area = {area.item()}."


def test_giou_area():
    source_boxes = torch.tensor([0, 0, 10, 10])[None,]

    h_offset, w_offset = 8, 8
    target_boxes = source_boxes + torch.tensor([h_offset, h_offset, w_offset, w_offset])

    intersection_area_manual = torch.tensor(4)
    union_area_manual        = torch.tensor(196)
    superbox_area_manual     = torch.tensor(18*18)

    margin_area_manual = superbox_area_manual - union_area_manual
    mos                = margin_area_manual / superbox_area_manual
    iou                = intersection_area_manual / union_area_manual
    giou_gt            = 1 - (iou - mos)

    # Calculate by the package...
    giou = GIOU.calculate_GIOU(source_boxes, target_boxes, returns_intermediate = False)
    giou, intersection_area, union_area, superbox_area = GIOU.calculate_GIOU(source_boxes, target_boxes, returns_intermediate = True)

    assert intersection_area.item() == intersection_area_manual.item()
    assert union_area.item()        == union_area_manual.item()
    assert superbox_area.item()     == superbox_area_manual.item()
    assert giou.item()              == giou_gt.item()

import torch
from math import sqrt
from utils import *
import torchvision.transforms.functional as F


def find_intersection(set_1, set_2):
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)

def cx_to_xmin(tens):
    return [tens[0]-tens[2]/2.0,tens[1]-tens[3]/2.0,tens[0]+tens[2]/2.0,tens[1]+tens[3]/2.0]

fmap_dims = {'conv4_3': 28,
             'conv7': 14,
             'conv8_2': 14,
             'conv9_2': 14,
             'conv10_2': 14,
             'conv11_2': 14}

obj_scales = {'conv4_3': 0.3,
              'conv7': 0.2,
              'conv8_2': 0.2,
              'conv9_2': 0.2,
              'conv10_2': 0.1,
              'conv11_2': 0.1}

aspect_ratios = {'conv4_3': [1., 2., 0.5, 3.],
                 'conv7': [1., 2., 0.5, 3.],
                 'conv8_2': [1., 2., 0.5, 3.],
                 'conv9_2': [1., 2., 0.5, 3.],
                 'conv10_2': [1., 2., 0.5, 3.],
                 'conv11_2': [1., 2., 0.5, 3.]}

fmaps = list(fmap_dims.keys())
prior_boxes = []
for k, fmap in enumerate(fmaps):
    for i in range(fmap_dims[fmap]):
        for j in range(fmap_dims[fmap]):
            cx = (j + 0.5) / fmap_dims[fmap]
            cy = (i + 0.5) / fmap_dims[fmap]

            for ratio in aspect_ratios[fmap]:
                prior_boxes.append(cx_to_xmin([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)]))

prior_boxes = torch.FloatTensor(prior_boxes)  # (8732, 4)
prior_boxes.clamp_(0, 1)  # (8732, 4)


def get_iou(bb1, bb2):
    bb1 = {'x1': bb1[0] - bb1[2] / 2, 'x2': bb1[0] + bb1[2] / 2, 'y1': bb1[1] - bb1[3] / 2,
           'y2': bb1[1] + bb1[3] / 2}
    bb2 = {'x1': bb2[0] - bb2[2] / 2, 'x2': bb2[0] + bb2[2] / 2, 'y1': bb2[1] - bb2[3] / 2,
           'y2': bb2[1] + bb2[3] / 2}
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# def prior_to_box(prior):
#
#     return box

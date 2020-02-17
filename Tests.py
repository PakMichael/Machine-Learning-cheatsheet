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
                prior_boxes.append([cx, cy, obj_scales[fmap] * sqrt(ratio), obj_scales[fmap] / sqrt(ratio)])

prior_boxes = torch.FloatTensor(prior_boxes)  # (8732, 4)
prior_boxes.clamp_(0, 1)  # (8732, 4)


# def prior_to_box(prior):
#
#     return box

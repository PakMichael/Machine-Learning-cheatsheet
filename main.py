from PretrainedAdaptable import VGG16
from Appendage import Appendage
from Locator import Locator
from VOCdataset import VOCdataset
from utils import *
import torchvision.transforms.functional as F
import torch
from Tests import *

train = VOCdataset('VOC2012')
validation = VOCdataset('VOC2007')

print(train[0][1])
print(prior_boxes[0])
show_img_obj_scaled(F.to_pil_image(train[0][0]), prior_boxes[3136:3145])

overl = find_jaccard_overlap(prior_boxes, torch.Tensor(train[0][1]))
print(torch.max(overl, dim=0))
#
#
# img = image.unsqueeze(0)  # torch.Size([1, 3, 224, 224])
#
# model = VGG16()
# appendage = Appendage()
# locator = Locator()
#
# res = model(img)
# conv4_3, conv7 = res
#
# app_res = appendage(conv7)
# conv8_2, conv9_2, conv10_2, conv11_2 = app_res
#
# loc_res = locator(conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2)
#
# print(conv4_3.shape, conv7.shape, conv8_2.shape, conv9_2.shape, conv10_2.shape, conv11_2.shape)
#
# print(loc_res.shape)

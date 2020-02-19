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
print(prior_boxes[0].numpy())

#overlaps=torch.Tensor([get_iou(train[0][1][0],prior.numpy()) for prior in prior_boxes])

#print((overlaps>0.4).nonzero())
boxxx= [1060, 1064, 1068, 1072, 1172, 1174, 1176, 1178, 1180, 1182, 1184, 1186, 1282, 1284, 1286, 1288, 1290, 1292, 1294, 1296, 1298, 1302, 1394, 1396, 1398, 1400, 1402, 1404, 1406, 1408, 1410, 1414, 1506, 1508, 1510, 1512, 1514, 1516, 1518, 1520, 1522, 1526, 1620, 1622, 1624, 1626, 1628, 1630, 1632, 1634, 1638, 1732, 1736, 1740, 1744, 1848, 1852]
show_img_obj_scaled(F.to_pil_image(train[0][0]), [prior_boxes[a] for a in boxxx] )

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

from PretrainedAdaptable import VGG16
from Appendage import Appendage
from Locator import Locator
from VOCdataset import VOCdataset
from utils import *


train = VOCdataset('VOC2012')
validation = VOCdataset('VOC2007')

image, objects = train[5]


# for index, val in enumerate(train):
#     if index > 2: break
#     show_item(val)

# img = getImage('D:/2.jpg')
# orig_img = img
#
# img = transform(img).unsqueeze(0)  # torch.Size([1, 3, 224, 224])
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
# print(loc_res.shape)

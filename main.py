from PretrainedAdaptable import VGG16
from Appendage import Appendage
from Locator import Locator
from utils import *

img = getImage('D:/2.jpg')
orig_img = img

img = transform(img).unsqueeze(0)  # torch.Size([1, 3, 224, 224])

model = VGG16()
appendage = Appendage()
locator = Locator()

res = model(img)
conv4_3, conv7 = res

app_res = appendage(conv7)
conv8_2, conv9_2, conv10_2, conv11_2 = app_res

loc_res = locator(conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2)

print(loc_res[0][10])

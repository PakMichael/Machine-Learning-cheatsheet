from torch.nn import Module

from PretrainedAdaptable import VGG16
from Appendage import Appendage
from Locator import Locator


class FullModel(Module):
    def __init__(self):
        super(FullModel, self).__init__()

        self.vgg16 = VGG16()
        self.appendage = Appendage()
        self.locator = Locator()

    def forward(self, image):
        conv4_3, conv7 = self.vgg16(image)
        print(conv4_3.shape)
        print(conv7.shape)
        conv8_2, conv9_2, conv10_2, conv11_2 = self.appendage(conv7)
        print(conv8_2.shape)
        print(conv9_2.shape)
        print(conv10_2.shape)
        print(conv11_2.shape)
        loc_res = self.locator(conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2)
        print(loc_res.shape)


        return loc_res
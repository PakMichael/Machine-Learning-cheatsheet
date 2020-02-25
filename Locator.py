import torch.nn as nn
import torch


class Locator(nn.Module):
    def __init__(self):
        super().__init__()
        self.loc4_3 = nn.Conv2d(512, 16, kernel_size=3, padding=1)
        self.loc7 = nn.Conv2d(4096, 16, kernel_size=3, padding=1)
        self.loc8_2 = nn.Conv2d(2048, 16, kernel_size=3, padding=1)
        self.loc9_2 = nn.Conv2d(1024, 16, kernel_size=3, padding=1)
        self.loc10_2 = nn.Conv2d(512, 16, kernel_size=3, padding=1)
        self.loc11_2 = nn.Conv2d(256, 16, kernel_size=3, padding=1)

        self.init_conv2d()

    def init_conv2d(self):
        """
        Initialize convolution parameters.
        """
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def featuremapToLoc(self, conv, featuremap):
        batch_size = featuremap.size(0)

        loc = conv(featuremap)
        loc = loc.permute(0, 2, 3, 1).contiguous()
        return loc.view(batch_size, -1, 4)

    def forward(self, conv4_3, conv7, conv8_2, conv9_2, conv10_2, conv11_2):
        loc_conv4_3 = self.featuremapToLoc(self.loc4_3, conv4_3)
        loc_conv7 = self.featuremapToLoc(self.loc7, conv7)
        loc_conv8_2 = self.featuremapToLoc(self.loc8_2, conv8_2)
        loc_conv9_2 = self.featuremapToLoc(self.loc9_2, conv9_2)
        loc_conv10_2 = self.featuremapToLoc(self.loc10_2, conv10_2)
        loc_conv11_2 = self.featuremapToLoc(self.loc11_2, conv11_2)

        locs = torch.cat([loc_conv4_3, loc_conv7, loc_conv8_2, loc_conv9_2, loc_conv10_2, loc_conv11_2], dim=1)
        return locs
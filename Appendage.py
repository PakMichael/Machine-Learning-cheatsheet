import torch.nn as nn
import torch.nn.functional as F


class Appendage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv8_1 = nn.Conv2d(4096, 1024, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)

        self.conv9_1 = nn.Conv2d(2048, 512, kernel_size=3, padding=1)
        self.conv9_2 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)

        self.conv10_1 = nn.Conv2d(1024, 256, kernel_size=3, padding=1)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.conv11_1 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv8_1(x))
        x = F.relu(self.conv8_2(x))
        conv8_2 = x

        x = F.relu(self.conv9_1(x))
        x = F.relu(self.conv9_2(x))
        conv9_2 = x

        x = F.relu(self.conv10_1(x))
        x = F.relu(self.conv10_2(x))
        conv10_2 = x

        x = F.relu(self.conv11_1(x))
        x = F.relu(self.conv11_2(x))
        conv11_2 = x

        return conv8_2, conv9_2, conv10_2, conv11_2

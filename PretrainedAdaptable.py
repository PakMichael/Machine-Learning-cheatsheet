import torch.nn as nn
import torch.nn.functional as F
from utils import *


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.create_vgg16()
        self.load_params()

    def create_vgg16(self):
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7, padding=3)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)

    def load_params(self):
        state_dict = self.state_dict()
        self_keys = list(state_dict.keys())
        for indx, key in enumerate(pretrained_model.state_dict().keys()):
            if indx >= 26: break;
            state_dict[self_keys[indx]] = pretrained_model.state_dict()[key]

        state_dict['conv6.weight'] = pretrained_model.state_dict()['classifier.0.weight'].view(4096, 512, 7, 7)
        state_dict['conv6.bias'] = pretrained_model.state_dict()['classifier.0.bias'].view(4096)

        state_dict['conv7.weight'] = pretrained_model.state_dict()['classifier.3.weight'].view(4096, 4096, 1, 1)
        state_dict['conv7.bias'] = pretrained_model.state_dict()['classifier.3.bias'].view(4096)
        self.load_state_dict(state_dict)

    def forward(self, image):
        x = F.relu(self.conv1_1(image))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4_3 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        conv7 = x
        return conv4_3, conv7

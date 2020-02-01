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
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        classifier = []
        classifier.append(nn.Linear(7 * 7 * 512, 4096))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Dropout(0.5))
        classifier.append(nn.Linear(4096, 4096))
        classifier.append(nn.ReLU(True))
        classifier.append(nn.Dropout(0.5))
        classifier.append(nn.Linear(4096, 1000))

        self.classifier = nn.Sequential(*classifier)

    def load_params(self):
        state_dict = self.state_dict()
        self_keys = list(state_dict.keys())
        for indx, key in enumerate(pretrained_model.state_dict().keys()):
            state_dict[self_keys[indx]] = pretrained_model.state_dict()[key]
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
        conv4_3 = F.relu(self.conv4_3(x))
        x = self.pool4(conv4_3)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        conv5_3 = F.relu(self.conv5_3(x))
        x = self.pool5(conv5_3)

        x = self.avgpool(x)
        x = self.classifier(x.view(-1, 512 * 7 * 7))
        return x, conv4_3, conv5_3
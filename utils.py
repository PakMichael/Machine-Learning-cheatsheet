import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
import json

def getLabels():
    with open('./labels.json', encoding='latin-1') as json_file:
        data = json.load(json_file)
    return data



transform=transforms.Compose([transforms.Resize((224,224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

pretrained_model=models.vgg16(pretrained=True)
pretrained_keys=list(pretrained_model.state_dict().keys())

def getImage(path):
    return Image.open(path)

def showImage(img):
    plt.imshow(img)
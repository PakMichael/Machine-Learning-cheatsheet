import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import models, transforms
import json
from torch import mean

def getLabels():
    with open('./labels.json', encoding='latin-1') as json_file:
        data = json.load(json_file)
    return data


transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

pretrained_model = models.vgg16(pretrained=True)
pretrained_keys = list(pretrained_model.state_dict().keys())


def getImage(path):
    return Image.open(path)

def showImage(img):
    plt.figure()
    plt.imshow(img)

def show_item(item):
    fig, ax = plt.subplots(1)
    ax.imshow(item['image'])
    for obj in item['objects']:
        width = (obj['xmax'] - obj['xmin'])*item['image'].width
        height = (obj['ymax'] - obj['ymin'])*item['image'].height
        rect = patches.Rectangle((obj['xmin']*item['image'].width, obj['ymin']*item['image'].height), width, height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(obj['xmin'], obj['ymin'], obj['name'])


    plt.show()

def showActivations(featureMap, img):
    activations = featureMap.squeeze(0)
    plt.figure()
    plt.imshow(mean(activations, dim=0).detach().numpy(),
               extent=[0, img.width, img.height, 0])  # extent=[(left, right, bottom, top)]
    plt.imshow(img, alpha=0.2)

import torch
import glob
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F

fullPath = "D:\\Downloads\\pascal-voc\\pascal-voc\\"
from utils import *


def getAllAnotations(path):
    files = [f for f in glob.glob(path + "**/Annotations/*.xml", recursive=True)]
    return files


def makeItemFromAnnotation(annotation, path):
    tree = ET.parse(annotation)
    root = tree.getroot()

    item = {}
    objects = []
    for child in root:
        if child.tag == 'filename':
            item.update({'image': path + 'JPEGImages\\' + child.text})
        if child.tag == 'size':
            for size in child:
                item.update({size.tag: size.text})
        if child.tag == 'object':
            img_obj = {}
            for object in child:
                if object.tag == 'name':
                    img_obj.update({object.tag: object.text})
                if object.tag == 'bndbox':
                    for bndbox in object:
                        img_obj.update({bndbox.tag: int(bndbox.text)})
            objects.append(img_obj)
    item.update({'objects': objects})
    return item


def scale_down(item, scale=(224, 224)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    image = item['image']
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)

    new_img = F.resize(image, (224, 224))
    new_dims = torch.FloatTensor([new_img.width, new_img.height, new_img.width, new_img.height]).unsqueeze(0)

    scale = new_dims / old_dims

    new_obs = []
    for obj in item['objects']:
        scaled_obj = obj
        scaled_obj['xmin'] *= scale[0][0].item() / new_img.width
        scaled_obj['ymin'] *= scale[0][1].item() / new_img.height
        scaled_obj['xmax'] *= scale[0][2].item() / new_img.width
        scaled_obj['ymax'] *= scale[0][3].item() / new_img.height
        new_obs += [scaled_obj]

    new_image = F.to_tensor(new_img)
    new_image = F.normalize(new_image, mean=mean, std=std)

    return new_image, new_obs


class VOCdataset(torch.utils.data.Dataset):
    def __init__(self, voc_year):
        self.voc_year = voc_year
        assert self.voc_year in {'VOC2012', 'VOC2007'}

        self.all_items = getAllAnotations(fullPath + self.voc_year + '\\')

    def __len__(self):
        return len(self.all_items)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = makeItemFromAnnotation(self.all_items[idx], fullPath + self.voc_year + '\\')
        item.update({'image': getImage(item['image'])})
        image, objects = scale_down(item)

        return image, objects

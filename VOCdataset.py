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
                # if object.tag == 'name':
                # img_obj.update({object.tag: object.text})
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

    all_objects = []
    for obj in item['objects']:
        single_object = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]

        single_object[0] *= scale[0][0].item() / new_img.width #xmin
        single_object[1] *= scale[0][1].item() / new_img.height #ymin

        single_object[2] =  single_object[2] * scale[0][2].item() / new_img.width -  single_object[0] #width
        single_object[3] = single_object[3] * scale[0][3].item() / new_img.height - single_object[1] #height

        single_object[0]+=single_object[2]/ 2 #cx
        single_object[1]+=single_object[3]/ 2 #cy

        all_objects+=[single_object]

    new_image = F.to_tensor(new_img)
    # new_image = F.normalize(new_image, mean=mean, std=std)

    return new_image, all_objects


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

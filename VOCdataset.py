import torch
import glob
import xml.etree.ElementTree as ET

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
        item = makeItemFromAnnotation(self.all_items[idx],fullPath + self.voc_year + '\\')
        item.update({'image': getImage(item['image'])})

        return item

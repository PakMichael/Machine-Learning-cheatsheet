import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import models, transforms
import json
from torch import mean
import torch

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
        width = (obj['xmax'] - obj['xmin']) * item['image'].width
        height = (obj['ymax'] - obj['ymin']) * item['image'].height
        rect = patches.Rectangle((obj['xmin'] * item['image'].width, obj['ymin'] * item['image'].height), width, height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(obj['xmin'], obj['ymin'], obj['name'])

    plt.show()


def show_img_obj_scaled(img, objs):
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for obj in objs:
        x = (obj[0] - obj[2] / 2) * img.width
        y = (obj[1] - obj[3] / 2) * img.height
        rect = patches.Rectangle((x, y), obj[2] * img.width, obj[3] * img.height,
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()


def showActivations(featureMap, img):
    activations = featureMap.squeeze(0)
    plt.figure()
    plt.imshow(mean(activations, dim=0).detach().numpy(),
               extent=[0, img.width, img.height, 0])  # extent=[(left, right, bottom, top)]
    plt.imshow(img, alpha=0.2)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, epochs_since_improvement, model, optimizer, loss, best_loss, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param model: model
    :param optimizer: optimizer
    :param loss: validation loss in this epoch
    :param best_loss: best validation loss achieved so far (not necessarily in this checkpoint)
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'loss': loss,
             'best_loss': best_loss,
             'model': model,
             'optimizer': optimizer}
    filename = 'checkpoint_ssd300.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)
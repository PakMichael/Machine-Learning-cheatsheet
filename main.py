import time
from FullModel import FullModel
from BoxLoss import BoxLoss
from VOCdataset import VOCdataset
from utils import *
import torchvision.transforms.functional as F
import torch
from Tests import *

import os.path
train_dataset = VOCdataset('VOC2012')
best_idx = [6568, 5784, 5785, 6569, 5786, 6570, 4216, 3432, 5000, 5787]

reduced_priors_best=torch.cat([ torch.Tensor(prior_boxes[a]).unsqueeze(0) for a in best_idx],dim=0)





print(reduced_priors_best, train_dataset[1][1])
object_boxes= torch.Tensor([[0.2660, 0.3132, 0.3940, 0.4377]])
reference_overlap = find_jaccard_overlap(object_boxes, reduced_priors_best)
ref_class_scores, ref_sort_ind = reference_overlap.sort(dim=1, descending=True)
print(ref_class_scores,ref_sort_ind)
def restoreBox(tens):
    tens = tens.numpy()
    w = tens[2] - tens[0]
    h = tens[3] - tens[1]
    return [tens[0] + w / 2, tens[1] + h / 2, w, h]
show_img_obj_scaled(F.to_pil_image(train_dataset[1][0]), [restoreBox(reduced_priors_best[ref_sort_ind.numpy()[0][a]]) for a in range(10)])
#
# def run():
#     torch.multiprocessing.freeze_support()
#
#
# if __name__ == '__main__':
#     run()
#
# model = FullModel()
#
# criterion = BoxLoss(prior_boxes)
# train_dataset = VOCdataset('VOC2012')
# validation_dataset = VOCdataset('VOC2007')
#
#
# image = train_dataset[1][0]
# #showImage(F.to_pil_image(image))
# img = image.unsqueeze(0)  # torch.Size([1, 3, 224, 224])
# #
# loc_res = model(img)
# # print(prior_boxes.shape)
# #
# # print(loc_res)
# reference_overlap = find_jaccard_overlap(train_dataset[1][1], prior_boxes)
# ref_class_scores, ref_sort_ind = reference_overlap.sort(dim=1, descending=True)
# #
# temp_score = ref_class_scores[0][:40]
#
# temp_score[temp_score < 0.43] = 0
#
#
#
# def restoreBox(tens):
#     tens = tens.numpy()
#     w = tens[2] - tens[0]
#     h = tens[3] - tens[1]
#     return [tens[0] + w / 2, tens[1] + h / 2, w, h]
#
# print('pri',ref_sort_ind.numpy()[1][:10])
# # show_img_obj_scaled(F.to_pil_image(train_dataset[1][0]), train_dataset[1][1])
# #
# #show_img_obj_scaled(F.to_pil_image(image), [train_dataset[1][1][0] ])
# show_img_obj_scaled(F.to_pil_image(image), [restoreBox(prior_boxes[ref_sort_ind.numpy()[1][a]]) for a in range(10)])
# # prior_boxes[ref_class_scores[0, :]<0.42]=0
# # #
# # # for idx in ref_sort_ind[0][:50].numpy():
# # #     print(prior_boxes[idx])
# #
# # # prior_boxes[ref_class_scores[0] < 0.1] = 0
# # # print(prior_boxes[ref_sort_ind[0][1]])
# #
# #
# # def restoreBox(tens):
# #     tens = tens.numpy()
# #     w = tens[2] - tens[0]
# #     h = tens[3] - tens[1]
# #     return [tens[0] + w / 2, tens[1] + h / 2, w, h]
# #
# #
# # for _, obj in enumerate(sort_ind):
# #     obj = obj.numpy()
# #     show_img_obj_scaled(F.to_pil_image(train[1][0]), [restoreBox(prior_boxes[obj[a]]) for a in range(10)])
# #
# # # item = train[1][1]
# # # overl = find_jaccard_overlap(item, prior_boxes)
# # # class_scores, sort_ind = overl.sort(dim=1, descending=True)
# #
# # print_freq = 200
# # batch_size = 1  # batch size
# # start_epoch = 0  # start at this epoch
# # workers = 0
# # epochs = 200  # number of epochs to run without early-stopping
# # lr = 1e-3
# # momentum = 0.9
# # weight_decay = 5e-4
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # grad_clip = None
# # epochs_since_improvement = 0
# #
# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
# #                                            collate_fn=train_dataset.collate_fn, num_workers=workers,
# #                                            pin_memory=True)  # note that we're passing the collate function here
# #
# # val_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True,
# #                                          collate_fn=validation_dataset.collate_fn, num_workers=workers,
# #                                          pin_memory=True)
# #
# # checkpoint = None
# # if os.path.isfile('checkpoint_ssd300.pth.tar'):
# #     print("Checkpoint exist")
# #     checkpoint = torch.load('checkpoint_ssd300.pth.tar')
# #     model = checkpoint['model']
# # else:
# #     print("Checkpoint doesn't exist")
# # torch.save(model, 'model')
# #
# # biases = list()
# # not_biases = list()
# # for param_name, param in model.named_parameters():
# #     if param.requires_grad:
# #         if param_name.endswith('.bias'):
# #             biases.append(param)
# #         else:
# #             not_biases.append(param)
# # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
# #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
# # if checkpoint!=None:
# #     optimizer=checkpoint['optimizer']
# #
# # def train(train_loader, model, criterion, optimizer, epoch):
# #     model.train()  # training mode enables dropout
# #
# #     batch_time = AverageMeter()  # forward prop. + back prop. time
# #     data_time = AverageMeter()  # data loading time
# #     losses = AverageMeter()  # loss
# #
# #     start = time.time()
# #
# #     # Batches
# #     for i, (images, boxes) in enumerate(train_loader):
# #         data_time.update(time.time() - start)
# #
# #         # Move to default device
# #         images = images.to(device)  # (batch_size (N), 3, 300, 300)
# #         boxes = [b.to(device) for b in boxes]
# #
# #         # Forward prop.
# #         predicted_locs = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
# #
# #         # Loss
# #         loss = criterion(predicted_locs, boxes)  # scalar
# #
# #         # Backward prop.
# #         optimizer.zero_grad()
# #         loss.backward()
# #
# #         # Clip gradients, if necessary
# #         if grad_clip is not None:
# #             clip_gradient(optimizer, grad_clip)
# #
# #         # Update model
# #         optimizer.step()
# #
# #         losses.update(loss.item(), images.size(0))
# #         batch_time.update(time.time() - start)
# #
# #         start = time.time()
# #
# #         # Print status
# #         if i % print_freq == 0:
# #             print('Epoch: [{0}][{1}/{2}]\t'
# #                   'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
# #                   'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
# #                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
# #                                                                   batch_time=batch_time,
# #                                                                   data_time=data_time, loss=losses))
# #     del predicted_locs, images, boxes
# #
# #
# # def validate(val_loader, model, criterion):
# #     """
# #     One epoch's validation.
# #
# #     :param val_loader: DataLoader for validation data
# #     :param model: model
# #     :param criterion: MultiBox loss
# #     :return: average validation loss
# #     """
# #     model.eval()  # eval mode disables dropout
# #
# #     batch_time = AverageMeter()
# #     losses = AverageMeter()
# #
# #     start = time.time()
# #
# #     # Prohibit gradient computation explicity because I had some problems with memory
# #     with torch.no_grad():
# #         # Batches
# #         for i, (images, boxes) in enumerate(val_loader):
# #
# #             # Move to default device
# #             images = images.to(device)  # (N, 3, 300, 300)
# #             boxes = [b.to(device) for b in boxes]
# #
# #             # Forward prop.
# #             predicted_locs = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
# #
# #             # Loss
# #             loss = criterion(predicted_locs, boxes)
# #
# #             losses.update(loss.item(), images.size(0))
# #             batch_time.update(time.time() - start)
# #
# #             start = time.time()
# #
# #             # Print status
# #             if i % print_freq == 0:
# #                 print('[{0}/{1}]\t'
# #                       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
# #                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i, len(val_loader),
# #                                                                       batch_time=batch_time,
# #                                                                       loss=losses))
# #
# #     print('\n * LOSS - {loss.avg:.3f}\n'.format(loss=losses))
# #
# #     return losses.avg
# #
# #
# # for epoch in range(start_epoch, epochs):
# #
# #     train(train_loader=train_loader,
# #           model=model,
# #           criterion=criterion,
# #           optimizer=optimizer,
# #           epoch=epoch)
# #
# #     val_loss = validate(val_loader=val_loader,
# #                         model=model,
# #                         criterion=criterion)
# #
# #     # Did validation loss improve?
# #     is_best = val_loss < best_loss
# #     best_loss = min(val_loss, best_loss)
# #     if not is_best:
# #         epochs_since_improvement += 1
# #         print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
# #
# #     else:
# #         epochs_since_improvement = 0
# #
# #     # Save checkpoint
# #     save_checkpoint(epoch, epochs_since_improvement, model, optimizer, val_loss, best_loss, is_best)
# #     print('save_checkpoint')

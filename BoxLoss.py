from torch import nn
import torch
from Tests import find_jaccard_overlap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BoxLoss(nn.Module):
    def __init__(self, priors, threshold=0.5, neg_pos_ratio=3, alpha=1.):
        super(BoxLoss, self).__init__()
        self.priors = priors
        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha

        self.smooth_l1 = nn.L1Loss()

    def forward(self, predicted_locs, boxes):
        batch_size = predicted_locs.size(0)
        n_priors = self.priors.size(0)

        assert n_priors == predicted_locs.size(1)

        true_locs = torch.zeros((batch_size, n_priors, 4), dtype=torch.float).to(device)
        true_classes = torch.ones((batch_size, n_priors), dtype=torch.long).to(device)

        # For each image
        for i in range(batch_size):
            n_objects = boxes[i].size(0)

            overlap = find_jaccard_overlap(boxes[i], self.priors)
            overlap_for_each_prior, object_for_each_prior = overlap.max(dim=0)

            _, prior_for_each_object = overlap.max(dim=1)
            object_for_each_prior[prior_for_each_object] = torch.LongTensor(range(n_objects))
            overlap_for_each_prior[prior_for_each_object] = 1.

            true_locs[i] = boxes[i][object_for_each_prior]


        positive_priors = true_classes != 0
        loc_loss = self.smooth_l1(predicted_locs[positive_priors], true_locs[positive_priors])


        return loc_loss

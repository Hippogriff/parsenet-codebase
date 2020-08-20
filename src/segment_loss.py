"""
This script defines loss functions for AE based training.
"""
import numpy as np
import torch
import time
import numpy as np
from lapsolver import solve_dense
from torch.nn import ReLU
from torch.autograd.variable import Variable
from src.mean_shift import MeanShift
import torch.nn.functional as F


meanshift = MeanShift()
WEIGHT = False
relu = ReLU()

if WEIGHT:
    nllloss = torch.nn.NLLLoss(weight=old_weight)
else:
    nllloss = torch.nn.NLLLoss()


class EmbeddingLoss:
    def __init__(self, margin=1.0, if_mean_shift=False):
        """
        Defines loss function to train embedding network.
        :param margin: margin to be used in triplet loss.
        :param if_mean_shift: bool, whether to use mean shift
        iterations. This is only used in end to end training.
        """
        self.margin = margin
        self.if_mean_shift = if_mean_shift

        
    def triplet_loss(self, output, labels: np.ndarray, iterations=5):
        """
        Triplet loss
        :param output: output embedding from the network. size: B x 128 x N
        where B is the batch size, 128 is the dim size and N is the number of points.
        :param labels: B x N
        """
        max_segments = 5
        batch_size = output.shape[0]
        N = output.shape[2]
        loss_diff = torch.tensor([0.], requires_grad=True).cuda()
        relu = torch.nn.ReLU()
        
        output = output.permute(0, 2, 1)
        output = torch.nn.functional.normalize(output, p=2, dim=2)
        new_output = []
        
        if self.if_mean_shift:
            for b in range(batch_size):
                new_X, bw = meanshift.mean_shift(output[b], 4000,
                                                 0.015, iterations=iterations,
                                                 nms=False)
                new_output.append(new_X)
            output = torch.stack(new_output, 0)
        
        num_sample_points = {}
        sampled_points = {}
        for i in range(batch_size):
            sampled_points[i] = {}
            p = labels[i]
            unique_labels = np.unique(p)

            # number of points from each cluster.
            num_sample_points[i] = min([N // unique_labels.shape[0] + 1, 30])
            for l in unique_labels:
                ix = np.isin(p, l)
                sampled_indices = np.where(ix)[0]
                # point indices that belong to a certain cluster.
                sampled_points[i][l] = np.random.choice(
                        list(sampled_indices),
                        num_sample_points[i],
                        replace=True)

        sampled_predictions = {}
        for i in range(batch_size):
            sampled_predictions[i] = {}
            for k, v in sampled_points[i].items():
                pred = output[i, v, :]
                sampled_predictions[i][k] = pred

        all_satisfied = 0
        only_one_segments = 0
        for i in range(batch_size):
            len_keys = len(sampled_predictions[i].keys())
            keys = list(sorted(sampled_predictions[i].keys()))
            num_iterations = min([max_segments * max_segments, len_keys * len_keys])
            normalization = 0
            if len_keys == 1:
                only_one_segments += 1
                continue

            loss_shape = torch.tensor([0.], requires_grad=True).cuda()
            for _ in range(num_iterations):
                k1 = np.random.choice(len_keys, 1)[0]
                k2 = np.random.choice(len_keys, 1)[0]
                if k1 == k2:
                    continue
                else:
                    normalization += 1

                pred1 = sampled_predictions[i][keys[k1]]
                pred2 = sampled_predictions[i][keys[k2]]

                Anchor = pred1.unsqueeze(1)
                Pos = pred1.unsqueeze(0)
                Neg = pred2.unsqueeze(0)

                diff_pos = torch.sum(torch.pow((Anchor - Pos), 2), 2)
                diff_neg = torch.sum(torch.pow((Anchor - Neg), 2), 2)
                constraint = diff_pos - diff_neg + self.margin
                constraint = relu(constraint)

                # remove diagonals corresponding to same points in anchors
                loss = torch.sum(constraint) - constraint.trace()

                satisfied = torch.sum(constraint > 0) + 1.0
                satisfied = satisfied.type(torch.cuda.FloatTensor)
                
                loss_shape = loss_shape + loss / satisfied.detach()

            loss_shape = loss_shape / (normalization + 1e-8)
            loss_diff = loss_diff + loss_shape
        loss_diff = loss_diff / (batch_size - only_one_segments + 1e-8)
        return loss_diff


def evaluate_miou(gt_labels, pred_labels):
    N = gt_labels.shape[0]
    C = pred_labels.shape[2]
    pred_labels = np.argmax(pred_labels, 2)
    IoU_category = 0

    for n in range(N):
        label_gt = gt_labels[n]
        label_pred = pred_labels[n]
        IoU_part = 0.0

        for label_idx in range(C):
            locations_gt = (label_gt == label_idx)
            locations_pred = (label_pred == label_idx)
            I_locations = np.logical_and(locations_gt, locations_pred)
            U_locations = np.logical_or(locations_gt, locations_pred)
            I = np.sum(I_locations) + np.finfo(np.float32).eps
            U = np.sum(U_locations) + np.finfo(np.float32).eps
            IoU_part = IoU_part + I / U
        IoU_sample = IoU_part / C
        IoU_category += IoU_sample
    return IoU_category / N


def primitive_loss(pred, gt):
    return nllloss(pred, gt)

import sys

sys.path.append("../")

import time
from scipy import stats
from src.primitives import ResidualLoss
from src.primitive_forward import fit_one_shape_torch
from src.fitting_optimization import FittingModule
import numpy as np
import torch
from src.fitting_utils import (
    weights_normalize,
    to_one_hot,
    match,
)
from open3d import *

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments

colors = np.random.random((10, 3))
colors[0] = np.array([0.2, 0.3, 0.6])
colors[1] = np.array([1, 0, 0])  # plane, red
colors[2] = np.array([0, 1, 0])  # bspline, green
colors[3] = np.array([0, 0, 1])  # Cone, blue
colors[4] = np.array([1, 1, 0])  # Cylinder, yellow
colors[5] = np.array([1.0, 0.50, 1.0])  # sphere, pink
colors[6] = np.array([1, 0.1, 0.7])  # other
colors[7] = np.array([0.8, 0.10, 0.30])  # revolution
colors[8] = np.array([0.2, 0.7, 0.1])  # extrusion
colors[9] = np.array([0.0, 0.0, 0.0])  # close bspline


def convert_to_one_hot(data):
    """
    Given a tensor of N x D, converts it into one_hot
    by filling zeros to non max along every row.
    """
    N, C = data.shape
    max_rows = torch.max(data, 1)[1]
    data = to_one_hot(max_rows, C)
    return data.float()


class Evaluation:
    def __init__(self, userspace=None, closed_path=None, open_path=None):
        """
        Calculates residual loss for train and eval mode.
        """
        if closed_path == None:
            closed_path = "logs/pretrained_models/closed_spline.pth"
        if open_path == None:
            open_path = "logs/pretrained_models/open_spline.pth"

        self.res_loss = ResidualLoss()
        self.fitter = FittingModule(closed_path, open_path)

        for param in self.fitter.closed_control_decoder.parameters():
            param.requires_grad = False

        for param in self.fitter.open_control_decoder.parameters():
            param.requires_grad = False
        self.ms = MeanShift()

    def guard_mean_shift(self, embedding, quantile, iterations, kernel_type="gaussian"):
        """
        Some times if band width is small, number of cluster can be larger than 50, that
        but we would like to keep max clusters 50 as it is the max number in our dataset.
        In that case you increase the quantile to increase the band width to decrease
        the number of clusters.
        """
        while True:
            _, center, bandwidth, cluster_ids = self.ms.mean_shift(
                embedding, 10000, quantile, iterations, kernel_type=kernel_type
            )
            if torch.unique(cluster_ids).shape[0] > 49:
                quantile *= 1.2
            else:
                break
        return center, bandwidth, cluster_ids

    def fitting_loss(
            self,
            embedding,
            points,
            normals,
            labels,
            primitives,
            primitives_log_prob,
            quantile=0.125,
            iterations=5,
            lamb=1.0,
            debug=False,
            eval=False,
    ):
        """
        Given point embedding does clusters to get the cluster centers and
        per point membership weights.
        :param embedding:
        :param points:
        :
        """
        batch_size = embedding.shape[0]
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=2)
        parameters = None

        for b in range(batch_size):
            center, bandwidth, cluster_ids = self.guard_mean_shift(
                embedding[b], quantile, iterations, kernel_type="gaussian"
            )

            weights = center @ torch.transpose(embedding[b], 1, 0)
            primitives_log_prob = torch.max(primitives_log_prob, 1)[1]
            primitives_log_prob = primitives_log_prob.data.cpu().numpy()

            if not eval:
                loss, parameters, pred_mesh, rows, cols, distance = self.residual_train_mode(
                    points[b],
                    normals[b],
                    labels[b],
                    cluster_ids,
                    primitives[b],
                    weights,
                    bandwidth,
                    lamb=lamb
                )
            else:
                loss, parameters, pred_mesh, gtpoints, distance, _, _ = self.residual_eval_mode(
                    points[b],
                    normals[b],
                    labels[b],
                    cluster_ids,
                    primitives[b],
                    primitives_log_prob[b],
                    weights,
                    bandwidth,
                    lamb=lamb,
                    sample_points=False,
                    if_optimize=False,
                )
                # in the eval mode, we make hard selection from weight matrix.
                weights = to_one_hot(cluster_ids, np.unique(cluster_ids.data.data.cpu().numpy()).shape[0]).T

            with torch.no_grad():
                s_iou, p_iou, _, _ = SIOU_matched_segments(labels[b], cluster_ids.data.cpu().numpy(),
                                                           primitives_log_prob[b], primitives[b], weights.T)
            loss = loss + [s_iou, p_iou]
        return loss, [parameters, cluster_ids.data.cpu().numpy(), weights]

    def residual_train_mode(
            self, points, normals, labels, cluster_ids, primitives, weights, bw, lamb=1.0):
        """
        Takes embedding and defines a residual loss in an end to end manner.
        :param points:
        :param normals:
        :param labels:
        :param cluster_ids:
        :param primitives:
        :param weights:
        :param bw:
        :param lamb:
        """
        if not isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.data.cpu().numpy()
        rows, cols, unique_target, unique_pred = match(labels, cluster_ids)

        select_pred_indices = []
        gt_indices = []
        pred_indices = []
        data = []
        all_segments = []
        for index, i in enumerate(unique_pred):
            gt_indices_i = labels == cols[i]
            pred_indices_i = cluster_ids == i

            if (np.sum(gt_indices_i) == 0) or (np.sum(pred_indices_i) == 0):
                continue

            select_pred_indices.append(i)
            gt_indices.append(gt_indices_i)
            pred_indices.append(pred_indices_i)

            l = stats.mode(primitives[gt_indices_i])[0]
            data.append([points, normals, l, points[gt_indices_i], None, (index, i)])

        all_segments.append([data, weights, bw])

        for index, data in enumerate(all_segments):
            torch.cuda.empty_cache()
            new_value = []
            new_data = None
            data_, weights_first, bw = data

            weights = weights_normalize(weights_first, float(bw))

            weights = torch.transpose(weights, 1, 0)
            gt_points, recon_points = fit_one_shape_torch(
                data_, self.fitter, weights, bw, eval=False
            )
            distance = self.res_loss.residual_loss(
                gt_points, self.fitter.fitting.parameters
            )
            Loss = self.separate_losses(distance, gt_points, lamb=lamb)
        return Loss, self.fitter.fitting.parameters, None, rows, cols, distance

    def residual_eval_mode(
            self,
            points,
            normals,
            labels,
            cluster_ids,
            primitives,
            pred_primitives,
            weights,
            bw,
            lamb=1.0,
            sample_points=False,
            if_optimize=False,
            if_visualize=False,
            epsilon=None
    ):
        """
        Computes residual error in eval mode.
        """
        primitives_pred_hot = to_one_hot(pred_primitives, 10)

        if not isinstance(cluster_ids, np.ndarray):
            cluster_ids = cluster_ids.data.cpu().numpy()

        # weights = weights.data.cpu().numpy()
        weights = (
            to_one_hot(cluster_ids,
                       np.unique(cluster_ids).shape[0], device_id=weights.get_device()).data.cpu().numpy().T)

        rows, cols, unique_target, unique_pred = match(labels, cluster_ids)
        gt_indices = []
        pred_indices = []
        data = []
        all_segments = []

        for index, i in enumerate(unique_pred):
            # TODO some labels might be missing from unique_pred
            gt_indices_i = labels == cols[index]
            pred_indices_i = cluster_ids == i

            if if_visualize:
                if (np.sum(gt_indices_i) == 0) and (np.sum(pred_indices_i) == 0):
                    continue
                elif (np.sum(gt_indices_i) > 0) and (np.sum(pred_indices_i) == 0):
                    continue
            else:
                if (np.sum(gt_indices_i) == 0) or (np.sum(pred_indices_i) == 0):
                    continue

            l = stats.mode(pred_primitives[pred_indices_i])[0]

            if if_visualize:
                # for post process refinement, we need to have matching of
                # predicted points with ground truth points. Since we don't
                # know the gt labels, we simply take the predicted points at matched points.
                data.append(
                    [
                        points[pred_indices_i],
                        normals[pred_indices_i],
                        l,
                        points[pred_indices_i],
                        pred_indices_i,
                        (index, i),
                    ]
                )
            else:
                data.append(
                    [
                        points[pred_indices_i],
                        normals[pred_indices_i],
                        l,
                        points[gt_indices_i],
                        pred_indices_i,
                        (index, i),
                    ]
                )
        all_segments.append([data, weights, bw])

        for index, data in enumerate(all_segments):
            new_value = []
            t1 = time.time()
            new_data = None
            data_, weights_first, bw = data

            if isinstance(new_data, np.ndarray):
                # take previous value
                weights_first = new_data

            weights_first = torch.from_numpy(weights_first.astype(np.float32)).cuda(points.get_device())

            weights = weights_normalize(weights_first, float(bw))
            weights = torch.transpose(weights, 1, 0)
            weights = to_one_hot(
                torch.max(weights, 1)[1].data.cpu().numpy(), weights.shape[1],
                device_id=points.get_device())

            gt_points, recon_points = fit_one_shape_torch(
                data_,
                self.fitter,
                weights,
                bw,
                eval=True,
                sample_points=sample_points,
                if_optimize=if_optimize,
                if_visualize=if_visualize,
            )

            Loss = []
            if not if_visualize:
                distance = self.res_loss.residual_loss(
                    gt_points, self.fitter.fitting.parameters, sqrt=True
                )
                # Note that gt_points keys are not continuous
                Loss = self.separate_losses(distance, gt_points, lamb=lamb)
            else:
                distance = None
            if sample_points:
                pred_meshes = None
            else:
                pred_meshes = None
        return Loss, self.fitter.fitting.parameters, pred_meshes, gt_points, distance, rows, cols

    def separate_losses(self, distance, gt_points, lamb=1.0):
        """
        The idea is to define losses for geometric primitives and splines separately.
        This is only used in evaluation mode.
        :param distance: dictionary containing residual loss for all the geometric
        primitives and splines
        :param gt_points: dictionary containing ground truth points for matched
        points, used to ignore loss for the surfaces with smaller than threshold points
        """
        Loss = []
        geometric_loss = []
        spline_loss = []
        # TODO remove parts that are way off from the ground truth points
        for item, v in enumerate(sorted(gt_points.keys())):
            # cases where number of points are less than 20 or
            # bspline surface patches with less than 100 points
            if gt_points[v] is None:
                continue
            # if gt_points[v].shape[0] < 100:
            #     continue
            if distance[v][1] > 1:
                # most probably a degenerate case
                distance[v][1] = torch.ones(1).cuda()[0] * 0.1

            if distance[v][0] in ["closed-spline", "open-spline"]:
                spline_loss.append(distance[v][1].item())
                Loss.append(distance[v][1] * lamb)
            else:
                geometric_loss.append(distance[v][1].item())
                Loss.append(distance[v][1])

        try:
            Loss = torch.mean(torch.stack(Loss))
        except:
            Loss = torch.zeros(1).cuda()

        if len(geometric_loss) > 0:
            geometric_loss = np.mean(geometric_loss)
        else:
            geometric_loss = None

        if len(spline_loss) > 0:
            spline_loss = np.mean(spline_loss)
        else:
            spline_loss = None
        return [Loss, geometric_loss, spline_loss]

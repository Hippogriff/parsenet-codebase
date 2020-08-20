import numpy as np
from src.utils import visualize_point_cloud_from_labels
from src.utils import draw_geometries
from lapsolver import solve_dense
import torch
from src.segment_utils import SIOU_matched_segments
from src.segment_utils import to_one_hot


def mean_IOU_one_sample(pred, gt, C):
    IoU_part = 0.0
    for label_idx in range(C):
        locations_gt = (gt == label_idx)
        locations_pred = (pred == label_idx)
        I_locations = np.logical_and(locations_gt, locations_pred)
        U_locations = np.logical_or(locations_gt, locations_pred)
        I = np.sum(I_locations) + np.finfo(np.float32).eps
        U = np.sum(U_locations) + np.finfo(np.float32).eps
        IoU_part = IoU_part + I / U
    return IoU_part / C


def iou_segmentation(pred, gt):
    # preprocess the predictions and gt to remove the extras
    # swap (0, 6, 7) to closed surfaces which is 9
    # swap 8 to 2
    gt[gt == 0] = 9
    gt[gt == 6] = 9
    gt[gt == 7] = 9
    gt[gt == 8] = 2
    
    pred[pred == 0] = 9
    pred[pred == 6] = 9
    pred[pred == 7] = 9
    pred[pred == 8] = 2
    
    return mean_IOU_one_sample(pred, gt, 6)


def to_one_hot(target, maxx=50):
    target = torch.from_numpy(target.astype(np.int64)).cuda()
    N = target.shape[0]
    target_one_hot = torch.zeros((N, maxx))
    
    target_one_hot = target_one_hot.cuda()
    target_t = target.unsqueeze(1)
    target_one_hot = target_one_hot.scatter_(1, target_t.long(), 1)
    return target_one_hot


def matching_iou(matching, predicted_labels, labels):
    batch_size = labels.shape[0]
    IOU = []
    new_pred = []
    for b in range(batch_size):
        iou_b = []
        len_labels = np.unique(predicted_labels[b]).shape[0]
        rows, cols = matching[b]
        count = 0
        for r, c in zip(rows, cols):
            pred_indices = predicted_labels[b] == r
            gt_indices = labels[b] == c
            
            # if both input and predictions are empty, ignore that.
            if (np.sum(gt_indices) == 0) and  (np.sum(pred_indices) == 0):
                continue
            iou = np.sum(np.logical_and(pred_indices, gt_indices)) / (np.sum(np.logical_or(pred_indices, gt_indices)) + 1e-8)
            iou_b.append(iou)

        # find the mean of IOU over this shape
        IOU.append(np.mean(iou_b))
    return np.mean(IOU)


def relaxed_iou(pred, gt, max_clusters=50):
    batch_size, N, K = pred.shape
    normalize = torch.nn.functional.normalize
    one = torch.ones(1).cuda()

    norms_p = torch.sum(pred, 1)
    norms_g = torch.sum(gt, 1)
    cost = []

    for b in range(batch_size):
        p = pred[b]
        g = gt[b]
        c_batch = []
        dots = p.transpose(1, 0) @ g

        for k1 in range(K):
            c = []
            for k2 in range(K):
                r_iou = dots[k1, k2]
                r_iou = r_iou / (norms_p[b, k1] + norms_g[b, k2] - dots[k1, k2] + 1e-10)
                if r_iou < 0:
                    import ipdb; ipdb.set_trace()
                c.append(r_iou)
            c_batch.append(c)
        cost.append(c_batch)
    return cost


def p_coverage(points, parameters, ResidualLoss):
    """
    Compute the p coverage as described in the SPFN paper.
    Basically, for each input point, it finds the closest
    primitive and define that as the distance from the predicted
    surface. Mean over all these distance is reported.
    :param points: input point cloud, numpy array, N x 3
    :param parameters: dictionary of parameters predicted by the algorithm.
    """
    residual_reduce = ResidualLoss(one_side=True, reduce=False)
    points = torch.from_numpy(points).cuda()
    gpoints = {k: points for k in parameters.keys()}
    reduce_distance = residual_reduce.residual_loss(gpoints,
                                             parameters,
                                             sqrt=True)

    reduce_distance = [v[1] for k, v in reduce_distance.items()]
    reduce_distance = torch.stack([r for r in reduce_distance], 0)
    print (reduce_distance.shape)
    reduce_distance = torch.min(reduce_distance, 0)[0]
    
    cover = reduce_distance < 0.01
    cover = torch.mean(cover.float())
    mean_coverage = torch.mean(reduce_distance)
    return mean_coverage, cover


def separate_losses(distance, gt_points, lamb=1.0):
    """
    The idea is to define losses for geometric primitives and splines separately.
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
        if gt_points[v].shape[0] < 100:
            continue
        if distance[v][1] > 1:
            # most probably a degenerate case
            # give a fixed error for this.
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

def IOU(data):
    """
    Take the per shape output predictions, and produces segment IOU, and
    primitive type IOU.
    """
    Mapping ={"torus": 0,
        "plane": 1,
         "cone": 3,
         "cylinder": 4,
         "sphere": 5,
         "open-spline": 2,
         "closed-spline": 9}

    parameters = data["primitive_dict"]
    # setting the not assigned 
    
    primitives = data["primitives"]

    label_to_primitive = {}
    if (data.get("pred_primitives") is None):
        pred_primitives = np.zeros(data["points"].shape[0])
        for k, v in data["primitive_dict"].items():
            pred_primitives[data["seg_id"] == k] = Mapping[v[0]]
    else:
        pred_primitives = data["pred_primitives"]
        pred_primitives[pred_primitives == 0] = 9
        pred_primitives[pred_primitives == 6] = 9
        pred_primitives[pred_primitives == 7] = 9
        pred_primitives[pred_primitives == 8] = 2

        primitives[primitives == 0] = 9
        primitives[primitives == 6] = 9
        primitives[primitives == 7] = 9
        primitives[primitives == 8] = 2

    if (data.get("weights") is None):
        weights = to_one_hot(data["seg_id"],
                   np.unique(data["seg_id"]).shape[0]).data.cpu().numpy()
    else:
        weights = data["weights"]

    s_iou, p_iou, _, iou_b_prims = SIOU_matched_segments(data["labels"],
                                            data["seg_id"],
                                            pred_primitives,
                                            data["primitives"],
                                            weights)
    return s_iou, p_iou, iou_b_prims


def IOU_simple(data):
    """
    Take the per shape output predictions, and produces segment IOU, and
    primitive type IOU.
    """
    Mapping ={"torus": 0,
        "plane": 1,
         "cone": 3,
         "cylinder": 4,
         "sphere": 5,
         "open-spline": 2,
         "closed-spline": 9}

    parameters = data["primitive_dict"]
    # setting the not assigned 
    pred_primitives = np.zeros(data["points"].shape[0])

    label_to_primitive = {}
    if (data.get("pred_primitives") is None):
        for k, v in data["primitive_dict"].items():
            pred_primitives[data["seg_id"] == k] = Mapping[v[0]]
    else:
        pred_primitives = data["pred_primitives"]

    if (data.get("weights") is None):
        weights = to_one_hot(data["seg_id"],
                   np.unique(data["seg_id"]).shape[0]).data.cpu().numpy()
    else:
        weights = data["weights"]
    
    s_iou, p_iou, _ = SIOU_matched_segments(data["labels"],
                                            data["seg_id"],
                                            pred_primitives,
                                            data["primitives"],
                                            weights)
    return s_iou, p_iou


def preprocess(data, rem_unassign=False):
    N = data["seg_id"].shape[0]
    keep_indices = np.logical_not(data["seg_id"] == 100)
    print ("unassigned no. points ", N - np.sum(keep_indices))
    if rem_unassign:
        # assign nearest labels
        data = remove_unassigned(data)
    else:
        # remove the points that are not assigned.
        data["points"] = data["points"][keep_indices]
        data["normals"] = data["normals"][keep_indices]
        data["seg_id"] = data["seg_id"][keep_indices]
        data["primitives"] = data["primitives"][keep_indices]
        data["labels"] = data["labels"][keep_indices]
    return data

def remove_unassigned(data):
    """
    For un assigned points, assign the nearest neighbors label.
    """
    points = torch.from_numpy(data['points'].astype(np.float32)).cuda()
    dst_matrix = torch.sum((torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)) ** 2, 2)
    unassigned_index = data['seg_id'] == 100
    
    dst_matrix = dst_matrix.fill_diagonal_(2e8)
    dst_matrix[:, unassigned_index] = 2e8
    nearest_index = torch.min(dst_matrix, 1)[1].data.cpu().numpy()
    
    data['seg_id'][unassigned_index] = data['seg_id'][nearest_index[unassigned_index]]
    return data

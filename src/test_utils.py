import torch

from src.fitting_utils import to_one_hot
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments
from src.utils import chamfer_distance
from src.utils import fit_surface_sample_points

ms = MeanShift()


def convert_to_one_hot(data):
    """
    Given a tensor of N x D, converts it into one_hot
    by filling zeros to non max along every row.
    """
    N, C = data.shape
    max_rows = torch.max(data, 1)[1]

    data = to_one_hot(max_rows, C)
    return data.float()


def test(output, points, num_points=900):
    predicted_points, fitted_surfaces = fit_surface_sample_points(
        output.permute(0, 2, 1).data.cpu().numpy()[:, 0:num_points],
        points.permute(0, 2, 1).data.cpu().numpy()[:, 0:num_points],
        30,
    )
    distance = chamfer_distance(
        points.permute(0, 2, 1), predicted_points
    )
    return distance, predicted_points, fitted_surfaces


def IOU_from_embeddings(embedding, labels, primitives_log_prob, primitives, quantile, iterations=20):
    """
    Starting from embedding, it first cluster the shape and 
    then calculate the IOU scores
    """
    # import ipdb; ipdb.set_trace()
    B = embedding.shape[0]
    embedding = embedding.permute(0, 2, 1)
    primitives_log_prob = primitives_log_prob.permute(0, 2, 1)

    embedding = torch.nn.functional.normalize(embedding, p=2, dim=2)
    seg_IOUs = []
    prim_IOUs = []
    primitives_log_prob = torch.max(primitives_log_prob, 2)[1]
    primitives_log_prob = primitives_log_prob.data.cpu().numpy()

    for b in range(B):
        center, bandwidth, cluster_ids = ms.guard_mean_shift(embedding[b], quantile, iterations)
        weight = center @ torch.transpose(embedding[b], 1, 0)
        weight = convert_to_one_hot(weight)
        s_iou, p_iou, _ = SIOU_matched_segments(labels[b], cluster_ids.data.cpu().numpy(), primitives_log_prob[b],
                                                primitives[b].data.cpu().numpy(), weight.T.data.cpu().numpy())
        seg_IOUs.append([s_iou])
        prim_IOUs.append([p_iou])
    return [seg_IOUs, prim_IOUs]

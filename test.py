import sys

from open3d import *

sys.path.append("../")

import h5py

import numpy as np
import torch
from src.fitting_utils import (
    to_one_hot,
)
import os
from src.segment_utils import SIOU_matched_segments
from src.utils import chamfer_distance_single_shape
from src.segment_utils import sample_from_collection_of_mesh
from src.primitives import SaveParameters
from src.dataset_segments import Dataset
from src.residual_utils import Evaluation
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
prefix = ""

dataset = Dataset(
    1,
    24000,
    4000,
    4000,
    normals=True,
    primitives=True,
    if_train_data=False,
    prefix=prefix
)


def continuous_labels(labels_):
    new_labels = np.zeros_like(labels_)
    for index, value in enumerate(np.sort(np.unique(labels_))):
        new_labels[labels_ == value] = index
    return new_labels


# root_path = "data/shapes/test_data.h5"
root_path = prefix + "data/shapes/test_data.h5"

with h5py.File(root_path, "r") as hf:
    # N x 3
    test_points = np.array(hf.get("points"))

    # N x 1
    test_labels = np.array(hf.get("labels"))

    # N x 3
    test_normals = np.array(hf.get("normals"))

    # N x 1
    test_primitives = np.array(hf.get("prim"))

method_name = "parsenet_with_normals.pth"

root_path = prefix + "logs/results/{}/results/predictions.h5".format(method_name)
print(root_path)
with h5py.File(root_path, "r") as hf:
    print(list(hf.keys()))
    test_cluster_ids = np.array(hf.get("seg_id")).astype(np.int32)
    test_pred_primitives = np.array(hf.get("pred_primitives"))

prim_ids = {}
prim_ids[11] = "torus"
prim_ids[1] = "plane"
prim_ids[2] = "open-bspline"
prim_ids[3] = "cone"
prim_ids[4] = "cylinder"
prim_ids[5] = "sphere"
prim_ids[6] = "other"
prim_ids[7] = "revolution"
prim_ids[8] = "extrusion"
prim_ids[9] = "closed-bspline"

saveparameters = SaveParameters()

root_path = "/mnt/nfs/work1/kalo/gopalsharma/Projects/surfacefitting/logs_curve_fitting/outputs/{}/"

all_pred_meshes = []
all_input_points = []
all_input_labels = []
all_input_normals = []
all_cluster_ids = []
evaluation = Evaluation()
all_segments = []

os.makedirs("../logs_curve_fitting/results/{}/results/".format(method_name), exist_ok=True)

test_res = []
test_s_iou = []
test_p_iou = []
s_k_1s = []
s_k_2s = []
p_k_1s = []
p_k_2s = []
s_ks = []
p_ks = []
test_cds = []

for i in range(start, end):
    bw = 0.01
    points = test_points[i].astype(np.float32)
    normals = test_normals[i].astype(np.float32)

    labels = test_labels[i].astype(np.int32)
    labels = continuous_labels(labels)

    cluster_ids = test_cluster_ids[i].astype(np.int32)
    cluster_ids = continuous_labels(cluster_ids)
    weights = to_one_hot(cluster_ids, np.unique(cluster_ids).shape[0])

    points, normals = dataset.normalize_points(points, normals)
    torch.cuda.empty_cache()
    with torch.no_grad():
        # if_visualize=True, will give you all segments
        # if_sample=True will return segments as trimmed meshes
        # if_optimize=True will optimize the spline surface patches
        _, parameters, newer_pred_mesh = evaluation.residual_eval_mode(
            torch.from_numpy(points).cuda(),
            torch.from_numpy(normals).cuda(),
            labels,
            cluster_ids,
            test_primitives[i],
            test_pred_primitives[i],
            weights.T,
            bw,
            sample_points=True,
            if_optimize=False,
            if_visualize=True,
            epsilon=0.1)

    torch.cuda.empty_cache()
    s_iou, p_iou, _, _ = SIOU_matched_segments(
        labels,
        cluster_ids,
        test_pred_primitives[i],
        test_primitives[i],
        weights,
    )

    test_s_iou.append(s_iou)
    test_p_iou.append(p_iou)

    try:
        Points = sample_from_collection_of_mesh(newer_pred_mesh)
    except Exception as e:
        print("error in sample_from_collection_of_mesh method", e)
        continue
    cd1 = chamfer_distance_single_shape(torch.from_numpy(Points).cuda(), torch.from_numpy(points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)
    cd2 = chamfer_distance_single_shape(torch.from_numpy(points).cuda(), torch.from_numpy(Points).cuda(), sqrt=True,
                                        one_side=True, reduce=False)

    s_k_1s.append(torch.mean((cd1 < 0.01).float()).item())
    s_k_2s.append(torch.mean((cd1 < 0.02).float()).item())
    s_ks.append(torch.mean(cd1).item())
    p_k_1s.append(torch.mean((cd2 < 0.01).float()).item())
    p_k_2s.append(torch.mean((cd2 < 0.02).float()).item())
    p_ks.append(torch.mean(cd2).item())
    test_cds.append((s_ks[-1] + p_ks[-1]) / 2.0)

    results = {"sk_1": s_k_1s[-1],
               "sk_2": s_k_2s[-1],
               "sk": s_ks[-1],
               "pk_1": p_k_1s[-1],
               "pk_2": p_k_2s[-1],
               "pk": p_ks[-1],
               "cd": test_cds[-1],
               "p_iou": p_iou,
               "s_iou": s_iou}

    print(i, s_iou, p_iou, test_cds[-1])

print("Test CD: {}, Test p cover: {}, Test s cover: {}".format(np.mean(test_cds), np.mean(s_ks), np.mean(p_ks)))
print("iou seg: {}, iou prim type: {}".format(np.mean(test_s_iou), np.mean(test_p_iou)))

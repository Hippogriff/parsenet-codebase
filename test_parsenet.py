from open3d import *

import sys
import logging
import json
import os
from shutil import copyfile
import numpy as np
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.PointNet import PrimitivesEmbeddingDGCNGn
from matplotlib import pyplot as plt
from src.utils import visualize_uv_maps, visualize_fitted_surface
from src.utils import chamfer_distance
from read_config import Config
from src.utils import fit_surface_sample_points
from src.dataset_segments import Dataset
from torch.utils.data import DataLoader
from src.utils import chamfer_distance
from src.segment_loss import EmbeddingLoss
from src.segment_utils import cluster
import time
from src.segment_loss import (
    EmbeddingLoss,
    primitive_loss,
    evaluate_miou,
)
from src.utils import visualize_point_cloud_from_labels, visualize_point_cloud
from src.dataset import generator_iter
from src.mean_shift import MeanShift
from src.segment_utils import SIOU_matched_segments
from src.residual_utils import Evaluation
import time
from src.primitives import SaveParameters


# Use only one gpu.
os.environ["CUDA_VISIBLE_DEVICES"]="0"
config = Config(sys.argv[1])
if_normals = config.normals

userspace = "../"
Loss = EmbeddingLoss(margin=1.0)

if config.mode == 0:
    # Just using points for training
    model = PrimitivesEmbeddingDGCNGn(
                            embedding=True,
                            emb_size=128,
                            primitives=True,
                            num_primitives=10,
                            loss_function=Loss.triplet_loss,
                            mode=config.mode,
                            num_channels=3,
                        )
elif config.mode == 5:
    # Using points and normals for training
    model = PrimitivesEmbeddingDGCNGn(
                            embedding=True,
                            emb_size=128,
                            primitives=True,
                            num_primitives=10,
                            loss_function=Loss.triplet_loss,
                            mode=config.mode,
                            num_channels=6,
                        )    

saveparameters = SaveParameters()

model_bkp = model
model_bkp.l_permute = np.arange(10000)
model = torch.nn.DataParallel(model, device_ids=[0])
model.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}
ms = MeanShift()

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    normals=True,
    primitives=True,
    if_train_data=False
)

get_test_data = dataset.get_test(align_canonical=True, anisotropic=False, if_normal_noise=True)

loader = generator_iter(get_test_data, int(1e10))
get_test_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

os.makedirs("logs/results/{}/results/".format(config.pretrain_model_path), exist_ok=True)

evaluation = Evaluation()
alt_gpu = 0
model.eval()

iterations = 50
quantile = 0.015

model.load_state_dict(
torch.load("logs/pretrained_models/" + config.pretrain_model_path)
)
test_res = []
test_s_iou = []
test_p_iou = []
test_g_res = []
test_s_res = []
for val_b_id in range(config.num_test // config.batch_size - 1):
    t1 = time.time()
    points_, labels, normals, primitives_ = next(get_test_data)[0]

    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    normals = torch.from_numpy(normals).cuda()

    # with torch.autograd.detect_anomaly():
    with torch.no_grad():
        if if_normals:
            input = torch.cat([points, normals], 2)
            embedding, primitives_log_prob, embed_loss = model(
                input.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )
        else:
            embedding, primitives_log_prob, embed_loss = model(
                points.permute(0, 2, 1), torch.from_numpy(labels).cuda(), True
            )
            
    res_loss, parameters = evaluation.fitting_loss(
        embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
        points.to(torch.device("cuda:{}".format(alt_gpu))),
        normals.to(torch.device("cuda:{}".format(alt_gpu))),
        labels,
        primitives_,
        primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
        quantile=quantile,
        debug=False,
        iterations=iterations,
        lamb=1,
        eval=True
    )

    test_res.append(res_loss[0].item())
    if not (res_loss[1] is None):
        test_g_res.append(res_loss[1])
    if not (res_loss[2] is None):
        test_s_res.append(res_loss[2])
    test_s_iou.append(res_loss[3])
    test_p_iou.append(res_loss[4])
    print (val_b_id, res_loss)

    primitives_log_prob = torch.max(primitives_log_prob[0], 0)[1].data.cpu().numpy()
    output = saveparameters.save(parameters[0], labels[0], parameters[1], primitives_[0], primitives_log_prob, path=None, if_save=False)

    output["points"] = points_[0]
    output["normals"] = normals[0].data.cpu().numpy()

    np.save("logs/results/{}/results/normal_noise_{}_{}_{}.npy".format(config.pretrain_model_path, iterations, quantile, val_b_id), output)

print ("Res all: {}, res geom prim: {}, res spline: {}, iou seg: {}, iou prim type: {}".format(np.mean(test_res),
                                                                                               np.mean(test_g_res),
                                                                                               np.mean(test_s_res),
                                                                                               np.mean(test_s_iou),
                                                                                               np.mean(test_p_iou)))

result = {"res": np.mean(test_res),
          "res_geom": np.mean(test_g_res),
          "res_spline": np.mean(test_s_res),
          "seg_iou": np.mean(test_s_iou),
          "prim_iou": np.mean(test_p_iou)}
print (result)

np.save("logs/results/{}/results_{}_{}.npy".format(config.pretrain_model_path, iterations, quantile), result)

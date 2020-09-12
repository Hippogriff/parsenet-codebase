import os

import numpy as np
import torch.utils.data
from open3d import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

from read_config import Config
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.dataset import generator_iter
from src.dataset_segments import Dataset
from src.mean_shift import MeanShift
from src.residual_utils import Evaluation
from src.segment_loss import (
    EmbeddingLoss,
)
from src.segment_utils import SIOU_matched_segments

config = Config("config.yml")
model_name = config.pretrain_modelpath
if_normals = True
print(model_name)
userspace = "../"

Loss = EmbeddingLoss(margin=1.0)

model = PrimitivesEmbeddingDGCNGn(
    embedding=True,
    emb_size=128,
    primitives=True,
    num_primitives=10,
    loss_function=Loss.triplet_loss,
    mode=config.mode,
    num_channels=6,
    nn_nb=100,
)

model_bkp = model
model_bkp.l_permute = np.arange(10000)
model = torch.nn.DataParallel(model)
model.cuda()

model.load_state_dict(
    torch.load("../logs_curve_fitting/trained_models/" + config.pretrain_modelpath)
)

Loss = EmbeddingLoss(sample_points=1000)
split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}
ms = MeanShift()

dataset = Dataset(
    "../dataset/filtered_data/points/".format(userspace),
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    normals=True,
    primitives=True,
    if_train_data=False
)

path_to_save = "../logs_curve_fitting/outputs/{}/".format(model_name)
os.makedirs(path_to_save, exist_ok=True)
evaluation = Evaluation()

model.eval()
for quantile in [0.015]:
    get_test_data = dataset.get_test(align_canonical=True, anisotropic=False, if_normal_noise=False)

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

    test_res = []
    test_siou = []
    test_piou = []
    test_res_geometry = []
    test_res_splines = []
    for val_b_id in range(config.num_test // config.batch_size - 1):
        points_, labels, normals, primitives = next(get_test_data)[0]

        points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
        l_permute = np.arange(10000)
        normals = torch.from_numpy(normals).cuda()

        # points = points.permute(0, 2, 1)
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
            embed_loss = torch.mean(embed_loss)

        ################# Analyszing embedding #####################
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        embedding = embedding.permute(0, 2, 1)
        cluster_ids = []
        centers = []
        distances = []
        bandwidths = []

        for b in range(config.batch_size):
            num_cluster = np.unique(labels[b]).shape[0]

            quant = quantile
            while True:
                center, bandwidth, new_labels = ms.mean_shift(embedding[b], 5000, quant, 20, kernel_type="gaussian")
                if torch.unique(new_labels).shape[0] <= 50:
                    break
                else:
                    print("Number of clusters more than 50!")
                    # increase the quantile so that number of clusters increases
                    quant *= 1.2

            distances.append(center @ torch.transpose(embedding[b], 1, 0))
            new_labels = new_labels.data.cpu().numpy()
            cluster_ids.append(new_labels)
            centers.append(center)
            bandwidths.append(bandwidth)

        cluster_ids = np.stack(cluster_ids, 0).astype(np.int16)

        primitives_log_prob = torch.max(primitives_log_prob, 1)[1]
        primitives_log_prob = primitives_log_prob.data.cpu().numpy()

        for b in range(config.batch_size):
            try:
                data = np.concatenate(
                    [
                        points_[b],
                        normals[b].data.cpu().numpy(),
                        np.expand_dims(labels[b], 1),
                        np.expand_dims(cluster_ids[b], 1),
                        np.expand_dims(primitives[b], 1),
                        np.expand_dims(primitives_log_prob[b], 1),
                    ],
                    1,
                )
            except:
                import ipdb;

                ipdb.set_trace()
            np.savetxt(
                path_to_save + "gauss_{}_{}.txt".format(quantile, val_b_id * config.batch_size + b),
                data,
                fmt="%1.3f",
            )

            np.savetxt(
                path_to_save + "gauss_{}_dots_{}.npy".format(quantile, val_b_id * config.batch_size + b),
                distances[b].data.cpu().numpy(),
                fmt="%1.3f",
            )

            np.savetxt(
                path_to_save + "gauss_{}_bw_{}.npy".format(quantile, val_b_id * config.batch_size + b),
                np.array([bandwidths[b].item()]),
                fmt="%1.3f",
            )

            s_iou, p_iou, _ = SIOU_matched_segments(labels[b], cluster_ids[b], primitives_log_prob[b], primitives[b],
                                                    distances[b].T.data.cpu().numpy())

            res, parameters, pred_mesh = evaluation.residual_eval_mode(points[b], normals[b], labels[b], cluster_ids[b],
                                                                       primitives[b], primitives_log_prob[b],
                                                                       distances[b].T.data.cpu().numpy(),
                                                                       bandwidths[b].item(), sample_points=False,
                                                                       if_optimize=False)

            test_siou.append(s_iou)
            test_piou.append(p_iou)
            test_res.append(res[0])

            # create an entry when there are geometric primitives or splines.
            if res[1] > 0:
                test_res_geometry.append(res[1])
            if res[2] > 0:
                test_res_splines.append(res[2])

            print(val_b_id, s_iou, p_iou, res[0].item(), res[1], res[2])
        # print("iter: {} in time: {}".format(val_b_id, time.time() - t1))

    print("result", quantile, torch.mean(torch.stack(test_res)), np.mean(test_res_geometry), np.mean(test_res_splines),
          np.mean(test_siou), np.mean(test_piou))

import gc
import json
import logging
import os
import sys
import time
import traceback
from shutil import copyfile

import numpy as np
import torch.optim as optim
import torch.utils.data
from tensorboard_logger import configure, log_value
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from read_config import Config
from src.PointNet import PrimitivesEmbeddingDGCNGn
from src.dataset import generator_iter
from src.dataset_segments import Dataset
from src.residual_utils import Evaluation
from src.segment_loss import (
    EmbeddingLoss,
    primitive_loss,
)
from src.utils import grad_norm

np.set_printoptions(precision=3)
config = Config(sys.argv[1])
model_name = config.model_path.format(
    config.batch_size,
    config.lr,
    config.num_train,
    config.num_test,
    config.loss_weight,
    config.mode,
)

print(model_name)
configure("logs/tensorboard/{}".format(model_name), flush_secs=15)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "../logs_curve_fitting/logs/{}.log".format(model_name), mode="w"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(handler)

with open(
        "logs/configs/{}_config.json".format(model_name), "w"
) as file:
    json.dump(vars(config), file)
source_file = __file__
destination_file = "logs/scripts/{}_{}".format(
    model_name, __file__.split("/")[-1]
)
copyfile(source_file, destination_file)
if_normals = True
if_normal_noise = True

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)

model = PrimitivesEmbeddingDGCNGn(
    embedding=True,
    emb_size=128,
    primitives=True,
    num_primitives=10,
    loss_function=Loss.triplet_loss,
    mode=config.mode,
    num_channels=6,
)

# device = torch.device("cuda:0")
model_bkp = model
model_bkp.l_permute = np.arange(7000)
model = torch.nn.DataParallel(model)

model.load_state_dict(
    torch.load("logs/pretrained_models/" + config.pretrain_model_path)
)
model.cuda()

# Do not train the encoder weights to save gpu memory.
for key, values in model.named_parameters():
    if key.startswith("module.encoder"):
        values.requires_grad = True
    else:
        values.requires_grad = True

evaluation = Evaluation()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    primitives=True,
    normals=True,
    if_train_data=True
)

get_train_data = dataset.get_train(
    randomize=True,
    augment=False,
    align_canonical=True,
    anisotropic=False,
    if_normal_noise=if_normal_noise,
)
get_val_data = dataset.get_val(
    align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise
)

optimizer = optim.Adam(model.parameters(), lr=config.lr)
optimizer.load_state_dict(torch.load("logs/pretrained_models/" +
                                     config.pretrain_model_path.split(".")[0] + "_optimizer.pth"))

os.makedirs("logs/trained_models/{}/".format(model_name), exist_ok=True)
loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

loader = generator_iter(get_val_data, int(1e10))
get_val_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=0,
        pin_memory=False,
    )
)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10, verbose=True, min_lr=1e-4
)

model_bkp.triplet_loss = Loss.triplet_loss
prev_test_loss = 1e4
print("started training!")

if torch.cuda.device_count() > 1:
    alt_gpu = 1
else:
    alt_gpu = 0

lamb = 0.1
# no updates to the bn
model.eval()
for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_res_losses = []
    train_res_geom_losses = []
    train_res_spline_losses = []
    train_iou = []
    train_losses = []
    train_seg_iou = []
    n_loss = None
    num_iter = 5
    # for train_b_id in range(config.num_train // config.batch_size // num_iter):
    for train_b_id in range(100000):
        optimizer.zero_grad()
        losses = 0
        ious = 0
        seg_ious = 0
        p_losses = 0
        embed_losses = 0
        res_g_losses = []
        res_s_losses = []
        res_losses = 0
        torch.cuda.empty_cache()
        t1 = time.time()
        mistake = False

        for count_iteration in range(num_iter):
            gc.collect()
            while True:
                points, labels, normals, primitives_ = next(get_train_data)[0]
                # Take only training dataset with no. segments more than 2
                break
                if np.unique(labels).shape[0] < 3:
                    continue
                else:
                    break
            l = np.arange(10000)
            np.random.shuffle(l)
            rand_num_points = 8000
            l = l[0:rand_num_points]
            points = points[:, l]
            labels = labels[:, l]
            normals = normals[:, l]
            primitives_ = primitives_[:, l]
            points = torch.from_numpy(points).cuda()
            normals = torch.from_numpy(normals).cuda()

            # TO make sure that the network doesn't compute the gradient w.r.t
            # these points.
            points.requires_grad = False
            normals.requires_grad = False
            primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()
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
            p_loss = primitive_loss(primitives_log_prob, primitives)
            torch.cuda.empty_cache()
            try:
                res_loss, _ = evaluation.fitting_loss(
                    embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                    points.to(torch.device("cuda:{}".format(alt_gpu))),
                    normals.to(torch.device("cuda:{}".format(alt_gpu))),
                    labels,
                    primitives_,
                    primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                    quantile=0.025,
                    debug=False,
                    iterations=10,
                    lamb=lamb
                )
                res_loss[0] = res_loss[0].to(torch.device("cuda:0"))
            except Exception as weird_except:
                import ipdb;

                ipdb.set_trace()
                traceback.print_exc()
                loss = embed_loss
                loss.backward()
                if grad_norm(model):
                    optimizer.zero_grad()
                    print("grad norm is nan or inf!")
                torch.cuda.empty_cache()
                print(weird_except)
                print("exception in training")
                mistake = True
                break
            s_iou, iou = res_loss[3:]

            loss = embed_loss + p_loss + 1 * res_loss[0]
            res_losses += res_loss[0].item() / num_iter

            if not (res_loss[1] is None):
                res_g_losses.append(res_loss[1])

            if not (res_loss[2] is None):
                res_s_losses.append(res_loss[2])

            seg_ious += s_iou / num_iter

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter
            torch.cuda.empty_cache()
            try:
                loss.backward()
            except:
                import ipdb;

                ipdb.set_trace()

        if mistake:
            continue
        # Avoid zero entries
        if len(res_g_losses) > 0:
            res_g_losses = np.mean(res_g_losses)
        else:
            res_g_losses = 1e-3
        if len(res_s_losses) > 0:
            res_s_losses = np.mean(res_s_losses)
        else:
            res_s_losses = 9e-3
        optimizer.step()
        # print ("train: ", train_b_id, time.time() - t1, res_loss.item(), loss.item(), embed_loss.item(), p_loss.item())

        del res_loss, loss, embed_loss, p_loss
        if train_b_id > 0 and (train_b_id % 2000 == 0):
            torch.save(
                model.state_dict(),
                "logs/trained_models/{}_{}.pth".format((train_b_id // 2000) * (1 + e), model_name),
            )
            torch.save(
                optimizer.state_dict(),
                "logs/trained_models/{}_{}_optimizer.pth".format((train_b_id // 2000) * (1 + e), model_name),
            )
        torch.cuda.empty_cache()
        train_iou.append(ious)
        train_seg_iou.append(seg_ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        train_res_losses.append(res_losses)
        train_res_geom_losses.append(res_g_losses)
        train_res_spline_losses.append(res_s_losses)

        log_value("iou", iou, train_b_id + e * (config.num_train // config.batch_size // num_iter))
        log_value(
            "embed_loss",
            embed_losses,
            train_b_id + e * (config.num_train // config.batch_size // num_iter),
        )
        log_value(
            "res_loss",
            res_losses,
            train_b_id + e * (config.num_train // config.batch_size // num_iter),
        )
        log_value(
            "res_g_loss",
            res_g_losses,
            train_b_id + e * (config.num_train // config.batch_size // num_iter),
        )
        log_value(
            "res_s_loss",
            res_s_losses,
            train_b_id + e * (config.num_train // config.batch_size // num_iter),
        )
        log_value("seg_iou",
                  seg_ious,
                  train_b_id + e * (config.num_train // config.batch_size // num_iter), )
    test_emb_losses = []
    test_prim_losses = []
    test_losses = []
    test_res_losses = []
    test_res_geom_losses = []
    test_res_spline_losses = []
    test_iou = []
    test_seg_iou = []

    model.eval()
    score = []
    torch.cuda.empty_cache()
    for val_b_id in range(config.num_test // config.batch_size - 1):
        t1 = time.time()
        points, labels, normals, primitives_ = next(get_val_data)[0]

        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:8000]
        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives_ = primitives_[:, l]
        points = torch.from_numpy(points).cuda()
        primitives = torch.from_numpy(primitives_.astype(np.int64)).cuda()
        normals = torch.from_numpy(normals).cuda()
        mistake = False

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

            try:
                res_loss, _ = evaluation.fitting_loss(
                    embedding.permute(0, 2, 1).to(torch.device("cuda:{}".format(alt_gpu))),
                    points.to(torch.device("cuda:{}".format(alt_gpu))),
                    normals.to(torch.device("cuda:{}".format(alt_gpu))),
                    labels,
                    primitives_,
                    primitives_log_prob.to(torch.device("cuda:{}".format(alt_gpu))),
                    quantile=0.025,
                    iterations=10,
                    lamb=1.0,
                    debug=False,
                    eval=True,
                )
            except Exception:
                traceback.print_exc()
                loss = embed_loss
                loss.backward()
                print("some exception in while testing")
                continue

        s_iou, iou = res_loss[3:]
        res_loss = res_loss[0:3]
        res_loss[0] = res_loss[0].to(torch.device("cuda:0"))

        test_res_losses.append(res_loss[0].item())
        if not (res_loss[1] is None):
            test_res_geom_losses.append(res_loss[1])
        if not (res_loss[2] is None):
            test_res_spline_losses.append(res_loss[2])

        embed_loss = torch.mean(embed_loss)
        p_loss = primitive_loss(primitives_log_prob, primitives)
        loss = embed_loss + p_loss

        print("test: ", val_b_id, time.time() - t1)
        test_iou.append(iou)
        test_seg_iou.append(s_iou)

        test_prim_losses.append(p_loss.data.cpu().numpy())
        test_emb_losses.append(embed_loss.data.cpu().numpy())
        test_losses.append(loss.data.cpu().numpy())
        torch.cuda.empty_cache()

    print("\n")
    logger.info(
        "Epoch: {}/{} => TrL:{}, TsL:{}, TrP:{}, TsP:{}, TrE:{}, TsE:{}, TrI:{}, TsI:{}".format(
            e,
            config.epochs,
            np.mean(train_losses),
            np.mean(test_losses),
            np.mean(train_prim_losses),
            np.mean(test_prim_losses),
            np.mean(train_emb_losses),
            np.mean(test_emb_losses),
            np.mean(train_iou),
            np.mean(test_iou),
        )
    )
    log_value("train iou", np.mean(train_iou), e)
    log_value("test iou", np.mean(test_iou), e)

    log_value("train emb loss", np.mean(train_emb_losses), e)
    log_value("test emb loss", np.mean(test_emb_losses), e)

    log_value("train res loss", np.mean(train_res_losses), e)
    log_value("test res loss", np.mean(train_res_losses), e)

    log_value("train geom res loss", np.mean(train_res_geom_losses), e)
    log_value("test geom res loss", np.mean(test_res_geom_losses), e)

    log_value("train spline res loss", np.mean(train_res_spline_losses), e)
    log_value("test spline res loss", np.mean(test_res_spline_losses), e)

    log_value("train seg iou", np.mean(train_seg_iou), e)
    log_value("test seg iou", np.mean(test_seg_iou), e)

    scheduler.step(np.mean(test_res_losses))
    if prev_test_loss > np.mean(test_res_losses):
        logger.info("improvement, saving model at epoch: {}".format(e))
        prev_test_loss = np.mean(test_res_losses)
        torch.save(
            model.state_dict(),
            "logs/trained_models/{}.pth".format(model_name),
        )
        torch.save(
            optimizer.state_dict(),
            "logs/trained_models/{}_optimizer.pth".format(model_name),
        )

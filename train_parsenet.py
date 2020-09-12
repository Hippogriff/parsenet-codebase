"""
This scrip trains model to predict per point primitive type.
"""
import json
import logging
import os
import sys
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
from src.segment_loss import (
    EmbeddingLoss,
    evaluate_miou,
    primitive_loss
)

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
configure("logs/tensorboard/{}".format(model_name), flush_secs=5)

userspace = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
file_handler = logging.FileHandler(
    "logs/logs/{}.log".format(model_name), mode="w"
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
if_normals = config.normals
if_normal_noise = True

Loss = EmbeddingLoss(margin=1.0, if_mean_shift=False)
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

model_bkp = model
model_bkp.l_permute = np.arange(7000)
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
model.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = Dataset(
    config.batch_size,
    config.num_train,
    config.num_val,
    config.num_test,
    primitives=True,
    normals=True,
)

get_train_data = dataset.get_train(
    randomize=True, augment=True, align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise
)
get_val_data = dataset.get_val(align_canonical=True, anisotropic=False, if_normal_noise=if_normal_noise)
optimizer = optim.Adam(model.parameters(), lr=config.lr)

loader = generator_iter(get_train_data, int(1e10))
get_train_data = iter(
    DataLoader(
        loader,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=2,
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
        num_workers=2,
        pin_memory=False,
    )
)

scheduler = ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, verbose=True, min_lr=1e-4
)

model_bkp.triplet_loss = Loss.triplet_loss
prev_test_loss = 1e4

for e in range(config.epochs):
    train_emb_losses = []
    train_prim_losses = []
    train_iou = []
    train_losses = []
    model.train()

    # this is used for gradient accumulation because of small gpu memory.
    num_iter = 3
    for train_b_id in range(config.num_train // config.batch_size):
        optimizer.zero_grad()
        losses = 0
        ious = 0
        p_losses = 0
        embed_losses = 0
        torch.cuda.empty_cache()
        for _ in range(num_iter):
            points, labels, normals, primitives = next(get_train_data)[0]
            l = np.arange(10000)
            np.random.shuffle(l)
            # randomly sub-sampling points to increase robustness to density and
            # saving gpu memory
            rand_num_points = 7000
            l = l[0:rand_num_points]
            points = points[:, l]
            labels = labels[:, l]
            normals = normals[:, l]
            primitives = primitives[:, l]
            points = torch.from_numpy(points).cuda()
            normals = torch.from_numpy(normals).cuda()

            primitives = torch.from_numpy(primitives.astype(np.int64)).cuda()
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
            iou = evaluate_miou(
                primitives.data.cpu().numpy(),
                primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
            )
            loss = embed_loss + p_loss
            loss.backward()

            losses += loss.data.cpu().numpy() / num_iter
            p_losses += p_loss.data.cpu().numpy() / num_iter
            ious += iou / num_iter
            embed_losses += embed_loss.data.cpu().numpy() / num_iter

        optimizer.step()
        train_iou.append(ious)
        train_losses.append(losses)
        train_prim_losses.append(p_losses)
        train_emb_losses.append(embed_losses)
        print(
            "\rEpoch: {} iter: {}, prim loss: {}, emb loss: {}, iou: {}".format(
                e, train_b_id, p_loss, embed_losses, iou
            ),
            end="",
        )
        log_value("iou", iou, train_b_id + e * (config.num_train // config.batch_size))
        log_value(
            "embed_loss",
            embed_losses,
            train_b_id + e * (config.num_train // config.batch_size),
        )

    test_emb_losses = []
    test_prim_losses = []
    test_losses = []
    test_iou = []
    model.eval()

    for val_b_id in range(config.num_test // config.batch_size - 1):
        points, labels, normals, primitives = next(get_val_data)[0]
        l = np.arange(10000)
        np.random.shuffle(l)
        l = l[0:7000]
        points = points[:, l]
        labels = labels[:, l]
        normals = normals[:, l]
        primitives = primitives[:, l]
        points = torch.from_numpy(points).cuda()
        primitives = torch.from_numpy(primitives.astype(np.int64)).cuda()
        normals = torch.from_numpy(normals).cuda()
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
        p_loss = primitive_loss(primitives_log_prob, primitives)
        loss = embed_loss + p_loss
        iou = evaluate_miou(
            primitives.data.cpu().numpy(),
            primitives_log_prob.permute(0, 2, 1).data.cpu().numpy(),
        )
        test_iou.append(iou)
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

    scheduler.step(np.mean(test_emb_losses))
    if prev_test_loss > np.mean(test_emb_losses):
        logger.info("improvement, saving model at epoch: {}".format(e))
        prev_test_loss = np.mean(test_emb_losses)
        torch.save(
            model.state_dict(),
            "logs/trained_models/{}.pth".format(model_name),
        )
        torch.save(
            optimizer.state_dict(),
            "logs/trained_models/{}_optimizer.pth".format(model_name),
        )

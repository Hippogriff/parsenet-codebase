import os

import numpy as np
import open3d
import torch.utils.data
from open3d import *
from torch.autograd import Variable
from torch.utils.data import DataLoader

from read_config import Config
from src.VisUtils import tessalate_points
from src.dataset import DataSetControlPoints
from src.dataset import generator_iter
from src.loss import control_points_permute_reg_loss
from src.loss import laplacian_loss
from src.loss import (
    uniform_knot_bspline,
    atlasnet_stretching_loss_edge_length,
    spline_reconstruction_loss,
)
from src.model import DGCNNControlPoints
from src.primitive_forward import optimize_open_spline

config = Config("config_controlpoints.yml")
model_name = config.pretrain_modelpath

print(model_name)
userspace = ".."

control_decoder = DGCNNControlPoints(20, num_points=10, mode=config.mode)
control_decoder = torch.nn.DataParallel(control_decoder)
control_decoder.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": 1000}

dataset = DataSetControlPoints(
    config.dataset_path.format(userspace),
    config.batch_size,
    2000,
    splits=split_dict,
    size_v=config.grid_size,
    size_u=config.grid_size,
    indices_file_path="{}/dataset/all_splines_2/open_small.txt".format(userspace),
)

nu, nv = uniform_knot_bspline(20, 20, 3, 3, 30)
nu = torch.from_numpy(nu.astype(np.float32)).cuda()
nv = torch.from_numpy(nv.astype(np.float32)).cuda()

nu_3, nv_3 = uniform_knot_bspline(30, 30, 3, 3, 50)
nu_3 = torch.from_numpy(nu_3.astype(np.float32)).cuda()
nv_3 = torch.from_numpy(nv_3.astype(np.float32)).cuda()

config.pretrain_modelpath = "open_splines_k_10_val_1_cd_aug_rand_points_400-2k-mode_0_700_0.9_bt_40_lr_0.001_trsz_32000_tsz_3000_wght_0.9.pth"

# We want to gather the regular grid points for tesellation
align_canonical = True
anisotropic = True
if_augment = True
if_rand_points = False
config.num_points = 700

get_test_data = dataset.load_test_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic,
    if_augment=if_augment)
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

control_decoder.load_state_dict(
    torch.load("../logs_curve_fitting/trained_models/" + config.pretrain_modelpath)
)
distances = []
test_reg = []
test_cd = []
test_str = []
control_decoder.eval()
count = 0
test_lap = []
if_optimize = False

os.makedirs(
    "logs/results/{}/".format(config.pretrain_modelpath),
    exist_ok=True,
)

for val_b_id in range(config.num_test // config.batch_size - 1):
    print(val_b_id)
    points_, parameters, control_points, scales, RS = next(get_test_data)[0]
    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    reg_points = np.copy(points_[:, 0:400])
    points_ = points_[:, 400:]
    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points.permute(0, 2, 1)

    with torch.no_grad():
        if if_rand_points:
            num_points = config.num_points + np.random.choice(np.arange(-200, 200), 1)[0]
        else:
            num_points = config.num_points
        L = np.arange(points.shape[2])
        np.random.shuffle(L)
        new_points = points[:, :, L[0:num_points]]

        output = control_decoder(new_points)

    for b in range(config.batch_size):
        # re-alinging back to original orientation for better comparison
        if anisotropic:
            s = torch.from_numpy(scales[b].astype(np.float32)).cuda()
            output[b] = output[b] * s.reshape(1, 3) / torch.max(s)
            points[b] = points[b] * s.reshape(3, 1) / torch.max(s)
            control_points[b] = (
                    control_points[b] * s.reshape(1, 1, 3) / torch.max(s)
            )
            reg_points[b] = (
                    reg_points[b] * scales[b].reshape(1, 3) / np.max(scales[b])
            )

    # Chamfer Distance loss, between predicted and GT surfaces
    cd, reconstructed_points = spline_reconstruction_loss(
        nu, nv, output, points, config, sqrt=True
    )

    if if_optimize:
        new_points = optimize_open_spline(reconstructed_points, points.permute(0, 2, 1))
        optimized_points = new_points.clone()
        cd, reconstructed_points = spline_reconstruction_loss(nu_3, nv_3, new_points, points, config, sqrt=True)

    l_reg, permute_cp = control_points_permute_reg_loss(
        output, control_points, config.grid_size
    )
    l_stretch = atlasnet_stretching_loss_edge_length(output, 20)
    laplac_loss = laplacian_loss(
        output.reshape((config.batch_size, config.grid_size, config.grid_size, 3)),
        permute_cp,
        dist_type="l2",
    )

    test_reg.append(l_reg.data.cpu().numpy())
    test_cd.append(cd.data.cpu().numpy())
    test_str.append(l_stretch.data.cpu().numpy())

    test_lap.append(laplac_loss.data.cpu().numpy())
    pred_meshes = []
    gt_meshes = []
    reconstructed_points = reconstructed_points.data.cpu().numpy()

    # Save the predictions.
    for b in range(config.batch_size):
        # re-alinging back to original orientation for better comparison
        # if anisotropic:
        #     reconstructed_points[b] = reconstructed_points[b] * scales[b].reshape(1, 3) / np.max(scales[b])
        optimized_points = optimized_points.data.cpu().numpy()

        if align_canonical:
            new_points = np.linalg.inv(RS[b]) @ reconstructed_points[b].T
            reconstructed_points[b] = new_points.T

            new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
            reg_points[b] = new_points.T

            new_points = np.linalg.inv(RS[b]) @ optimized_points[b].T
            optimized_points[b] = new_points.T

        pred_mesh = tessalate_points(reconstructed_points[b], 50, 50)
        pred_mesh.paint_uniform_color([1, 0, 0])

        gt_mesh = tessalate_points(reg_points[b], 20, 20)
        optim_mesh = tessalate_points(optimized_points[0], 30, 30)
        # draw_geometries([pred_mesh, gt_mesh])
        # grid_meshes_lists_visulation([[gt_mesh, pred_mesh, optim_mesh]], viz=True)

        open3d.io.write_triangle_mesh(
            "../logs/results/{}/gt_{}.ply".format(
                config.pretrain_modelpath, val_b_id
            ),
            gt_mesh,
        )
        open3d.io.write_triangle_mesh(
            "../logs/results/{}/pred_{}.ply".format(
                config.pretrain_modelpath, val_b_id
            ),
            pred_mesh,
        )
        open3d.io.write_triangle_mesh(
            "../logs/results/{}/optim_{}.ply".format(
                config.pretrain_modelpath, val_b_id
            ),
            optim_mesh,
        )

        # np.save("../logs/results/{}/points_{}.npy".format(
        #         config.pretrain_modelpath, val_b_id), points[0].T.data.cpu().numpy())
        # np.save("../logs/results/{}/gt_cpts_{}.npy".format(
        #         config.pretrain_modelpath, val_b_id), control_points[b].T.data.cpu().numpy())

        # # Save output control points
        # np.save("../logs/results/{}/pred_cpts_{}.npy".format(
        #         config.pretrain_modelpath, val_b_id), output[b].T.data.cpu().numpy())
        # # Save outimized gridded points.
        # np.save("../logs/results/{}/optim_cpts_{}.npy".format(
        #         config.pretrain_modelpath, val_b_id), optimized_points[0])                        

results = {}
results["test_reg"] = str(np.mean(test_reg))
results["test_cd"] = str(np.mean(test_cd))
results["test_stretch"] = str(np.mean(test_str))
results["test_lap"] = str(np.mean(test_lap))
print(results)
print(
    "Test Reg Loss: {}, Test CD Loss: {}, Test Stretch: {}, Test Lap: {}".format(
        np.mean(test_reg), np.mean(test_cd), np.mean(test_str), np.mean(test_lap)
    )
)

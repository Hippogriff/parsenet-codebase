import open3d
import sys
import logging
import json
import os
from shutil import copyfile
import numpy as np
import torch.optim as optim
from src.curve_utils import fit_surface
from src.utils import visualize_point_cloud
from src.test_utils import test
from src.loss import control_points_permute_reg_loss
import torch.utils.data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.dataset import DataSetControlPointsPoisson
from src.model import DGCNNControlPoints
from matplotlib import pyplot as plt
from src.utils import visualize_uv_maps, visualize_fitted_surface
from src.utils import chamfer_distance
from read_config import Config
from src.utils import fit_surface_sample_points
from src.dataset import generator_iter
from torch.utils.data import DataLoader
from src.utils import chamfer_distance
from src.loss import (
    basis_function_one,
    uniform_knot_bspline,
    spline_reconstruction_loss,
)
from src.fitting_utils import sample_points_from_control_points_
from src.utils import chamfer_distance
from src.VisUtils import tessalate_points
import open3d
from open3d import *
import json
from src.loss import laplacian_loss
from src.VisUtils import grid_meshes_lists_visulation
from src.fitting_utils import up_sample_points_torch_in_range
from src.primitive_forward import optimize_close_spline
from src.utils import chamfer_distance_single_shape
from src.VisUtils import grid_meshes_lists_visulation

config = Config(sys.argv[1])

userspace = ".."
print (config.mode)
control_decoder = DGCNNControlPoints(20, num_points=10, mode=config.mode)
control_decoder = torch.nn.DataParallel(control_decoder)
control_decoder.cuda()
config.batch_size = 1
split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = DataSetControlPointsPoisson(
    path=config.dataset_path,
    batch_size=config.batch_size,
    splits=split_dict,
    size_v=config.grid_size,
    size_u=config.grid_size,
    closed=True
)

nu, nv = uniform_knot_bspline(20, 20, 3, 3, 30)
nu = torch.from_numpy(nu.astype(np.float32)).cuda()
nv = torch.from_numpy(nv.astype(np.float32)).cuda()

nu_3, nv_3 = uniform_knot_bspline(31, 30, 3, 3, 50)
nu_3 = torch.from_numpy(nu_3.astype(np.float32)).cuda()
nv_3 = torch.from_numpy(nv_3.astype(np.float32)).cuda()

# We want to gather the regular grid points for tesellation
align_canonical = True
anisotropic = True
if_augmentation = False
if_rand_num_points = False
if_upsample = False
visualize = True
if_optimize = True

os.makedirs(
    "logs/results/{}/".format(config.pretrain_model_path),
    exist_ok=True,
)

config.num_points = 700

get_test_data = dataset.load_test_data(
    if_regular_points=True, align_canonical=align_canonical, anisotropic=anisotropic, if_augment=if_augmentation
)
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
    torch.load("logs/pretrained_models/" + config.pretrain_model_path)
)

distances = []
test_reg = []
test_cd = []
test_str = []

count = 0
test_lap = []

control_decoder.eval()

for val_b_id in range(config.num_test // config.batch_size - 1):
    points_, parameters, control_points, scales, RS = next(get_test_data)[0]

    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points.permute(0, 2, 1)

    if if_rand_num_points:
        rand_num_points = config.num_points + np.random.choice(np.arange(-200, 200), 1)[0]
    else:
        rand_num_points = config.num_points

    with torch.no_grad():
        L = np.arange(points.shape[2])
        np.random.shuffle(L)
        new_points = points[:, :, L[0:rand_num_points]]

        if if_upsample:
            new_points = up_sample_points_torch_in_range(new_points[0].permute(1, 0), 1200, 1800).permute(1, 0)
            new_points = torch.unsqueeze(new_points, 0)

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

    # Chamfer Distance loss, between predicted and GT surfaces
    cd, reconstructed_points = spline_reconstruction_loss(
        nu, nv, output, points, config, sqrt=True
    )

    temp = reconstructed_points[b].reshape((30, 30, 3))
    temp = torch.cat([temp, temp[0:1]], 0)
    temp = torch.unsqueeze(temp, 0)

    if if_optimize:
        new_points = optimize_close_spline(temp, points.permute(0, 2, 1))
        optimized_points = new_points.clone()
        cd = chamfer_distance_single_shape(new_points[0], points[0].permute(1, 0), sqrt=True)

    l_reg, permute_cp = control_points_permute_reg_loss(
        output, control_points, config.grid_size
    )
    laplac_loss = laplacian_loss(
        output.reshape((config.batch_size, config.grid_size, config.grid_size, 3)),
        permute_cp,
        dist_type="l2",
    )

    test_reg.append(l_reg.data.cpu().numpy())
    test_cd.append(cd.data.cpu().numpy())
    test_lap.append(laplac_loss.data.cpu().numpy())

    print (val_b_id, cd.item())
    if visualize:
        pred_meshes = []
        gt_meshes = []
        reconstructed_points = reconstructed_points.data.cpu().numpy()
        control_points = control_points.reshape((config.batch_size, 400, 3))
        for b in range(config.batch_size):
            temp = reconstructed_points[b].reshape((30, 30, 3))
            temp = np.concatenate([temp, temp[0:1]], 0)
            pred_mesh = tessalate_points(temp, 31, 30)
            pred_mesh.paint_uniform_color([1, 0.0, 0])
            
            gt_points = sample_points_from_control_points_(nu, nv, control_points[b:b+1], 1).data.cpu().numpy()
            temp = gt_points[b].reshape((30, 30, 3))
            gt_points = np.concatenate([temp, temp[0:1]], 0)
            gt_mesh = tessalate_points(gt_points, 31, 30)

            temp = optimized_points[0].reshape((31, 30, 3))
            optimized_points = torch.cat([temp, temp[0:1]], 0)
            optim_mesh = tessalate_points(optimized_points.data.cpu().numpy(), 32, 30)

            open3d.io.write_triangle_mesh(
                "logs/results/{}/gt_{}.ply".format(
                    config.pretrain_model_path, val_b_id * config.batch_size + b
                ),
                gt_mesh,
            )
            open3d.io.write_triangle_mesh(
                "logs/results/{}/pred_{}.ply".format(
                    config.pretrain_model_path, val_b_id * config.batch_size + b
                ),
                pred_mesh,
            )
            open3d.io.write_triangle_mesh(
                "logs/results/{}/optim_{}.ply".format(
                    config.pretrain_model_path, val_b_id * config.batch_size + b
                ),
                optim_mesh,
            )

results = {}
results["test_reg"] = str(np.mean(test_reg))
results["test_cd"] = str(np.mean(test_cd))
results["test_lap"] = str(np.mean(test_lap))

print(
    "Test Reg Loss: {}, Test CD Loss: {},  Test Lap: {}".format(
        np.mean(test_reg), np.mean(test_cd), np.mean(test_lap)
    )
)

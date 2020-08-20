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
from src.utils import chamfer_distance
from src.VisUtils import tessalate_points
import open3d
from open3d import *
import json
from src.loss import laplacian_loss
from src.utils import draw_geometries
from src.fitting_utils import up_sample_points_torch_in_range
from src.primitive_forward import optimize_open_spline, optimize_open_spline_kronecker
from src.primitive_forward import optimize_open_spline
from src.VisUtils import grid_meshes_lists_visulation
from src.fitting_utils import sample_points_from_control_points_


config = Config(sys.argv[1])

control_decoder = DGCNNControlPoints(20, num_points=10, mode=config.mode)
control_decoder = torch.nn.DataParallel(control_decoder)
control_decoder.cuda()

split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = DataSetControlPointsPoisson(
    config.dataset_path,
    config.batch_size,
    splits=split_dict,
    size_v=config.grid_size,
    size_u=config.grid_size)

nu, nv = uniform_knot_bspline(20, 20, 3, 3, 30)
nu = torch.from_numpy(nu.astype(np.float32)).cuda()
nv = torch.from_numpy(nv.astype(np.float32)).cuda()

nu_3, nv_3 = uniform_knot_bspline(30, 30, 3, 3, 50)
nu_3 = torch.from_numpy(nu_3.astype(np.float32)).cuda()
nv_3 = torch.from_numpy(nv_3.astype(np.float32)).cuda()
    
align_canonical = True
anisotropic = True
if_augment = False
if_rand_points = False
if_optimize = False
if_save_meshes = True
if_upsample = False


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
    torch.load("logs/pretrained_models/" + config.pretrain_model_path)
)
os.makedirs(
    "logs/results/{}/".format(config.pretrain_model_path),
    exist_ok=True,
)

distances = []
test_reg = []
test_cd = []
test_str = []
test_lap = []
config.num_points = 700
control_decoder.eval()
for val_b_id in range(config.num_test // config.batch_size - 2):
    points_, parameters, control_points, scales, RS = next(get_test_data)[0]
    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    points_ = points_
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
        
        if if_upsample:
            new_points = up_sample_points_torch_in_range(new_points[0].permute(1, 0), 800, 1200).permute(1, 0)
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

    if if_optimize:
        new_points = optimize_open_spline(reconstructed_points, points.permute(0, 2, 1))

        cd,  optimized_points = spline_reconstruction_loss(nu_3, nv_3, new_points, points, config, sqrt=True)
        optimized_points = optimized_points.data.cpu().numpy()
        
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
    print (val_b_id)
    if if_save_meshes:
        reconstructed_points = reconstructed_points.data.cpu().numpy()
        reg_points = sample_points_from_control_points_(nu, nv, control_points, config.batch_size,
                                                        input_size_u=20, input_size_v=20).data.cpu().numpy()

        # Save the predictions.
        for b in range(config.batch_size):
            if align_canonical:
                # to bring back into cannonical orientation.
                new_points = np.linalg.inv(RS[b]) @ reconstructed_points[b].T
                reconstructed_points[b] = new_points.T

                new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
                reg_points[b] = new_points.T
                
                if if_optimize:
                    new_points = np.linalg.inv(RS[b]) @ optimized_points[b].T
                    optimized_points[b] = new_points.T

            pred_mesh = tessalate_points(reconstructed_points[b], 30, 30)
            pred_mesh.paint_uniform_color([1, 0, 0])

            gt_mesh = tessalate_points(reg_points[b], 30, 30)

            open3d.io.write_triangle_mesh(
                "logs/results/{}/gt_{}.ply".format(
                    config.pretrain_model_path, val_b_id
                ),
                gt_mesh,
            )
            open3d.io.write_triangle_mesh(
                "logs/results/{}/pred_{}.ply".format(
                    config.pretrain_model_path, val_b_id
                ),
                pred_mesh,
            )

            if if_optimize:
                optim_mesh = tessalate_points(optimized_points[b], 50, 50)
                open3d.io.write_triangle_mesh(
                    "logs/results/{}/optim_{}.ply".format(
                        config.pretrain_model_path, val_b_id
                    ),
                optim_mesh,
                )

results = {}
results["test_reg"] = str(np.mean(test_reg))
results["test_cd"] = str(np.mean(test_cd))
results["test_lap"] = str(np.mean(test_lap))
print (results)
print(
    "Test Reg Loss: {}, Test CD Loss: {},  Test Lap: {}".format(
        np.mean(test_reg), np.mean(test_cd), np.mean(test_lap)
    )
)

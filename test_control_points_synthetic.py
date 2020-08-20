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
from src.synthetic_dataset import DataSet
from src.PointNet import ControlPointPredict
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
    atlasnet_stretching_loss_edge_length,
    spline_reconstruction_loss,
)
from src.utils import chamfer_distance
from src.VisUtils import tessalate_points
import open3d
from src.approximation import generate_bezier_surface_on_grid, uniform_knot_bspline

config = Config("config_controlpoints.yml")
model_name = config.pretrain_modelpath

print(model_name)
userspace = "/mnt/gypsum/mnt/nfs/work1/kalo/gopalsharma/Projects/surfacefitting"

control_decoder = DGCNNControlPoints(3)
control_decoder.cuda()
control_decoder.load_state_dict(
    torch.load("../logs_curve_fitting/trained_models/" + config.pretrain_modelpath)
)
split_dict = {"train": config.num_train, "val": config.num_val, "test": config.num_test}

dataset = DataSet(
    path="../logs_curve_fitting/dataset/mix_v2.h5",
    train_size=config.num_train,
    test_size=config.num_test,
    val_size=config.num_val,
)
get_train_data = dataset.get_train_data(batch_size=config.batch_size)
get_test_data = dataset.get_test_data(batch_size=config.batch_size)

nu, nv = uniform_knot_bspline(3, 3, 2, 2, grid_size=30)

# We want to gather the regular grid points for tesellation
distances = []
test_reg = []
test_cd = []

control_decoder.eval()
count = 0

os.makedirs(
    "../logs_curve_fitting/results/{}/".format(config.pretrain_modelpath), exist_ok=True
)

for val_b_id in range(config.num_test // config.batch_size - 1):
    points_, control_points = next(get_test_data)
    control_points = Variable(
        torch.from_numpy(control_points.astype(np.float32))
    ).cuda()

    points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points.permute(0, 2, 1)
    with torch.no_grad():
        output = control_decoder(points)
    reconstructed_points = []

    for b in range(config.batch_size):
        reconstructed_points.append(
            generate_bezier_surface_on_grid(
                output.data.cpu().numpy()[b].reshape((3, 3, 3)), nu, nv
            )
        )
    reconstructed_points = np.stack(reconstructed_points, 0)
    reconstructed_points = reconstructed_points.reshape((config.batch_size, 900, 3))

    # import ipdb; ipdb.set_trace()

    l_reg, _ = control_points_permute_reg_loss(output, control_points, config.grid_size)
    cd = chamfer_distance(reconstructed_points, points_)
    test_reg.append(l_reg.data.cpu().numpy())
    test_cd.append(cd.data.cpu().numpy())
    pred_meshes = []
    gt_meshes = []
    # import ipdb; ipdb.set_trace()
    from open3d import *

    for b in range(config.batch_size):
        pred_mesh = tessalate_points(reconstructed_points[b], 30, 30)
        gt_mesh = tessalate_points(points_[b], 30, 30)

        # draw_geometries([pred_mesh, gt_mesh])

        open3d.io.write_triangle_mesh(
            "../logs_curve_fitting/results/{}/gt_{}.ply".format(
                config.pretrain_modelpath, count
            ),
            gt_mesh,
        )
        open3d.io.write_triangle_mesh(
            "../logs_curve_fitting/results/{}/pred_{}.ply".format(
                config.pretrain_modelpath, count
            ),
            pred_mesh,
        )
        count = count + 1

print(
    "Test Reg Loss: {}, Test CD Loss: {}, Test Stretch: {}".format(
        np.mean(test_reg), np.mean(test_cd), 0
    )
)

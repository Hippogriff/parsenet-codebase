import numpy as np
from open3d import *
from matplotlib import pyplot as plt
from src.curve_utils import fit_surface
import lap
from torch.autograd.variable import Variable
import torch
import os
import open3d
import open3d as o3d
from src.guard import guard_exp, guard_sqrt

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
draw_geometries = o3d.visualization.draw_geometries


def get_rotation_matrix(theta):
    R=np.array([[ np.cos(theta),  np.sin(theta),  0],
      [-np.sin(theta),  np.cos(theta),  0],
      [ 0,        0  ,  1]])
    return R


def rotation_matrix_a_to_b(A, B):
    """
    Finds rotation matrix from vector A in 3d to vector B
    in 3d.
    B = R @ A
    """
    EPS = 1e-8
    cos = np.dot(A, B)
    sin = np.linalg.norm(np.cross(B, A))
    u = A
    v = B - np.dot(A, B) * A
    v = v / (np.linalg.norm(v) + EPS)
    w = np.cross(B, A)
    w = w / (np.linalg.norm(w) + EPS)
    F = np.stack([u, v, w], 1)
    G = np.array([[cos, -sin, 0],
              [sin, cos, 0],
              [0, 0, 1]])
    # B = R @ A
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def save_point_cloud(filename, data):
    np.savetxt(filename, data, delimiter=" ")


def visualize_point_cloud(points, normals=[], colors=[], file="", viz=False):
    # pcd = PointCloud()
    pcd = open3d.open3d.geometry.PointCloud()
    pcd.points = Vector3dVector(points)

    # estimate_normals(pcd, search_param = KDTreeSearchParamHybrid(
    #         radius = 0.1, max_nn = 30))
    if isinstance(normals, np.ndarray):
        pcd.normals = Vector3dVector(normals)
    if isinstance(colors, np.ndarray):
        pcd.colors = Vector3dVector(colors)

    if file:
        write_point_cloud(file, pcd, write_ascii=True)

    if viz:
        draw_geometries([pcd])
    return pcd


def visualize_point_cloud_from_labels(points, labels, COLORS=None, normals=None, viz=False):
    if not isinstance(COLORS, np.ndarray):
        COLORS = np.random.rand(500, 3)

    colors = COLORS[labels]
    pcd = visualize_point_cloud(points, colors=colors, normals=normals, viz=viz)
    return pcd

def sample_mesh_torch(
    v1, v2, v3, n, face_normals=[], rgb1=[], rgb2=[], rgb3=[], norms=False, rgb=False
):
    """
    Samples mesh given its vertices
    :param rgb:
    :param v1: first vertex of the face, N x 3
    :param v2: second vertex of the face, N x 3
    :param v3: third vertex of the face, N x 3
    :param n: number of points to be sampled
    :return:
    """
    areas = 0.5 * torch.norm(torch.cross(v2 - v1, v3 - v1), dim=1)
    # To avoid zero areas
    areas = areas + torch.min(areas) + 1e-8
    probabilities = areas / torch.sum(areas)
    face_ids = np.random.choice(np.arange(len(areas)), size=n, p=probabilities.data.cpu.numpy())
    # import ipdb; ipdb.set_trace()
    
    v1 = v1[face_ids]
    v2 = v2[face_ids]
    v3 = v3[face_ids]

    # (n, 1) the 1 is for broadcasting
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    sample_points = (v1 * u) + (v2 * v) + ((1 - (u + v)) * v3)
    sample_points = sample_points.data.cpu().numpy()

    sample_point_normals = face_normals[face_ids].data.cpu().numpy()
    
    return sample_points, sample_point_normals

    
def sample_mesh(
    v1, v2, v3, n, face_normals=[], rgb1=[], rgb2=[], rgb3=[], norms=False, rgb=False
):
    """
    Samples mesh given its vertices
    :param rgb:
    :param v1: first vertex of the face, N x 3
    :param v2: second vertex of the face, N x 3
    :param v3: third vertex of the face, N x 3
    :param n: number of points to be sampled
    :return:
    """
    areas = triangle_area_multi(v1, v2, v3)
    # To avoid zero areas
    areas = areas + np.min(areas) + 1e-10
    probabilities = areas / np.sum(areas)

    face_ids = np.random.choice(np.arange(len(areas)), size=n, p=probabilities)
    
    v1 = v1[face_ids]
    v2 = v2[face_ids]
    v3 = v3[face_ids]

    # (n, 1) the 1 is for broadcasting
    u = np.random.rand(n, 1)
    v = np.random.rand(n, 1)
    is_a_problem = u + v > 1

    u[is_a_problem] = 1 - u[is_a_problem]
    v[is_a_problem] = 1 - v[is_a_problem]
    sample_points = (v1 * u) + (v2 * v) + ((1 - (u + v)) * v3)
    sample_points = sample_points.astype(np.float32)

    sample_rgb = []
    sample_normals = []

    if rgb:
        v1_rgb = rgb1[face_ids, :]
        v2_rgb = rgb2[face_ids, :]
        v3_rgb = rgb3[face_ids, :]

        sample_rgb = (v1_rgb * u) + (v2_rgb * v) + ((1 - (u + v)) * v3_rgb)

    if norms:
        sample_point_normals = face_normals[face_ids]
        sample_point_normals = sample_point_normals.astype(np.float32)
        return sample_points, sample_point_normals, sample_rgb, face_ids
    else:
        return sample_points, sample_rgb, face_ids


def triangle_area_multi(v1, v2, v3):
    """ v1, v2, v3 are (N,3) arrays. Each one represents the vertices
    such as v1[i], v2[i], v3[i] represent the ith triangle
    """
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1), axis=1)


def visualize_uv_maps(
    output, root_path="data/uvmaps/", iter=0, grid_size=20, viz=False
):
    """
    visualizes uv map using the output of the network
    :param output:
    :param root_path:
    :param iter:
    :return:
    """
    os.makedirs(root_path, exist_ok=True)
    B = output.shape[0]
    for index in range(B):
        figure, a = plt.subplots(1, 3)
        uvmap = output[index, :].reshape((grid_size, grid_size, 2))
        a[0].imshow(np.sum(uvmap, 2))
        uvmap = output[index, :].reshape((grid_size, grid_size, 2))
        
        for ind in range(grid_size):
            a[1].plot(uvmap[ind, :, 1])
        for ind in range(grid_size):
            a[1].plot(uvmap[:, ind, 0])
        temp = output[index, :].reshape((grid_size ** 2, 2))
        a[2].scatter(temp[:, 0], temp[:, 1])
        if viz:
            plt.show()
        plt.savefig("{}/plots_iter_{}.png".format(root_path, iter * B + index))
        plt.close("all")
        np.save("{}/plots_iter_{}.npy".format(root_path, iter * B + index), uvmap)


def visualize_fitted_surface(output, points, grid_size, viz=True, path="data/uvmaps/"):
    os.makedirs(path, exist_ok=True)
    nx, ny = (grid_size, grid_size)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    par = np.concatenate([xv, yv], 1)

    B = output.shape[0]
    predicted_points = []
    surfaces = []
    for index in range(B):
        uv = output[index]
        C = np.sum(np.square(np.expand_dims(uv, 1) - np.expand_dims(par, 0)), 2)
        cost, x, y = lap.lapjv(C)
        p = points[index]
        p = p[y]
        fitted_surface, fitted_points = fit_surface(p, grid_size, grid_size, 2, 2)
        fitted_points = fitted_points - np.expand_dims(np.mean(fitted_points, 0), 0)
        colors_gt = np.ones((np.array(points[index]).shape[0], 3))
        colors_pred = np.ones((np.array(fitted_points).shape[0], 3))
        colors_gt[:, 2] = 0
        colors_pred[:, 1] = 0
        color = np.concatenate([colors_gt, colors_pred])
        p = np.concatenate([np.array(points[index]), np.array(fitted_points)])
        pcd = visualize_point_cloud(p, colors=color, viz=viz)
        open3d.io.write_point_cloud("{}pcd_{}.pcd".format(path, index), pcd)
        predicted_points.append(fitted_points)
        surfaces.append(fitted_surface)
    predicted_points = np.stack(predicted_points, 0)
    return predicted_points, surfaces


def fit_surface_sample_points(output, points, grid_size, regular_grids=False):
    nx, ny = (grid_size, grid_size)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    par = np.concatenate([xv, yv], 1)
    B = output.shape[0]
    predicted_points = []
    fitted_surfaces = []
    for index in range(B):
        uv = output[index]
        # TODO include the optimal rotation matrix
        C = np.sum(np.square(np.expand_dims(uv, 1) - np.expand_dims(par, 0)), 2)
        cost, x, y = lap.lapjv(C)
        p = points[index]
        p = p[y]
        fitted_surface, fitted_points = fit_surface(p, grid_size, grid_size, 2, 2, regular_grids)
        fitted_points = fitted_points - np.expand_dims(np.mean(fitted_points, 0), 0)
        predicted_points.append(fitted_points)
        fitted_surfaces.append(fitted_surface)
    predicted_points = np.stack(predicted_points, 0)
    return predicted_points, fitted_surfaces


def chamfer_distance(pred, gt, sqrt=False):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if sqrt:
        diff = guard_sqrt(diff)
    
    cd = torch.mean(torch.min(diff, 1)[0], 1) + torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) / 2.0
    return cd


def chamfer_distance_one_side(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3)
    if side == 0:
        cd = torch.mean(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd)
    return cd


def chamfer_distance_single_shape(pred, gt, one_side=False, sqrt=False, reduce=True):
    """
    Computes average chamfer distance prediction and groundtruth
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).cuda()

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).cuda()
    pred = torch.unsqueeze(pred, 0)
    gt = torch.unsqueeze(gt, 1)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 2)

    if sqrt:
        distance = guard_sqrt(diff)

    if one_side:
        cd = torch.min(diff, 1)[0]
        if reduce:
            cd = torch.mean(cd, 0)
    else:
        cd1 = torch.min(diff, 0)[0]
        cd2 = torch.min(diff, 1)[0]
        if reduce:
            cd1 = torch.mean(cd1)
            cd2 = torch.mean(cd2)
        cd = (cd1 + cd2) / 2.0
    return cd


def rescale_input_outputs(scales, output, points, control_points, batch_size):
    """
    In the case of anisotropic scaling, we need to rescale the tensors
    to original dimensions to compute the loss and eval metric.
    """
    scales = np.stack(scales, 0).astype(np.float32)
    scales = torch.from_numpy(scales).cuda()
    scales = scales.reshape((batch_size, 1, 3))
    output = (
        output
        * scales
        / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
            (batch_size, 1, 1)
        )
    )
    points = (
        points
        * scales.reshape((batch_size, 3, 1))
        / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
            (batch_size, 1, 1)
        )
    )
    control_points = (
        control_points
        * scales.reshape((batch_size, 1, 1, 3))
        / torch.max(scales.reshape((batch_size, 3)), 1)[0].reshape(
            (batch_size, 1, 1, 1)
        )
    )
    return scales, output, points, control_points

def grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm
    total_norm = total_norm.item()
    return np.isnan(total_norm) or np.isinf(total_norm)

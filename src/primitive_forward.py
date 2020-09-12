from open3d import *
import numpy as np
import torch
from geomdl import fitting as geomdl_fitting
from lapsolver import solve_dense
from open3d import *
from open3d import *

from src.VisUtils import tessalate_points
from src.approximation import fit_bezier_surface_fit_kronecker, BSpline, uniform_knot_bspline_
from src.curve_utils import DrawSurfs
from src.fitting_utils import LeastSquares
from src.fitting_utils import customsvd
from src.fitting_utils import remove_outliers
from src.fitting_utils import standardize_points_torch, sample_points_from_control_points_
from src.fitting_utils import up_sample_points_in_range
from src.fitting_utils import up_sample_points_torch_in_range
from src.guard import guard_sqrt
from src.utils import draw_geometries
from src.utils import rotation_matrix_a_to_b, get_rotation_matrix

draw_surf = DrawSurfs()
EPS = np.finfo(np.float32).eps
torch.manual_seed(2)
np.random.seed(2)
draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)


def print_norm(x):
    print("printing norm 2", torch.norm(x))


def forward_pass_open_spline(
        input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True
):
    nu = nu.cuda(input_points_.get_device())
    nv = nv.cuda(input_points_.get_device())
    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    batch_size = points_.shape[0]
    if viz:
        reg_points = np.copy(points_[:, 0:400])

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)
    output = control_decoder(points, weights.T)

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )
    output = output.view(1, 400, 3)

    out_recon_points = []
    new_outputs = []
    for b in range(batch_size):
        # re-alinging back to original orientation for better comparison
        s = scales[b]

        temp = reconstructed_points[b].clone() * s.reshape((1, 3))
        new_points = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(new_points, 1, 0)
        temp = temp + means[b]

        out_recon_points.append(temp)

        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]
        new_outputs.append(temp)
        if viz:
            new_points = np.linalg.inv(RS[b]) @ reg_points[b].T
            reg_points[b] = new_points.T
            pred_mesh = tessalate_points(reconstructed_points[b], 30, 30)
            gt_mesh = tessalate_points(reg_points[b], 20, 20)
            draw_geometries([pred_mesh, gt_mesh])

    output = torch.stack(new_outputs, 0)
    reconstructed_points = torch.stack(out_recon_points, 0)
    if if_optimize:
        reconstructed_points = optimize_open_spline_kronecker(reconstructed_points, input_points_, output, deform=True)
    return reconstructed_points, reconstructed_points


def initialize_open_spline_model(modelname, mode):
    from src.model import DGCNNControlPoints

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    if torch.cuda.device_count() > 1:
        control_decoder_.cuda(1)
    else:
        control_decoder_.cuda(0)
    control_decoder_.eval()
    return control_decoder_


def optimize_close_spline(reconstructed_points, input_points_):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    out = reconstructed_points[0]
    out = out.data.cpu().numpy()
    out = out.reshape((31, 30, 3))
    out = out[np.arange(0, 31, 1.5).astype(np.int32)][
          :, np.arange(0, 30, 1.5).astype(np.int32).tolist()
          ]
    out = out.reshape((20 * 21, 3))

    input = input_points_[0]
    N = input.shape[0]
    input = up_sample_points_torch_in_range(input, 2000, 2100)
    # L = np.random.choice(np.arange(N), 30 * 31, replace=False)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(out, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]
    size_u = 21
    size_v = 20
    degree_u = 3
    degree_v = 3

    # Do global surface approximation
    surf = geomdl_fitting.approximate_surface(
        matched.tolist(),
        size_u,
        size_v,
        degree_u,
        degree_v,
        ctrlpts_size_u=10,
        ctrlpts_size_v=10,
    )

    regular_parameters = draw_surf.regular_parameterization(31, 30)
    optimized_points = surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def optimize_close_spline_kronecker(reconstructed_points,
                                    input_points_,
                                    control_points,
                                    new_cp_size=10,
                                    new_degree=3,
                                    deform=True):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    if deform:
        from src.fitting_optimization import Arap
        arap = Arap()
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    parameters = draw_surf.boundary_parameterization(30)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(21, 20, 3, 3, 2)

    spline_surf = bspline.create_geomdl_surface(control_points.reshape((21, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 2000, 2100)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = optimized_points.reshape((30, 30, 3))
    optimized_points = torch.cat([optimized_points, optimized_points[0:1]], 0)
    optimized_points = optimized_points.reshape((930, 3))
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def optimize_open_spline_kronecker(reconstructed_points, input_points_, control_points, new_cp_size=10, new_degree=2,
                                   deform=False):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    from src.fitting_optimization import Arap
    bspline = BSpline()
    N = input_points_.shape[1]
    control_points = control_points[0].data.cpu().numpy()
    if deform:
        arap = Arap(30, 30)
        new_mesh = arap.deform(reconstructed_points[0].data.cpu().numpy(),
                               input_points_[0].data.cpu().numpy(), viz=False)
        reconstructed_points = torch.from_numpy(np.array(new_mesh.vertices)).cuda()
        reconstructed_points = torch.unsqueeze(reconstructed_points, 0)

    new_cp_size = new_cp_size
    new_degree = new_degree

    # Note that boundary parameterization is necessary for the fitting
    # otherwise you 
    parameters = draw_surf.boundary_parameterization(20)
    parameters = np.concatenate([np.random.random((1600 - parameters.shape[0], 2)), parameters], 0)

    _, _, ku, kv = uniform_knot_bspline_(20, 20, 3, 3, 2)
    spline_surf = bspline.create_geomdl_surface(control_points.reshape((20, 20, 3)),
                                                np.array(ku),
                                                np.array(kv),
                                                3, 3)

    # these are randomly sampled points on the surface of the predicted spline
    points = np.array(spline_surf.evaluate_list(parameters))

    input = up_sample_points_torch_in_range(input_points_[0], 1600, 2000)

    L = np.random.choice(np.arange(input.shape[0]), 1600, replace=False)
    input = input[L].data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(points, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]

    _, _, ku, kv = uniform_knot_bspline_(new_cp_size, new_cp_size, new_degree, new_degree, 2)

    NU = []
    NV = []
    for index in range(parameters.shape[0]):
        nu, nv = bspline.basis_functions(parameters[index], new_cp_size, new_cp_size, ku, kv, new_degree, new_degree)
        NU.append(nu)
        NV.append(nv)
    NU = np.concatenate(NU, 1).T
    NV = np.concatenate(NV, 1).T

    new_control_points = fit_bezier_surface_fit_kronecker(matched, NU, NV)
    new_spline_surf = bspline.create_geomdl_surface(new_control_points,
                                                    np.array(ku),
                                                    np.array(kv),
                                                    new_degree, new_degree)

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = new_spline_surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def optimize_open_spline(reconstructed_points, input_points_):
    """
    Assuming that initial point cloud size is greater than or equal to
    400.
    """
    out = reconstructed_points[0]
    out = out.data.cpu().numpy()
    out = out.reshape((30, 30, 3))
    out = out.reshape((900, 3))

    input = input_points_[0]
    N = input.shape[0]
    input = up_sample_points_torch_in_range(input, 1200, 1300)
    input = input.data.cpu().numpy()

    dist = np.linalg.norm(
        np.expand_dims(out, 1) - np.expand_dims(input, 0), axis=2
    )

    rids, cids = solve_dense(dist)
    matched = input[cids]
    size_u = 30
    size_v = 30
    degree_u = 2
    degree_v = 2

    # Do global surface approximation
    try:
        surf = geomdl_fitting.approximate_surface(
            matched.tolist(),
            size_u,
            size_v,
            degree_u,
            degree_v,
            ctrlpts_size_u=10,
            ctrlpts_size_v=10,
        )
    except:
        print("open spline, smaller than 400")
        return reconstructed_points

    regular_parameters = draw_surf.regular_parameterization(30, 30)
    optimized_points = surf.evaluate_list(regular_parameters)
    optimized_points = torch.from_numpy(np.array(optimized_points).astype(np.float32)).cuda()
    optimized_points = torch.unsqueeze(optimized_points, 0)
    return optimized_points


def forward_closed_splines(input_points_, control_decoder, nu, nv, viz=False, weights=None, if_optimize=True):
    batch_size = input_points_.shape[0]
    nu = nu.cuda(input_points_.get_device())
    nv = nv.cuda(input_points_.get_device())

    with torch.no_grad():
        points_, scales, means, RS = standardize_points_torch(input_points_, weights)

    if viz:
        reg_points = points_[:, 0:400]

    # points = Variable(torch.from_numpy(points_.astype(np.float32))).cuda()
    points = points_.permute(0, 2, 1)
    output = control_decoder(points, weights.T)

    # Chamfer Distance loss, between predicted and GT surfaces
    reconstructed_points = sample_points_from_control_points_(
        nu, nv, output, batch_size
    )

    closed_reconst = []
    closed_control_points = []

    for b in range(batch_size):
        s = scales[b]
        temp = output[b] * s.reshape((1, 3))
        temp = torch.inverse(RS[b]) @ torch.transpose(temp, 1, 0)
        temp = torch.transpose(temp, 1, 0)
        temp = temp + means[b]

        temp = temp.reshape((20, 20, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_control_points.append(temp)

        temp = (
                reconstructed_points[b].clone() * scales[b].reshape(1, 3)
        )
        temp = torch.inverse(RS[b]) @ temp.T
        temp = torch.transpose(temp, 1, 0) + means[b]
        temp = temp.reshape((30, 30, 3))
        temp = torch.cat([temp, temp[0:1]], 0)
        closed_reconst.append(temp)

    output = torch.stack(closed_control_points, 0)
    reconstructed_points = torch.stack(closed_reconst, 0)
    reconstructed_points = reconstructed_points.reshape((1, 930, 3))

    if if_optimize and (input_points_.shape[1] > 200):
        reconstructed_points = optimize_close_spline_kronecker(reconstructed_points, input_points_, output)
        reconstructed_points = reconstructed_points.reshape((1, 930, 3))
    return reconstructed_points, None, reconstructed_points


def initialize_closed_spline_model(modelname, mode):
    from src.model import DGCNNControlPoints

    control_decoder_ = DGCNNControlPoints(20, num_points=10, mode=mode)
    control_decoder = torch.nn.DataParallel(control_decoder_)
    control_decoder.load_state_dict(
        torch.load(modelname)
    )

    if torch.cuda.device_count() > 1:
        control_decoder_.cuda(1)
    else:
        control_decoder_.cuda(0)

    control_decoder_.eval()
    return control_decoder_


class Fit:
    def __init__(self):
        """
        Defines fitting and sampling modules for geometric primitives.
        """
        LS = LeastSquares()
        self.lstsq = LS.lstsq
        self.parameters = {}

    def sample_torus(self, r_major, r_minor, center, axis):
        d_theta = 60
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1) * r_minor

        circle = np.concatenate([np.zeros((circle.shape[0], 1)), circle], 1)
        circle[:, 1] += r_major

        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])

        torus = []
        for i in range(d_theta):
            R = get_rotation_matrix(theta[i])
            torus.append((R @ circle.T).T)

        torus = np.concatenate(torus, 0)
        R = rotation_matrix_a_to_b(np.array([0, 0, 1.0]), axis)
        torus = (R @ torus.T).T
        torus = torus + center
        return torus

    def sample_plane(self, d, n, mean):
        regular_parameters = draw_surf.regular_parameterization(120, 120)
        n = n.reshape(3)
        r1 = np.random.random()
        r2 = np.random.random()
        a = (d - r1 * n[1] - r2 * n[2]) / (n[0] + EPS)
        x = np.array([a, r1, r2]) - d * n

        x = x / np.linalg.norm(x)
        n = n.reshape((1, 3))

        # let us find the perpendicular vector to a lying on the plane
        y = np.cross(x, n)
        y = y / np.linalg.norm(y)

        param = 1 - 2 * np.array(regular_parameters)
        param = param * 0.75

        gridded_points = param[:, 0:1] * x + param[:, 1:2] * y
        gridded_points = gridded_points + mean
        return gridded_points

    def sample_cone_trim(self, c, a, theta, points):
        """
        Trims the cone's height based points. Basically we project 
        the points on the axis and retain only the points that are in
        the range.
        """
        bkp_points = points
        c = c.reshape((3))
        a = a.reshape((3))
        norm_a = np.linalg.norm(a)
        a = a / norm_a
        proj = (points - c.reshape(1, 3)) @ a
        proj_max = np.max(proj)
        proj_min = np.min(proj)

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        rel_unit_vector = p - c
        rel_unit_vector = (p - c) / np.linalg.norm(p - c)
        rel_unit_vector_min = rel_unit_vector * (proj_min) / (np.cos(theta) + EPS)
        rel_unit_vector_max = rel_unit_vector * (proj_max) / (np.cos(theta) + EPS)

        for j in range(100):
            # p_ = (p - c) * (0.01) * j
            p_ = rel_unit_vector_min + (rel_unit_vector_max - rel_unit_vector_min) * 0.01 * j

            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(rotate_point - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a)

            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        # projecting points to the axis to trim the cone along the height.
        proj = (points - c.reshape((1, 3))) @ a
        proj = proj[:, 0]
        indices = np.logical_and(proj < proj_max, proj > proj_min)
        # project points on the axis, remove points that are beyond the limits.
        return points[indices], normals[indices]

    def sample_cone(self, c, a, theta):
        norm_a = np.linalg.norm(a)
        a = a / norm_a

        # find one point on the cone
        k = np.dot(c, a)
        x = (k - a[1] - a[2]) / (a[0] + EPS)
        y = 1
        z = 1
        d = np.array([x, y, z])
        p = a * (np.linalg.norm(d)) / (np.sin(theta) + EPS) * np.cos(theta) + d

        # This is a point on the surface
        p = p.reshape((3, 1))

        # Now rotate the vector p around axis a by variable degree
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        points = []
        normals = []
        c = c.reshape((3, 1))
        a = a.reshape((3, 1))
        for j in range(100):
            p_ = (p - c) * (0.01) * j
            d_points = []
            d_normals = []
            for d in range(50):
                degrees = 2 * np.pi * 0.01 * d * 2
                R = np.eye(3) + np.sin(degrees) * K + (1 - np.cos(degrees)) * K @ K
                rotate_point = R @ p_
                d_points.append(rotate_point + c)
                d_normals.append(rotate_point - np.linalg.norm(rotate_point) / np.cos(theta) * a / norm_a)
            # repeat the points to close the circle
            d_points.append(d_points[0])
            d_normals.append(d_normals[0])

            points += d_points
            normals += d_normals

        points = np.stack(points, 0)[:, :, 0]
        normals = np.stack(normals, 0)[:, :, 0]
        normals = normals / (np.expand_dims(np.linalg.norm(normals, axis=1), 1) + EPS)

        points = points - c.reshape((1, 3))
        points = 2 * points / (np.max(np.linalg.norm(points, ord=2, axis=1, keepdims=True)) + EPS)
        points = points + c.reshape((1, 3))
        return points, normals

    def sample_sphere(self, radius, center, N=1000):

        theta = 1 - 2 * np.random.random(N) * 3.14
        phi = 1 - 2 * np.random.random(N) * 3.14
        points = np.stack([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                           np.sin(theta)], 1)
        normals = points
        points = points * radius

        points = points + center
        return points, normals

    def sample_sphere(self, radius, center, N=1000):
        center = center.reshape((1, 3))
        d_theta = 100
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta
        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        lam = np.linspace(-1 + 1e-7, 1 - 1e-7, 100)
        radii = radius * np.sqrt(1 - lam ** 2)
        circle = np.concatenate([circle] * lam.shape[0], 0)
        spread_radii = np.repeat(radii, d_theta, 0)
        new_circle = circle * spread_radii.reshape((-1, 1))
        height = np.repeat(lam, d_theta, 0)
        points = np.concatenate([new_circle, height.reshape((-1, 1))], 1)
        points = points - np.mean(points, 0)
        normals = points / np.linalg.norm(points, axis=1, keepdims=True)
        points = points + center
        return points, normals

    def sample_cylinder_trim(self, radius, center, axis, points, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))

        d_theta = 60
        d_height = 100

        R = self.rotation_matrix_a_to_b(np.array([0, 0, 1]), axis[:, 0])

        # project points on to the axis
        points = points - center

        projection = points @ axis
        arg_min_proj = np.argmin(projection)
        arg_max_proj = np.argmax(projection)

        min_proj = np.squeeze(projection[arg_min_proj])
        max_proj = np.squeeze(projection[arg_max_proj])

        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        normals = np.concatenate([circle, np.zeros((circle.shape[0], 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        height = np.expand_dims(np.linspace(min_proj, max_proj, 2 * d_height), 1)
        height = np.repeat(height, d_theta, axis=0)
        try:
            points = np.concatenate([circle, height], 1)
        except:
            import ipdb;
            ipdb.set_trace()
        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    def sample_cylinder(self, radius, center, axis, N=1000):
        """
        :param center: center of size 1 x 3
        :param radius: radius of the cylinder
        :param axis: axis of the cylinder, size 3 x 1
        """
        d_theta = 30
        d_height = 200
        theta = np.arange(d_theta - 1) * 3.14 * 2 / d_theta

        theta = np.concatenate([theta, np.zeros(1)])
        circle = np.stack([np.cos(theta), np.sin(theta)], 1)
        circle = np.concatenate([circle] * 2 * d_height, 0) * radius

        heights = [np.ones(d_theta) * i for i in range(-d_height, d_height)]
        heights = np.concatenate(heights, 0).reshape((-1, 1)) * 0.01

        points = np.concatenate([circle, heights], 1)
        N = points.shape[0]

        normals = np.concatenate([circle, np.zeros((N, 1))], 1)
        normals = normals / np.linalg.norm(normals, axis=1).reshape((N, 1))
        R = self.rotation_matrix_a_to_b(np.array([0, 0, 1]), axis)

        points = R @ points.T
        points = points.T + center
        normals = (R @ normals.T).T

        return points, normals

    def fit_plane_numpy(self, points, normals, weights):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        X = points - np.sum(weights * points, 0).reshape((1, 3)) / np.sum(weights, 0)
        _, s, V = np.linalg.svd(weights * X, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))
        d = np.sum(weights * (a @ points.T).T) / np.sum(weights, 0)
        return a, d

    def fit_plane_torch(self, points, normals, weights, ids=0, show_warning=False):
        """
        Fits plane
        :param points: points with size N x 3
        :param weights: weights with size N x 1
        """
        weights_sum = torch.sum(weights) + EPS

        X = points - torch.sum(weights * points, 0).reshape((1, 3)) / weights_sum

        weighted_X = weights * X
        np_weighted_X = weighted_X.data.cpu().numpy()
        if np.linalg.cond(np_weighted_X) > 1e5:
            if show_warning:
                print("condition number is large in plane!", np.sum(np_weighted_X))
                print(torch.sum(points), torch.sum(weights))

        U, s, V = customsvd(weighted_X)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))
        d = torch.sum(weights * (a @ points.permute(1, 0)).permute(1, 0)) / weights_sum
        return a, d

    def fit_sphere_numpy(self, points, normals, weights):
        dimension = points.shape[1]
        N = weights.shape[0]
        sum_weights = np.sum(weights)
        A = 2 * (- points + np.sum(points * weights, 0) / sum_weights)
        dot_points = np.sum(points * points, 1)
        normalization = np.sum(dot_points * weights) / sum_weights
        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y
        center = -np.linalg.lstsq(A, Y)[0].reshape((1, dimension))
        radius = np.sqrt(np.sum(weights[:, 0] * np.sum((points - center) ** 2, 1)) / sum_weights)
        return center, radius

    def fit_sphere_torch(self, points, normals, weights, ids=0, show_warning=False):

        N = weights.shape[0]
        sum_weights = torch.sum(weights) + EPS
        A = 2 * (- points + torch.sum(points * weights, 0) / sum_weights)

        dot_points = weights * torch.sum(points * points, 1, keepdim=True)

        normalization = torch.sum(dot_points) / sum_weights

        Y = dot_points - normalization
        Y = Y.reshape((N, 1))
        A = weights * A
        Y = weights * Y

        if np.linalg.cond(A.data.cpu().numpy()) > 1e8:
            if show_warning:
                print("condition number is large in sphere!")

        center = -self.lstsq(A, Y, 0.01).reshape((1, 3))
        radius_square = torch.sum(weights[:, 0] * torch.sum((points - center) ** 2, 1)) / sum_weights
        radius_square = torch.clamp(radius_square, min=1e-3)
        radius = guard_sqrt(radius_square)
        return center, radius

    def fit_cylinder_numpy(self, points, normals, weights):
        _, s, V = np.linalg.svd(weights * normals, compute_uv=True)
        a = V.T[:, np.argmin(s)]
        a = np.reshape(a, (1, 3))

        # find the projection onto a plane perpendicular to the axis
        a = a.reshape((3, 1))
        a = a / (np.linalg.norm(a, ord=2) + EPS)

        prj_circle = points - ((points @ a).T * a).T
        center, radius = self.fit_sphere_numpy(prj_circle, normals, weights)
        return a, center, radius

    def fit_cylinder_torch(self, points, normals, weights, ids=0, show_warning=False):
        # compute
        # U, s, V = torch.svd(weights * normals)
        weighted_normals = weights * normals

        if np.linalg.cond(weighted_normals.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large in cylinder")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())

        U, s, V = customsvd(weighted_normals)
        a = V[:, -1]
        a = torch.reshape(a, (1, 3))

        # find the projection onto a plane perpendicular to the axis
        a = a.reshape((3, 1))
        a = a / (torch.norm(a, 2) + EPS)

        prj_circle = points - ((points @ a).permute(1, 0) * a).permute(1, 0)

        # torch doesn't have least square for
        center, radius = self.fit_sphere_torch(prj_circle, normals, weights)
        return a, center, radius

    def fit_cone_torch(self, points, normals, weights, ids=0, show_warning=False):
        """ Need to incorporate the cholesky decomposition based
        least square fitting because it is stable and faster."""

        N = points.shape[0]
        A = weights * normals
        Y = torch.sum(normals * points, 1).reshape((N, 1))
        Y = weights * Y

        # if condition number is too large, return a very zero cone.
        if np.linalg.cond(A.data.cpu().numpy()) > 1e5:
            if show_warning:
                print("condition number is large, cone")
                print(torch.sum(normals).item(), torch.sum(points).item(), torch.sum(weights).item())
            return torch.zeros((1, 3)).cuda(points.get_device()), torch.Tensor([[1.0, 0.0, 0.0]]).cuda(
                points.get_device()), torch.zeros(1).cuda(points.get_device())

        c = self.lstsq(A, Y, lamb=1e-3)

        a, _ = self.fit_plane_torch(normals, None, weights)
        if torch.sum(normals @ a.transpose(1, 0)) > 0:
            # we want normals to be pointing outside and axis to
            # be pointing inside the cone.
            a = - 1 * a

        diff = points - c.transpose(1, 0)
        diff = torch.nn.functional.normalize(diff, p=2, dim=1)
        diff = diff @ a.transpose(1, 0)

        # This is done to avoid the numerical issue when diff = 1 or -1
        # the derivative of acos becomes inf
        diff = torch.abs(diff)
        diff = torch.clamp(diff, max=0.999)
        theta = torch.sum(weights * torch.acos(diff)) / (torch.sum(weights) + EPS)
        theta = torch.clamp(theta, min=1e-3, max=3.142 / 2 - 1e-3)
        return c, a, theta

    def rotation_matrix_a_to_b(self, A, B):
        """
        Finds rotation matrix from vector A in 3d to vector B
        in 3d.
        B = R @ A
        """
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

    def reg_lstsq(self, A, y, lamb=0):
        n_col = A.shape[1]
        return np.linalg.lstsq(A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y))

    def reg_lstsq_torch(self, A, y, lamb=0):
        n_col = A.shape[1]
        A_dash = A.permute(1, 0) @ A + lamb * torch.eye(n_col)
        y_dash = A.permute(1, 0) @ y

        center = self.lstsq(y_dash, A_dash)
        return center


def fit_one_shape(data, fitter):
    """
    Fits primitives/splines to one shape
    """
    input_shape = []
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}

    for part_index, d in enumerate(data):
        points, normals, labels, _ = d
        weights = np.ones((points.shape[0], 1), dtype=np.float32)
        if labels[0] in [0, 9, 6, 7]:
            # closed bspline surface
            # Ignore the patches that are very small, that is smaller than 1%.
            if points.shape[0] < 100:
                continue
            recon_points = fitter.forward_pass_closed_spline(points, weights=weights, ids=part_index)

        elif labels[0] == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weights, ids=part_index)

        elif labels[0] == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weights, ids=part_index)

        elif labels[0] == 4:
            # cylinder
            recon_points = fitter.forward_pass_cylinder(points, normals, weights, ids=part_index)

        elif labels[0] == 5:
            # sphere
            recon_points = fitter.forward_pass_sphere(points, normals, weights, ids=part_index)

        elif labels[0] in [2, 8]:
            # open splines
            recon_points = fitter.forward_pass_open_spline(points, ids=part_index)
        gt_points[part_index] = points
        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape


def fit_one_shape_torch(data, fitter, weights, bw, eval=False, sample_points=False, if_optimize=False,
                        if_visualize=False):
    """
    Fits primitives/splines to 
    """
    input_shape = []
    reconstructed_shape = []
    fitter.fitting.parameters = {}
    gt_points = {}
    spline_count = 0

    for _, d in enumerate(data):
        points, normals, labels, gpoints, segment_indices, part_index = d
        # NOTE: part index and label index are different when one of the predicted
        # labels are missing.
        part_index, label_index = part_index
        N = points.shape[0]
        if not eval:
            weight = weights[:, part_index:part_index + 1] + EPS
            drop_indices = torch.arange(0, N, 2)
            points = points[drop_indices]
            normals = normals[drop_indices]
            weight = weight[drop_indices]

        else:
            weight = weights[segment_indices, part_index:part_index + 1] + EPS

        if not eval:
            # in the training mode, only process upto 5 splines
            # because of the memory constraints.
            if labels in [0, 2, 6, 7, 9, 8]:
                spline_count += 1
                if spline_count > 4:
                    reconstructed_shape.append(None)
                    gt_points[label_index] = None
                    fitter.fitting.parameters[label_index] = None
                    continue
            else:
                # down sample points for geometric primitives, what is the point
                N = points.shape[0]
                drop_indices = torch.arange(0, N, 2)
                points = points[drop_indices]
                normals = normals[drop_indices]
                weight = weight[drop_indices]

        if points.shape[0] < 20:
            reconstructed_shape.append(None)
            gt_points[label_index] = None
            fitter.fitting.parameters[label_index] = None
            continue

        if labels in [0, 9, 6, 7]:
            # closed bspline surface

            if points.shape[0] < 100:
                # drop smaller patches
                reconstructed_shape.append(None)
                gt_points[label_index] = None
                fitter.fitting.parameters[label_index] = None
                continue

            if eval:
                # since this is a eval mode, weights are all one.
                Z = points.shape[0]
                points = torch.from_numpy(remove_outliers(points.data.cpu().numpy()).astype(np.float32)).cuda(
                    points.get_device())
                weight = weight[0:points.shape[0]]

                # Note: we can apply poisson disk sampling to remove points.
                # Rarely results in removal of points.
                points, weight = up_sample_points_in_range(points, weight, 1400, 1800)

            recon_points = fitter.forward_pass_closed_spline(points, weights=weight, ids=label_index,
                                                             if_optimize=if_optimize and (Z > 200))

        elif labels == 1:
            # Fit plane
            recon_points = fitter.forward_pass_plane(points, normals, weight, ids=label_index,
                                                     sample_points=sample_points)

        elif labels == 3:
            # Cone
            recon_points = fitter.forward_pass_cone(points, normals, weight, ids=label_index,
                                                    sample_points=sample_points)

        elif labels == 4:
            # cylinder
            recon_points = fitter.forward_pass_cylinder(points, normals, weight, ids=label_index,
                                                        sample_points=sample_points)

        elif labels == 5:
            # "sphere"
            recon_points = fitter.forward_pass_sphere(points, normals, weight, ids=label_index,
                                                      sample_points=sample_points)

        elif labels in [2, 8]:
            # open splines
            if points.shape[0] < 100:
                reconstructed_shape.append(None)
                gt_points[label_index] = None
                fitter.fitting.parameters[label_index] = None
                continue
            if eval:
                # in the eval mode, make the number of points per segment to lie in a range that is suitable for the spline network.
                # remove outliers. Only occur rarely, but worth removing them.
                points = torch.from_numpy(remove_outliers(points.data.cpu().numpy()).astype(np.float32)).cuda(
                    points.get_device())
                weight = weight[0:points.shape[0]]
                points, weight = up_sample_points_in_range(points, weight, 1000, 1500)

            recon_points = fitter.forward_pass_open_spline(points, weights=weight, ids=label_index,
                                                           if_optimize=if_optimize)

        if if_visualize:
            try:
                gt_points[label_index] = torch.from_numpy(gpoints).cuda(points.get_device())
            except:
                gt_points[label_index] = None
        else:
            gt_points[label_index] = gpoints

        reconstructed_shape.append(recon_points)
    return gt_points, reconstructed_shape

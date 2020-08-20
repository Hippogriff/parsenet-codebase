import torch
import numpy as np
from torch.autograd.variable import Variable
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.nn.functional import normalize
from src.utils import chamfer_distance, chamfer_distance_one_side


mse = MSELoss(size_average=True, reduce=True)


def regressions_loss_per_shape(output, points):
    """
    Both are in square grid
    """
    dist = torch.sum((output - points) ** 2, 2)
    dist = torch.mean(dist)
    return dist


def all_permutations(array):
    """
    This method is used to generate permutation of control points grid.
    This is specifically used for open b-spline surfaces.
    """
    permutations = []
    permutations.append(array)
    permutations.append(torch.flip(array, (1,)))
    permutations.append(torch.flip(array, (2,)))
    permutations.append(torch.flip(array, (1, 2)))

    permutations.append(torch.transpose(array, 2, 1))
    permutations.append(torch.transpose(torch.flip(array, (1,)), 2, 1))
    permutations.append(torch.transpose(torch.flip(array, (2,)), 2, 1))
    permutations.append(torch.transpose(torch.flip(array, (1, 2)), 2, 1))
    permutations = torch.stack(permutations, 0)
    permutations = permutations.permute(1, 0, 2, 3, 4)
    return permutations


def all_permutations_half(array):
    """
    This method is used to generate permutation of control points grid.
    This is specifically used for closed b-spline surfaces. Note that
    In the pre-processing step, all closed splines are made to close in u
    direction only, thereby reducing the possible permutations to half. This
    is done to speedup the training and also to facilitate learning for neural
    network.
    """
    permutations = []
    permutations.append(array)
    permutations.append(torch.flip(array, (1,)))
    permutations.append(torch.flip(array, (2,)))
    permutations.append(torch.flip(array, (1, 2)))
    permutations = torch.stack(permutations, 0)
    permutations = permutations.permute(1, 0, 2, 3, 4)
    return permutations


def roll(x: torch.Tensor, shift: int, dim: int = -1, fill_pad=None):
    """
    Rolls the tensor by certain shifts along certain dimension.
    """
    if 0 == shift:
        return x
    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift))
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim))), gap], dim=dim)
    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)).cuda())
        return torch.cat([gap, x.index_select(dim, torch.arange(shift).cuda())], dim=dim)


def control_points_permute_reg_loss(output, control_points, grid_size):
    """
    control points prediction with permutation invariant loss
    :param output: output of the network
    :param control_points: N x grid_size x grid_size x 3
    :param grid_size_x: size of the control points in u direction
    :param grid_size_y: size of the control points in v direction
    """
    batch_size = output.shape[0]
    # TODO Check whether this permutation is good or not.
    output = output.view(batch_size, grid_size, grid_size, 3)
    output = torch.unsqueeze(output, 1)
    
    # N x 8 x grid_size x grid_size x 3
    control_points = all_permutations(control_points)
    diff = (output - control_points) ** 2
    diff = torch.sum(diff, (2, 3, 4))
    loss, index = torch.min(diff, 1)
    loss = torch.mean(loss) / (grid_size * grid_size * 3)
    # returns the loss and also the permutation that matches
    # best with the input.
    return loss, control_points[np.arange(batch_size), index]


def control_points_permute_closed_reg_loss(output, control_points, grid_size_x, grid_size_y):
    """
    control points prediction with permutation invariant loss
    :param output: output of the network
    :param control_points: N x grid_size x grid_size x 3
    :param grid_size_x: size of the control points in u direction
    :param grid_size_y: size of the control points in v direction
    """
    batch_size = output.shape[0]
    output = output.view(batch_size, grid_size_x, grid_size_y, 3)
    output = torch.unsqueeze(output, 1)
    
    # N x 8 x grid_size x grid_size x 3
    rhos = []
    for i in range(grid_size_y):
        new_control_points = roll(control_points, i, 1)
        rhos.append(all_permutations_half(new_control_points))
    control_points = torch.cat(rhos, 1)

    diff = (output - control_points) ** 2
    diff = torch.sum(diff, (2, 3, 4))
    loss, index = torch.min(diff, 1)
    loss = torch.mean(loss) / (grid_size_x * grid_size_y * 3)

    return loss, control_points[np.arange(batch_size), index]


def control_points_loss(output, control_points, grid_size):
    """
    control points prediction with permutation invariant loss
    :param output: N x C x 3
    :param control_points: N x grid_size x grid_size x 3
    """
    batch_size = output.shape[0]
    # N x 8 x grid_size x grid_size x 3
    output = output.view(batch_size, grid_size, grid_size, 3)
    diff = (output - control_points) ** 2
    diff = torch.sum(diff, (1, 2, 3))
    loss = torch.mean(diff) / (grid_size * grid_size * 3)
    return loss


def spline_reconstruction_loss_one_sided(nu, nv, output, points, config, side=1):
    """
    Spline reconsutruction loss defined using chamfer distance, but one
    sided either gt surface can cover the prediction or otherwise, which
    is defined by the network. side=1 means prediction can cover gt.
    :param nu: spline basis function in u direction.
    :param nv: spline basis function in v direction.
    :param points: points sampled over the spline.
    :param config: object of configuration class for extra parameters. 
    """
    reconst_points = []
    batch_size = output.shape[0]
    c_size_u = output.shape[1]
    c_size_v = output.shape[2]
    grid_size_u = nu.shape[0]
    grid_size_v = nv.shape[0]
    
    output = output.view(config.batch_size, config.grid_size, config.grid_size, 3)
    points = points.permute(0, 2, 1)
    for b in range(config.batch_size):
        point = []
        for i in range(3):
            point.append(torch.matmul(torch.matmul(nu, output[b, :, :, i]), torch.transpose(nv, 1, 0)))
        reconst_points.append(torch.stack(point, 2))

    reconst_points = torch.stack(reconst_points, 0)
    reconst_points = reconst_points.view(config.batch_size, grid_size_u * grid_size_v, 3)
    dist = chamfer_distance_one_side(reconst_points, points, side)
    return dist, reconst_points


def spline_reconstruction_loss(nu, nv, output, points, config, sqrt=False):
    reconst_points = []
    batch_size = output.shape[0]
    grid_size = nu.shape[0]
    output = output.reshape(config.batch_size, nu.shape[1], nv.shape[1], 3)
    points = points.permute(0, 2, 1)
    for b in range(config.batch_size):
        point = []
        for i in range(3):
            point.append(torch.matmul(torch.matmul(nu, output[b, :, :, i]), torch.transpose(nv, 1, 0)))
        reconst_points.append(torch.stack(point, 2))
    reconst_points = torch.stack(reconst_points, 0)
    reconst_points = reconst_points.view(config.batch_size, grid_size ** 2, 3)
    dist = chamfer_distance(reconst_points, points, sqrt=sqrt)
    return dist, reconst_points


def uniform_knot_bspline(control_points_u, control_points_v, degree_u, degree_v, grid_size=30):
    """
    Returns uniform knots, given the number of control points in u and v directions and 
    their degrees.
    """
    u = v = np.arange(0., 1, 1 / grid_size)

    knots_u = [0.0] * degree_u + np.arange(0, 1.01, 1 / (control_points_u - degree_u)).tolist() + [1.0] * degree_u
    knots_v = [0.0] * degree_v + np.arange(0, 1.01, 1 / (control_points_v - degree_v)).tolist() + [1.0] * degree_v

    nu = []
    nu = np.zeros((u.shape[0], control_points_u))
    for i in range(u.shape[0]):
        for j in range(0, control_points_u):
            nu[i, j] = basis_function_one(degree_u, knots_u, j, u[i])

    nv = np.zeros((v.shape[0], control_points_v))
    for i in range(v.shape[0]):
        for j in range(0, control_points_v):
            nv[i, j] = basis_function_one(degree_v, knots_v, j, v[i])
    return nu, nv


def laplacian_loss(output, gt, dist_type="l2"):
    """
    Computes the laplacian of the input and output grid and defines
    regression loss.
    :param output: predicted control points grid. Makes sure the orientation/
    permutation of this output grid matches with the ground truth orientation.
    This is done by finding the least cost orientation during training.
    :param gt: gt control points grid.
    """
    batch_size, grid_size, grid_size, input_channels = gt.shape
    filter = ([[[0.0, 0.25, 0.0], [0.25, -1.0, 0.25], [0.0, 0.25, 0.0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
               [[0, 0, 0], [0, 0, 0], [0, 0, 0]]])
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])

    filter = -np.array(filter, dtype=np.float32)
    filter = Variable(torch.from_numpy(filter)).cuda()

    laplacian_output = F.conv2d(output.permute(0, 3, 1, 2), filter, padding=1)
    laplacian_input = F.conv2d(gt.permute(0, 3, 1, 2), filter, padding=1)
    if dist_type == "l2":
        dist = (laplacian_output - laplacian_input) ** 2
    elif dist_type == "l1":
        dist = torch.abs(laplacian_output - laplacian_input)
    dist = torch.sum(dist, 1)
    dist = torch.mean(dist)
    return dist


def basis_function_one(degree, knot_vector, span, knot):
    """ Computes the value of a basis function for a single parameter.

    Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.
    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis function, :math:`N_{i,p}`
    :rtype: float
    """
    # Special case at boundaries
    if (
        (span == 0 and knot == knot_vector[0])
        or (span == len(knot_vector) - degree - 2)
        and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    # Knot is outside of span range
    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = [0.0 for _ in range(degree + span + 1)]

    # Initialize the zeroth degree basis functions
    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    # Computing triangular table of basis functions
    for k in range(1, degree + 1):
        # Detecting zeros saves computations
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (
                knot_vector[span + k] - knot_vector[span]
            )

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            # Zero detection
            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp
    return N[0]

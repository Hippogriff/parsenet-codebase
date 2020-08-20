import numpy as np
from src.utils import visualize_point_cloud
import torch
from torch.autograd.gradcheck import gradcheck
from src.curve_utils import DrawSurfs
from open3d import *
from src.curve_utils import DrawSurfs
import open3d
import numpy as np
import torch
from src.VisUtils import tessalate_points
from src.utils import draw_geometries
from torch.autograd.variable import Variable
from torch.autograd import Function
import time
from src.loss import (
    basis_function_one,
    uniform_knot_bspline,
    spline_reconstruction_loss,
)
from src.utils import visualize_point_cloud
from geomdl import fitting as geomdl_fitting
from lapsolver import solve_dense
from src.curve_utils import DrawSurfs
from open3d import *
import copy
from src.eval_utils import to_one_hot
from src.segment_utils import mean_IOU_one_sample, iou_segmentation, to_one_hot, matching_iou, relaxed_iou, relaxed_iou_fast
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from src.guard import guard_exp
import torch.nn.functional as F

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
draw_surf = DrawSurfs()
EPS = float(np.finfo(np.float32).eps)
torch.manual_seed(2)
np.random.seed(2)
draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)


class LeastSquares:
    def __init__(self):
        pass
    
    def lstsq(self, A, Y, lamb=0.0):
        """
        Differentiable least square
        :param A: m x n
        :param Y: n x 1
        """
        cols = A.shape[1]
        if np.isinf(A.data.cpu().numpy()).any():
            import ipdb; ipdb.set_trace()

        # Assuming A to be full column rank
        if cols == torch.matrix_rank(A):
            # Full column rank
            q, r = torch.qr(A)
            x = torch.inverse(r) @ q.transpose(1, 0) @ Y
        else:
            # rank(A) < n, do regularized least square.
            AtA = A.transpose(1, 0) @ A

            # get the smallest lambda that suits our purpose, so that error in
            # results minimized.
            with torch.no_grad():
                lamb = best_lambda(AtA)
            A_dash = AtA + lamb * torch.eye(cols, device=A.get_device())
            Y_dash = A.transpose(1, 0) @ Y

            # if it still doesn't work, just set the lamb to be very high value.
            x = self.lstsq(A_dash, Y_dash, 1)
        return x


def best_lambda(A):
    """
    Takes an under determined system and small lambda value,
    and comes up with lambda that makes the matrix A + lambda I
    invertible. Assuming A to be square matrix.
    """
    lamb = 1e-6
    cols = A.shape[0]
    
    for i in range(7):
        A_dash = A + lamb * torch.eye(cols, device=A.get_device())
        if cols == torch.matrix_rank(A_dash):
            # we achieved the required rank
            break
        else:
            # factor by which to increase the lambda. Choosing 10 for performance.
            lamb *= 10
    return lamb


def up_sample_all(points, normals, weights, cluster_ids, primitives, labels):
    """
    Upsamples points based on nearest neighbors.
    """
    dist = np.expand_dims(points, 1) - np.expand_dims(points, 0)
    dist = np.sum(dist ** 2, 2)
    indices = np.argsort(dist, 1)
    neighbors = points[indices[:, 0:3]]
    centers = np.mean(neighbors, 1)
    
    new_points = np.concatenate([points, centers])
    new_normals = np.concatenate([normals, normals])
    new_weights = np.concatenate([weights, weights], 1)
    
    new_primitives = np.concatenate([primitives, primitives])
    new_cluster_ids = np.concatenate([cluster_ids, cluster_ids])
    new_labels = np.concatenate([labels, labels])
    
    return new_points, new_normals, new_weights, new_primitives, new_cluster_ids, new_labels


def up_sample_points(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    """

    points = points.data.cpu()
    batch_size = points.shape[0]
    points = points.permute(0, 2, 1)

    for t in range(times):
        Points = []
        for b in range(batch_size):
            dist = torch.unsqueeze(points[b], 1) - torch.unsqueeze(points[b], 0)
            dist = torch.sum(dist ** 2, 2)
            _, indices = torch.topk(dist, k=3, dim=1, largest=False)
            neighbors = points[b][indices]
            centers = torch.mean(neighbors, 1)

            new_points = torch.cat([points[b], centers])
            Points.append(new_points)
        points = torch.stack(Points, 0)
    return points.permute(0, 2, 1).cuda()


def up_sample_points_numpy(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for t in range(times):
        dist = np.expand_dims(points, 1) - np.expand_dims(points, 0)
        dist = np.sum(dist ** 2, 2)
        indices = np.argsort(dist, 1)
        neighbors = points[indices[:, 0:3]]
        centers = np.mean(neighbors, 1)
        points = np.concatenate([points, centers])
    return points

def up_sample_points_torch(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for t in range(times):
        dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
        dist = torch.sum(dist ** 2, 2)
        _, indices = torch.topk(dist, 5, 1, largest=False)
        neighbors = points[indices[:, 1:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points


def up_sample_points_torch_memory_efficient(points, times=1):
    """
    Upsamples points based on nearest neighbors.
    Takes two neareast neighbors and finds the centroid
    and that becomes the new point.
    :param points: N x 3
    """
    for t in range(times):
        # dist = torch.unsqueeze(points, 1) - torch.unsqueeze(points, 0)
        # dist = torch.sum(dist ** 2, 2)
        indices = []
        N = min(points.shape[0], 100)
        for i in range(points.shape[0] // N):
            diff_ = torch.sum((torch.unsqueeze(points[i * N :(i+1) * N], 1) - torch.unsqueeze(points, 0)) ** 2, 2)
            _, diff_indices = torch.topk(diff_, 5, 1, largest=False)
            indices.append(diff_indices)
        indices = torch.cat(indices, 0)
        # dist = dist_memory_efficient(points, points)
        # _, indices = torch.topk(dist, 5, 1, largest=False)
        neighbors = points[indices[:, 0:]]
        centers = torch.mean(neighbors, 1)
        points = torch.cat([points, centers])
    return points

def dist_memory_efficient(p, q):
    diff = []
    for i in range(p.shape[0]):
        diff.append(torch.sum((torch.unsqueeze(p[i:i+1], 1) - torch.unsqueeze(q, 0)) ** 2, 2).data.cpu().numpy())
    diff = np.concantenate(diff, 0)
    # diff = torch.sqrt(diff)

    return diff

def up_sample_points_in_range(points, weights, a_min, a_max):
    
    N = points.shape[0]
    if N > a_max:
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        weights = weights[L]
        return points, weights
    else:
        while True:
            points = up_sample_points_torch(points)
            weights = torch.cat([weights, weights], 0)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    weights = weights[L]
    return points, weights
    

def up_sample_points_torch_in_range(points, a_min, a_max):
    N = points.shape[0]
    if N > a_max:
        N = points.shape[0]
        L = np.random.choice(np.arange(N), a_max, replace=False)
        points = points[L]
        return points
    else:
        while True:
            points = up_sample_points_torch(points)
            if points.shape[0] >= a_max:
                break
    N = points.shape[0]
    L = np.random.choice(np.arange(N), a_max, replace=False)
    points = points[L]
    return points
    


def create_grid(input, grid_points, size_u, size_v, thres=0.02):
    grid_points = torch.from_numpy(grid_points.astype(np.float32)).cuda()
    input = torch.from_numpy(input.astype(np.float32)).cuda()
    grid_points = grid_points.reshape((size_u, size_v, 3))
    
    grid_points.permute(2, 0, 1)
    grid_points = torch.unsqueeze(grid_points, 0)

    filter = np.array([[[0.25, 0.25], [0.25, 0.25]],
           [[0, 0], [0, 0]],
           [[0.0, 0.0], [0.0, 0.0]]]).astype(np.float32)
    filter = np.stack([filter, np.roll(filter, 1, 0), np.roll(filter, 2, 0)])
    filter = torch.from_numpy(filter).cuda()
    grid_mean_points = F.conv2d(grid_points.permute(0, 3, 1, 2), filter, padding=0)
    grid_mean_points = grid_mean_points.permute(0, 2, 3, 1)
    grid_mean_points = grid_mean_points.reshape(((size_u - 1) * (size_v - 1), 3))

    if True:
        # diff = (torch.unsqueeze(grid_mean_points, 1) - torch.unsqueeze(input, 0)) ** 2
        diff = []
        for i in range(grid_mean_points.shape[0]):
            diff.append(torch.sum((torch.unsqueeze(grid_mean_points[i:i+1], 1) - torch.unsqueeze(input, 0)) ** 2, 2))
        diff = torch.cat(diff, 0)
        diff = torch.sqrt(diff)
        indices = torch.min(diff, 1)[0] < thres
    else:
        grid_mean_points = grid_mean_points.data.cpu().numpy()
        input = input.data.cpu().numpy()
        diff = (np.expand_dims(grid_mean_points, 1) - np.expand_dims(input, 0)) ** 2 
        diff = np.sqrt(np.sum(diff, 2))
        indices = np.min(diff, 1) < thres
    
    mask_grid = indices.reshape(((size_u - 1), (size_v - 1)))
    return mask_grid, diff, filter, grid_mean_points


def tessalate_points_fast(points, size_u, size_v, mask=None, viz=False):
    """
    Given a grid points, this returns a tessalation of the grid using triangle.
    Furthermore, if the mask is given those grids are avoided.
    """
    def index_to_id(i, j, size_v):
        return i * size_v + j
    triangles = []
    vertices = points
    for i in range(0, size_u - 1):
        for j in range(0, size_v - 1):
            if mask is not None:
                if mask[i, j] == 0:
                    continue
            tri = [index_to_id(i, j, size_v), index_to_id(i+1, j, size_v), index_to_id(i+1, j+1, size_v)]
            triangles.append(tri)
            tri = [index_to_id(i, j, size_v), index_to_id(i+1, j+1, size_v), index_to_id(i, j+1, size_v)]
            triangles.append(tri)
    new_mesh = geometry.TriangleMesh()
    new_mesh.triangles = utility.Vector3iVector(np.array(triangles))
    new_mesh.vertices = utility.Vector3dVector(np.stack(vertices, 0))
    new_mesh.remove_unreferenced_vertices()
    new_mesh.compute_vertex_normals()
    if viz:
        draw_geometries([new_mesh])
    return new_mesh


def weights_normalize(weights, bw):
    """
    Assuming that weights contains dot product of embedding of a
    points with embedding of cluster center, we want to normalize
    these weights to get probabilities. Since the clustering is
    gotten by mean shift clustering, we use the same kernel to compute
    the probabilities also.
    """
    prob = guard_exp(weights / (bw ** 2) / 2)
    prob = prob / torch.sum(prob, 0, keepdim=True)

    # This is to avoid numerical issues
    if weights.shape[0] == 1:
        return prob
    
    # This is done to ensure that max probability is 1 at the center.
    # this will be helpful for the spline fitting network
    prob = prob - torch.min(prob, 1, keepdim=True)[0]
    prob = prob / (torch.max(prob, 1, keepdim=True)[0] + EPS)
    return prob
    
def one_hot_normalization(weights):
    N, K = weights.shape
    weights = np.argmax(weights, 1)
    one_hot = to_one_hot(weights, K)
    weights = one_hot.float()
    return weights

def SIOU(target, pred_labels):
    """
    First it computes the matching using hungarian matching
    between predicted and groun truth labels.
    Then it computes the iou score, starting from matching pairs
    coming out from hungarian matching solver. Note that
    it is assumed that the iou is only computed over matched pairs.
    
    That is to say, if any column in the matched pair has zero
    number of points, that pair is not considered.
    """
    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)
    cost = relaxed_iou(torch.unsqueeze(cluster_ids_one_hot, 0).double(), torch.unsqueeze(labels_one_hot, 0).double())
    cost_ = 1.0 - torch.as_tensor(cost)
    cost_ = cost_.data.cpu().numpy()
    matching = []

    for b in range(1):
        rids, cids = solve_dense(cost_[b])
        matching.append([rids, cids])

    s_iou = matching_iou(matching, np.expand_dims(pred_labels, 0), np.expand_dims(target, 0))
    return s_iou


def match(target, pred_labels):
    labels_one_hot = to_one_hot(target)
    cluster_ids_one_hot = to_one_hot(pred_labels)

    # cost = relaxed_iou(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    # cost_ = 1.0 - torch.as_tensor(cost)
    cost = relaxed_iou_fast(torch.unsqueeze(cluster_ids_one_hot, 0).float(), torch.unsqueeze(labels_one_hot, 0).float())
    
    # cost_ = 1.0 - torch.as_tensor(cost)
    cost_ = 1.0 - cost.data.cpu().numpy()
    rids, cids = solve_dense(cost_[0])
    
    unique_target = np.unique(target)
    unique_pred = np.unique(pred_labels)
    return rids, cids, unique_target, unique_pred


def visualize_weighted_points(points, w, normals=None, viz=False):
    N = points.shape[0]
    colors = cm.get_cmap("seismic")(w)[:, 0:3]
    return visualize_point_cloud(points, colors=colors, normals=normals, viz=viz)


def compute_grad_V(U, S, V, grad_V):
    N = S.shape[0]
    K = svd_grad_K(S)
    S = torch.eye(N).cuda(S.get_device()) * S.reshape((N, 1))
    inner = K.T * (V.T @ grad_V)
    inner = (inner + inner.T) / 2.0
    return 2 * U @ S @ inner @ V.T


def svd_grad_K(S):
    N = S.shape[0]
    s1 = S.view((1, N))
    s2 = S.view((N, 1))
    diff = s2 - s1
    plus = s2 + s1

    # TODO Look into it
    eps = torch.ones((N, N)) * 10**(-6)
    eps = eps.cuda(S.get_device())
    max_diff = torch.max(torch.abs(diff), eps)
    sign_diff = torch.sign(diff)

    K_neg = sign_diff * max_diff

    # gaurd the matrix inversion
    K_neg[torch.arange(N), torch.arange(N)] = 10 ** (-6)
    K_neg = 1 / K_neg
    K_pos = 1 / plus

    ones = torch.ones((N, N)).cuda(S.get_device())
    rm_diag = ones - torch.eye(N).cuda(S.get_device())
    K = K_neg * K_pos * rm_diag
    return K


class CustomSVD(Function):
    """
    Costum SVD to deal with the situations when the
    singular values are equal. In this case, if dealt
    normally the gradient w.r.t to the input goes to inf.
    To deal with this situation, we replace the entries of
    a K matrix from eq: 13 in https://arxiv.org/pdf/1509.07838.pdf
    to high value.
    Note: only applicable for the tall and square matrix and doesn't
    give correct gradients for fat matrix. Maybe transpose of the
    original matrix is requires to deal with this situation. Left for
    future work.
    """
    @staticmethod
    def forward(ctx, input):
        # Note: input is matrix of size m x n with m >= n.
        # Note: if above assumption is voilated, the gradients
        # will be wrong.
        try:
            U, S, V = torch.svd(input, some=True)
        except:
            import ipdb; ipdb.set_trace()

        ctx.save_for_backward(U, S, V)
        return U, S, V

    @staticmethod
    def backward(ctx, grad_U, grad_S, grad_V):
        U, S, V = ctx.saved_tensors
        grad_input = compute_grad_V(U, S, V, grad_V)
        return grad_input

customsvd = CustomSVD.apply

def standardize_points(points):
    Points = []
    stds = []
    Rs = []
    means = []
    batch_size = points.shape[0]

    for i in range(batch_size):
        point, std, mean, R = standardize_point(points[i])
        Points.append(point)
        stds.append(std)
        means.append(mean)
        Rs.append(R)
    
    Points = np.stack(Points, 0)
    return Points, stds, means, Rs


def standardize_point(point):
    mean = torch.mean(point, 0)[0]
    point = point - mean
    
    S, U = pca_numpy(point)
    smallest_ev = U[:, np.argmin(S)]
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
    # axis aligns with x axis.
    point = R @ point.T
    point = point.T

    std = np.abs(np.max(point, 0) - np.min(point, 0))
    std = std.reshape((1, 3))
    point = point / (std + EPS)
    return point, std, mean, R


def standardize_points_torch(points, weights):
    Points = []
    stds = []
    Rs = []
    means = []
    batch_size = points.shape[0]

    for i in range(batch_size):
        point, std, mean, R = standardize_point_torch(points[i], weights)

        Points.append(point)
        stds.append(std)
        means.append(mean)
        Rs.append(R)
    
    Points = torch.stack(Points, 0)
    return Points, stds, means, Rs


def standardize_point_torch(point, weights):
    # TODO: not back propagation through rotation matrix and scaling yet.
    # Change this 0.8 to 0 to include all points.
    higher_indices = weights[:, 0] > 0.8

    # some heuristic
    if torch.sum(higher_indices) < 400:
        if weights.shape[0] >= 7500:
            _, higher_indices = torch.topk(weights[:, 0], weights.shape[0] // 4)
        else:
            _, higher_indices = torch.topk(weights[:, 0], weights.shape[0] // 2)

    weighted_points = point[higher_indices] * weights[higher_indices]
    
    # Note: gradients throught means, force the network to produce correct means.
    mean = torch.sum(weighted_points, 0) / (torch.sum(weights[higher_indices]) + EPS)

    point = point - mean

    # take only very confident points to compute PCA direction.
    S, U = pca_torch(point[higher_indices])
    smallest_ev = U[:, torch.min(S[:, 0], 0)[1]].data.cpu().numpy()
    
    R = rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))

    # axis aligns with x axis.
    R = R.astype(np.float32)
    
    R = torch.from_numpy(R).cuda(point.get_device()).detach()

    point = R @ torch.transpose(point, 1, 0)
    point = torch.transpose(point, 1, 0)

    weighted_points = point[higher_indices] * weights[higher_indices]
    try:
        std = torch.abs(torch.max(weighted_points, 0)[0] - torch.min(weighted_points, 0)[0])
    except:
        import ipdb; ipdb.set_trace()
    std = std.reshape((1, 3)).detach()
    point = point / (std + EPS)
    return point, std, mean, R


def rotation_matrix_a_to_b(A, B):
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
    try:
        R = F @ G @ np.linalg.inv(F)
    except:
        R = np.eye(3, dtype=np.float32)
    return R


def pca_numpy(X):
    S, U = np.linalg.eig(X.T @ X)
    return S, U


def pca_torch(X):
    # TODO 2Change this to do SVD, because it is stable and computationally
    # less intensive.
    covariance = torch.transpose(X, 1, 0) @ X
    S, U = torch.eig(covariance, eigenvectors=True)
    return S, U


def reverse_all_transformations(points, means, stds, Rs):
    new_points = []
    for i in range(len(Rs)):
        new_points.append(reverse_all_transformation(points[i], means[i], stds[i], Rs[i]))
    new_points = np.stack(new_points, 0)
    return new_points


def reverse_all_transformation(point, mean, std, R):
    std = std.reshape((1, 3))
    new_points_scaled = point * std
    new_points_inv_rotation = np.linalg.inv(R) @ new_points_scaled.T
    new_points_final = new_points_inv_rotation.T + mean
    return new_points_final


def sample_points_from_control_points_(nu, nv, outputs, batch_size, input_size_u=20, input_size_v=20):
    batch_size = outputs.shape[0]
    grid_size = nu.shape[0]
    reconst_points = []
    outputs = outputs.reshape((batch_size, input_size_u, input_size_v, 3))
    for b in range(batch_size):
        point = []
        for i in range(3):
            # cloning because it is giving error in back ward pass.
            point.append(torch.matmul(torch.matmul(nu, outputs[b, :, :, i].clone()), torch.transpose(nv, 1, 0)))
        reconst_points.append(torch.stack(point, 2))
    reconst_points = torch.stack(reconst_points, 0)
    reconst_points = reconst_points.view(batch_size, grid_size ** 2, 3)
    return reconst_points


def project_to_plane(points, a, d):
    a = a.reshape((3, 1))
    a = a / torch.norm(a, 2)
    # Project on the same plane but passing through origin
    projections = points - ((points @ a).permute(1, 0) * a).permute(1, 0)
    
    # shift the points on the plane back to the original d distance
    # from origin
    projections = projections + a.transpose(1, 0) * d
    return projections


def project_to_point_cloud(points, surface):
    """
    project points on to the surface defined by points
    """
    diff = (np.expand_dims(points, 1) - np.expand_dims(surface, 0)) ** 2
    diff = np.sum(diff, 2)
    return surface[np.argmin(diff, 1)]


def bit_mapping_points(input, output_points, thres, size_u, size_v, mesh=None):
    if mesh:
        pass
    else:
        mesh = tessalate_points(output_points, size_u, size_v)
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles) 
    output = np.mean(vertices[triangles], 1)
    diff = (np.expand_dims(output, 1) - np.expand_dims(input, 0)) ** 2 
    diff = np.sqrt(np.sum(diff, 2)) 
    indices = np.min(diff, 1) < thres
    mesh = copy.deepcopy(mesh)
    t = np.array(mesh.triangles)
    mesh.triangles = Vector3iVector(t[indices])
    return mesh


def bit_mapping_points_torch(input, output_points, thres, size_u, size_v, mesh=None):
    mask, diff, filter, grid_mean_points = create_grid(input, output_points, size_u, size_v, thres=thres)
    mesh = tessalate_points_fast(output_points, size_u, size_v, mask=mask)
    t3 = time.time()
    return mesh


def bit_mapping_points(input, output_points, thres, size_u, size_v, mesh=None):
    if mesh:
        pass
    else:
        mesh = tessalate_points(output_points, size_u, size_v)
    vertices = np.array(mesh.vertices)
    triangles = np.array(mesh.triangles) 
    output = np.mean(vertices[triangles], 1)
    diff = (np.expand_dims(output, 1) - np.expand_dims(input, 0)) ** 2 
    diff = np.sqrt(np.sum(diff, 2)) 
    indices = np.min(diff, 1) < thres
    mesh = copy.deepcopy(mesh)
    t = np.array(mesh.triangles)
    mesh.triangles = Vector3iVector(t[indices])
    return mesh


def bit_map_mesh(mesh, include_indices):
    mesh = copy.deepcopy(mesh)
    t = np.array(mesh.triangles)
    mesh.triangles = Vector3iVector(t[include_indices])
    return mesh


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_down_sample(ind)
    outlier_cloud = cloud.select_down_sample(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    open3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def remove_outliers(points, viz=False):
    pcd = visualize_point_cloud(points)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                    std_ratio=0.50)
    if viz:
        display_inlier_outlier(voxel_down_pcd, ind)
    return np.array(cl.points)

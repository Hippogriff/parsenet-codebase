"""
This script describes all fitting modules like bspline fitting, geometric 
primitives. The idea is to call each module with required input parameters
and get as an output the parameters of fitting.
"""
import open3d
import numpy as np
import torch
from src.VisUtils import tessalate_points
from src.utils import draw_geometries
from torch.autograd.variable import Variable
from src.primitive_forward import Fit
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
from src.fitting_utils import (
    sample_points_from_control_points_,
    standardize_points_torch,
    rotation_matrix_a_to_b,
    pca_numpy,
    reverse_all_transformations,
    project_to_point_cloud,
    project_to_plane,
    bit_map_mesh,
    bit_mapping_points,
)
from src.primitive_forward import forward_pass_open_spline, forward_closed_splines, initialize_open_spline_model, initialize_closed_spline_model
import scipy


Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector
draw_surf = DrawSurfs()
regular_parameters = draw_surf.regular_parameterization(30, 30)
EPS = np.finfo(np.float32).eps


class Arap:
    def __init__(self, size_u=31, size_v=30):
        """
        As rigid as possible transformation of mesh,
        """
        self.size_u = size_u
        self.size_v = size_v
        l = np.array(self.get_boundary_indices(size_u, size_v))

        indices = []
        for i in range(l.shape[0]):
            indices.append(np.unravel_index(np.ravel_multi_index(l[i],
                                                                 [size_u, size_v]),
                                            [size_u * size_v])[0])
        self.indices = indices
        
    def deform(self, recon_points, gt_points, viz=False):
        """
        ARAP, given recon_points, that are in grid, we first create a mesh out of
        it, then we do max matching to find correspondance between gt and boundary
        points. Then we do ARAP over the mesh, making the boundary points go closer
        to the matched points. Note that this is taking only the points 
        TODO: better way to do it is do maximal matching between all points and use
        only the boundary points as the pivot points.
        """
        new_recon_points = recon_points.reshape((self.size_u, self.size_v, 3))
        mesh = tessalate_points(recon_points, self.size_u, self.size_v)
    
        new_recon_points = recon_points.reshape((self.size_u, self.size_v, 3))

        mesh_ = mesh
        for i in range(1):
            mesh, constraint_ids, constraint_pos = self.generate_handles(mesh_,
                                                                         self.indices,
                                                                         gt_points,
                                                                         np.array(mesh_.vertices))
            constraint_ids = np.array(constraint_ids, dtype=np.int32)
            constraint_pos = open3d.utility.Vector3dVector(constraint_pos)
            
            mesh_prime = mesh.deform_as_rigid_as_possible(
                open3d.utility.IntVector(constraint_ids), constraint_pos, max_iter=500)
            mesh_ = mesh_prime
            
        if viz:
            pcd = visualize_point_cloud(gt_points)
            mesh_prime.compute_vertex_normals()
            mesh.paint_uniform_color((1, 0, 0))
            handles = open3d.geometry.PointCloud()
            handles.points = constraint_pos
            handles.paint_uniform_color((0, 1, 0))
            open3d.visualization.draw_geometries([mesh, mesh_prime, handles, pcd]) 
        return mesh_prime
    
    def get_boundary_indices(self, m, n):
        l = []
        for i in range(m): 
            for j in range(n): 
                if (j == 0): 
                    l.append((i, j) )
                elif (j == n-1): 
                    l.append((i, j))
        return l

    def generate_handles(self, mesh, indices, input_points, recon_points):
        matched_points = self.define_matching(input_points, recon_points)
        dist = matched_points - recon_points
        vertices = np.asarray(mesh.vertices)

        handle_ids = indices
        handle_positions = []
        for i in indices:
            handle_positions.append(vertices[i] + dist[i])
        return mesh, handle_ids, handle_positions

    def define_matching(self, input, out):
        # Input points need to at least 1.2 times more than output points
        L = np.random.choice(np.arange(input.shape[0]), int(1.2 * out.shape[0]), replace=False)
        input = input[L]
        
        dist = scipy.spatial.distance.cdist(out, input)
        rids, cids = solve_dense(dist)
        matched = input[cids]
        return matched


class FittingModule:
    def __init__(self, closed_splinenet_path, open_splinenet_path):
        self.fitting = Fit()
        self.closed_splinenet_path = closed_splinenet_path
        self.open_splinenet_path = open_splinenet_path

        # get routine for the spline prediction
        nu, nv = uniform_knot_bspline(20, 20, 3, 3, 30)
        self.nu = torch.from_numpy(nu.astype(np.float32))
        self.nv = torch.from_numpy(nv.astype(np.float32))
        self.open_control_decoder = initialize_open_spline_model(
            self.open_splinenet_path, 0
        )
        self.closed_control_decoder = initialize_closed_spline_model(
            self.closed_splinenet_path, 1
        )

    def forward_pass_open_spline(self, points, ids, weights, if_optimize=False):
        points = torch.unsqueeze(points, 0)

        # NOTE: this will avoid back ward pass through the encoder of SplineNet.
        points.requires_grad = False
        reconst_points = forward_pass_open_spline(
            input_points_=points, control_decoder=self.open_control_decoder, nu=self.nu, nv=self.nv, if_optimize=if_optimize, weights=weights)[1]
        # reconst_points = np.array(reconst_points).astype(np.float32)
        torch.cuda.empty_cache()
        self.fitting.parameters[ids] = ["open-spline", reconst_points]
        return reconst_points

    def forward_pass_closed_spline(self, points, ids, weights, if_optimize=False):
        points = torch.unsqueeze(points, 0)
        points.requires_grad = False
        reconst_points = forward_closed_splines(
            points, self.closed_control_decoder, self.nu, self.nv, if_optimize=if_optimize, weights=weights)[2]
        torch.cuda.empty_cache()
        self.fitting.parameters[ids] = ["closed-spline", reconst_points]
        
        return reconst_points

    def forward_pass_plane(self, points, normals, weights, ids, sample_points=False):
        axis, distance = self.fitting.fit_plane_torch(
            points=points,
            normals=normals,
            weights=weights,
            ids=ids,
        )
        self.fitting.parameters[ids] = ["plane", axis.reshape((3, 1)), distance]
        if sample_points:
            # Project points on the surface
            new_points = project_to_plane(
                points, axis, distance.item()
            )

            new_points = self.fitting.sample_plane(
                distance.item(),
                axis.data.cpu().numpy(),
                mean=torch.mean(new_points, 0).data.cpu().numpy(),
            )
            return new_points
        else:
            None

    def forward_pass_cone(self, points, normals, weights, ids, sample_points=False):
        try:
            apex, axis, theta = self.fitting.fit_cone_torch(
                points,
                normals,
                weights=weights,
                ids=ids,
            )
        except:
            import ipdb; ipdb.set_trace()
        
        self.fitting.parameters[ids] = ["cone", apex.reshape((1, 3)), axis.reshape((3, 1)), theta]
        if sample_points:
            new_points, new_normals = self.fitting.sample_cone_trim(
                apex.data.cpu().numpy().reshape(3), axis.data.cpu().numpy().reshape(3), theta.item(), points.data.cpu().numpy()
            )
            # new_points = project_to_point_cloud(points, new_points)
            return new_points
        else:
            None

    def forward_pass_cylinder(self, points, normals, weights, ids, sample_points=False):
        a, center, radius = self.fitting.fit_cylinder_torch(
            points,
            normals,
            weights,
            ids=ids,
        )
        self.fitting.parameters[ids] = ["cylinder", a, center, radius]
        if sample_points:
            new_points, new_normals = self.fitting.sample_cylinder_trim(
                radius.item(),
                center.data.cpu().numpy().reshape(3),
                a.data.cpu().numpy().reshape(3),
                points.data.cpu().numpy(),
                N=10000,
            )

            # new_points = project_to_point_cloud(points, new_points)
            return new_points
        else:
            return None

    def forward_pass_sphere(self, points, normals, weights, ids, sample_points=False):
        center, radius = self.fitting.fit_sphere_torch(
            points,
            normals,
            weights,
            ids=ids,
        )
        self.fitting.parameters[ids] = ["sphere", center, radius]
        if sample_points:
            # Project points on the surface
            new_points, new_normals = self.fitting.sample_sphere(radius.item(), center.data.cpu().numpy(), N=10000)
            center = center.data.cpu().numpy()
            # sphere = geometry.TriangleMesh.create_sphere(radius=radius, resolution=50)
            # sphere.translate(center.reshape(3).astype(np.float64).tolist())
            # new_points = project_to_point_cloud(points.data.cpu().numpy(), new_points)
            return new_points
        else:
            return None

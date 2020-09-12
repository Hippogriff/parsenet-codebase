"""
This defines a module for all sorts of visualization necessary for debugging and other
final visualization.
"""
import copy
import os
from random import shuffle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from geomdl.tessellate import make_triangle_mesh
from open3d import *
from open3d import *
from open3d import utility
from open3d import visualization
from transforms3d.affines import compose
from transforms3d.euler import euler2mat

Vector3dVector, Vector3iVector = utility.Vector3dVector, utility.Vector3iVector


# TODO Visualizing input and output in a grid
# TODO look at the meshutils
# TODO Other grid visualization
# TODO Visualize the spline surfaces
# TODO Find some representative shapes that are difficult, and can be used to benchmark algorithms
# TODO see how the surfaces reconstructs after the UV predictions


def plotall(images: List, cmap="Greys_r"):
    """
    Awesome function to plot figures in list of list fashion.
    Every list inside the list, is assumed to be drawn in one row.
    :param images: List of list containing images
    :param cmap: color map to be used for all images
    :return: List of figures.
    """
    figures = []
    num_rows = len(images)
    for r in range(num_rows):
        cols = len(images[r])
        f, a = plt.subplots(1, cols)
        for c in range(cols):
            a[c].imshow(images[r][c], cmap=cmap)
            a[c].title.set_text("{}".format(c))
            a[c].axis("off")
            a[c].grid("off")
        figures.append(f)
    return figures


class PlotSurface:
    def __init__(self, abstract_class="vtk"):
        self.abstract_class = abstract_class
        if abstract_class == "plotly":
            from geomdl.visualization.VisPlotly import VisSurface
        elif abstract_class == "vtk":
            from geomdl.visualization.VisVTK import VisSurface
        self.VisSurface = VisSurface

    def plot(self, surf, colormap=None):
        surf.vis = self.VisSurface()
        if colormap:
            surf.render(colormap=cm.cool)
        else:
            surf.render()


def load_points_from_directory(path, suffix=".xyz", tessalate=False, random=True, max_models=50):
    pcds = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(suffix):
                pcds.append(root + "/" + f)
    if not random:
        pcds.sort()
    else:
        shuffle(pcds)
    pcds = pcds[0:max_models]
    for index, value in enumerate(pcds):
        pcds[index] = np.loadtxt(value)
    return pcds


def visualize_from_directory(path, suffix=".xyz", tessalate=False, random=True, max_models=50):
    pcds = []
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(suffix):
                pcds.append(root + "/" + f)
    if not random:
        pcds.sort()
    else:
        shuffle(pcds)
    pcds = pcds[0:max_models]
    for index, value in enumerate(pcds):
        pcds[index] = np.loadtxt(value)
    pcds = np.stack(pcds, 0)
    vis_batch_in_grid(pcds, tessalate)


def convert_into_open3d_format(points, tessellate=False):
    if tessellate:
        size_u = int(np.sqrt(points.shape[0]))
        pcd = tessalate_points(points[:, 0:3], size_u, size_u)
    else:
        pcd = PointCloud()
        size = points.shape[1]
        pcd.points = Vector3dVector(points[:, 0:3])
        if size > 3:
            pcd.colors = Vector3dVector(points[:, 3:] / 255.0)
    return pcd


def generate_grid(pcds: List):
    batch_size = len(pcds)

    height = int(np.sqrt(batch_size))
    width = int(batch_size // height)
    grids = []
    for i in range(int(height)):
        grid = []
        for j in range(int(width)):
            grid.append(pcds[i * width + j])
        grids.append(grid)

    grid = []
    for k in range(height * width, batch_size):
        grid.append(pcds[k])
    grids.append(grid)
    return grids


def visualize_compare_gt_pred(path_gt, path_pred, suffix=".xyz", tessalte=False):
    print(path_gt, path_pred)
    pcds_gt = []
    for root, dirs, files in os.walk(path_gt):
        for f in files:
            if f.endswith(suffix):
                pcds_gt.append(root + "/" + f)
    pcds_gt.sort()

    pcds_pred = []
    for root, dirs, files in os.walk(path_pred):
        for f in files:
            if f.endswith(suffix):
                pcds_pred.append(root + "/" + f)
    pcds_pred.sort()
    print(len(pcds_pred))
    for i in range(min(len(pcds_pred), len(pcds_gt))):
        pcds = []
        print(np.loadtxt(pcds_gt[i])[:, 0:3].shape)
        pts_pred = np.loadtxt(pcds_pred[i])[:, 0:3]
        pts_gt = np.loadtxt(pcds_gt[i])[:, 0:3]
        pcds.append()
        pcds.append()
        pcds = np.stack(pcds, 0)
        vis_batch_in_grid(pcds, tessalate)


def tessalate_points(points, size_u, size_v, mask=None):
    points = points.reshape((size_u * size_v, 3))
    points = [list(points[i, :]) for i in range(points.shape[0])]
    vertex, triangle = make_triangle_mesh(points, size_u, size_v, mask=mask)

    triangle = [t.data for t in triangle]
    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = Vector3dVector(np.array(vertex))
    mesh.triangles = Vector3iVector(np.array(triangle))
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


def save_xyz(points, root_path, epoch, prefix, color=None):
    os.makedirs(root_path, exist_ok=True)
    batch_size = points.shape[0]
    for i in range(batch_size):
        if isinstance(color, np.ndarray):
            pcd = np.concatenate([points[i], color], 1)
        else:
            pcd = points[i]
        np.savetxt(root_path + "{}_{}_{}.xyz".format(prefix, epoch, i), pcd)


def save_xyz_continuous(points, root_path, id, prefix, color=None):
    """
    Saves xyz in continuous manner used for saving testing.
    """
    os.makedirs(root_path, exist_ok=True)
    batch_size = points.shape[0]
    for i in range(batch_size):
        if isinstance(color, np.ndarray):
            pcd = np.concatenate([points[i], color], 1)
        else:
            pcd = points[i]
        np.savetxt(root_path + "{}_{}.xyz".format(prefix, id * batch_size + i), pcd)


def vis_batch_in_grid(points, tessalate=False):
    """
    It takes the points cloud in batch fomrat and returns a grid containing
    pcds for the open3d visualization.
    :param points: numpy array of size B x N x 3
    """
    batch_size = points.shape[0]
    height = int(np.sqrt(batch_size))
    width = int(batch_size // height)
    grids = []
    size = points.shape[2]
    for i in range(int(height)):
        grid = []
        for j in range(int(width)):
            if tessalate:
                size_u = int(np.sqrt(points[i * width + j, :, 0:3].shape[0]))
                pcd = tessalate_points(points[i * width + j, :, 0:3], size_u, size_u)
            else:
                pcd = PointCloud()
                pcd.points = Vector3dVector(points[i * width + j, :, 0:3])
                if size > 3:
                    pcd.colors = Vector3dVector(points[i * width + j, :, 3:] / 255.0)
            grid.append(pcd)
        grids.append(grid)

    grid = []
    for k in range(height * width, batch_size):
        if tessalate:
            size_u = int(np.sqrt(points[k, :, 0:3].shape[0]))
            pcd = tessalate_points(points[k, :, 0:3], size_u, size_u)
        else:
            pcd = PointCloud()
            pcd.points = Vector3dVector(points[k, :, 0:3])
            if size > 3:
                pcd.colors = Vector3dVector(points[k, :, 3:] / 255.0)
        grid.append(pcd)
    grids.append(grid)

    if tessalate:
        grid_meshes_lists_visulation(grids, viz=True)
    else:
        grid_points_lists_visulation(grids, viz=True)


def custom_draw_geometry_load_option(pcds, render=False):
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    vis = visualization.Visualizer()
    vis.create_window()
    for pcd in pcds:
        pcd.transform(M)
        vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("render_options.json")
    vis.run()
    if render:
        image = vis.capture_screen_float_buffer()
        vis.destroy_window()
        return image
    vis.destroy_window()


def save_images_from_list_pcds(pcds: List, vis, pcd, path_template=None):
    pcd = copy.deepcopy(pcd)
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    for index, p in enumerate(pcds):
        if index == 0:
            pcd.transform(M)
            vis.add_geometry(pcd)
            vis.run()
        else:
            pcd.points = p.points
            pcd.colors = p.colors
            pcd.normals = p.normals
            pcd.transform(M)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        plt.imsave(path_template.format(index), image)


def save_images_from_list_pcds_meshes(pcds: List, vis, pcd, path_template=None):
    R = euler2mat(-15 * 3.14 / 180, -35 * 3.14 / 180, 35)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    for index, p in enumerate(pcds):
        if index == 0:
            pcd.vertices = p.vertices
            pcd.triangles = p.triangles
            pcd.transform(M)
            pcd.compute_vertex_normals()
            vis.add_geometry(pcd)
            vis.run()
        else:
            pcd.vertices = p.vertices
            pcd.triangles = p.triangles
            pcd.transform(M)
            pcd.compute_vertex_normals()
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        image = np.array(vis.capture_screen_float_buffer())

        plt.imsave(path_template.format(index), image[200:-200, 200:-200])


def save_images_shape_patches_collection(Pcds: List, path_template=None):
    """
    Given a list of list, where the inner list containts open3d meshes
    Now, the task is to consider the inner list contains surface patches
    for each segment of the shape. We need to visualize the shape at different
    rotations.
    """
    os.makedirs(path_template, exist_ok=True)
    R = euler2mat(60 * 3.14 / 180, 45 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(1, 1, 1))

    for index, shape_list in enumerate(Pcds):
        vis = visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().load_from_json("../render_options.json")
        #         param = io.read_pinhole_camera_parameters("viewpoint.json")

        for s in shape_list:
            vis.add_geometry(s)

        for i in range(3):
            if i == 0:
                pass
            else:
                for s in shape_list:
                    s.transform(M)
                    vis.add_geometry(s)
                vis.poll_events()

                vis.update_renderer()
            #             ctr = vis.get_view_control()
            #             ctr.convert_from_pinhole_camera_parameters(param)

            vis.run()
            image = np.array(vis.capture_screen_float_buffer())
            plt.imsave("{}{}_{}.png".format(path_template, index, i), image[50:-50, 300:-300])
        vis.destroy_window()
    return vis


def grid_pcd_visulation_save_images(pcds: List, pcd, vis=None, first=True):
    """
    Assuming the the elements of List are itself point clouds of numpy arrays
    """
    # First normalize them
    R = euler2mat(-75 * 3.14 / 180, -75 * 3.14 / 180, 0)
    M = compose(T=(0, 0, 0), R=R, Z=(5, 5, 5))
    half_length = np.min((len(pcds) // 2, 10))

    for index, p in enumerate(pcds):
        p.points = Vector3dVector(
            p.points - np.mean(np.array(p.points), 0).reshape(1, 3)
        )
        pcds[index] = p

    points = []
    colors = []
    for j in range(2):
        for i in range(half_length):
            p = pcds[j * half_length + i]
            shift_y = j * 1.3
            shift_x = i * 1.3
            temp = np.array(p.points)
            temp = np.matmul(temp, M[0:3, 0:3])
            temp = temp + np.matmul(np.array([shift_x, shift_y, 0]), M[0:3, 0:3])
            points.append(temp)
            colors.append(p.colors)

    points = np.concatenate(points, 0)
    colors = np.concatenate(colors, 0)
    pcd.points = Vector3dVector(points)
    pcd.colors = Vector3dVector(colors)

    if first:
        vis = Visualizer()
        vis.create_window()
        vis.get_render_option().load_from_json("renderoption.json")
        vis.add_geometry(pcd)
        vis.run()
        first = False
    else:
        print("here")
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.run()

    image = vis.capture_screen_float_buffer()
    return image, vis, first


class VizGridAll:
    def __init__(self):
        pass

    def load_file_paths(path, file_type):
        # TODO Use wild card to get files
        retrieved_path = []
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith(file_type):
                    retrieved_path.append(root + "/" + f)
        return retrieved_path

    def load_files(retrieved_path, file_type):
        if file_type == "xyz":
            retrieved_path.sort()
            pcds = []
            for index, value in enumerate(retrieved_path):
                pcds[index] = np.loadtxt(value)
            pcds = np.stack(pcds, 0)
            vis_batch_in_grid(pcds)
        elif file_type == ".ply":
            print("Not Impletementd Yet!")


# def grid_general_lists_visulation(pcds: List[List], viz=False):
#     """
#     Every list contains a list of points clouds to be visualized.
#     Every element of the list of list is a point cloud in pcd format
#     """
#     # First normalize them
#     import open3d as o3d
#     for pcd_list in pcds:
#         for index, p in enumerate(pcd_list):
#             if isinstance(p, PointCloud):
#                 maxx = np.max(np.array(p.points), 0)
#                 minn = np.min(np.array(p.points), 0)
#                 points = np.array(p.points) - np.mean(np.array(p.points), 0).reshape(1, 3)
#                 points = points / np.linalg.norm(maxx - minn)
#                 p.points = Vector3dVector(points)
#             elif isinstance(p, TriangleMesh):
#                 maxx = np.max(np.array(p.vertices), 0)
#                 minn = np.min(np.array(p.vertices), 0)
#                 points = np.array(p.vertices) - np.mean(np.array(p.vertices), 0).reshape(1, 3)
#                 points = points / np.linalg.norm(maxx - minn)
#                 p.vertices = Vector3dVector(points)

#     new_meshes = []
#     for j in range(len(pcds)):
#         for i in range(len(pcds[j])):
#             if isinstance(p, PointCloud):
#                 p = pcds[j][i]
#                 shift_y = j * 2
#                 shift_x = i * 2
#                 p.points = Vector3dVector(
#                     np.array(p.points) + np.array([shift_x, shift_y, 0])
#                 )
#                 new_meshes.append(p)
#             elif isinstance(p, TriangleMesh):
#                 p = pcds[j][i]
#                 shift_y = j * 2
#                 shift_x = i * 2
#                 p.vertices = Vector3dVector(
#                     np.array(p.vertices) + np.array([shift_x, shift_y, 0])
#                 )
#                 new_meshes.append(p)

#     if viz:
#         # draw_geometries(new_meshes)
#         custom_draw_geometry_load_option(new_meshes)
#     return new_meshes


def grid_points_lists_visulation(pcds: List, viz=False):
    """
    Every list contains a list of points clouds to be visualized.
    Every element of the list of list is a point cloud in pcd format
    """
    # First normalize them
    for pcd_list in pcds:
        for index, p in enumerate(pcd_list):
            maxx = np.max(np.array(p.points), 0)
            minn = np.min(np.array(p.points), 0)
            points = np.array(p.points) - np.mean(np.array(p.points), 0).reshape(1, 3)
            points = points / np.linalg.norm(maxx - minn)
            p.points = Vector3dVector(points)

    new_meshes = []
    for j in range(len(pcds)):
        for i in range(len(pcds[j])):
            p = pcds[j][i]
            shift_y = j * 1.1
            shift_x = i * 1.1
            p.points = Vector3dVector(
                np.array(p.points) + np.array([shift_x, shift_y, 0])
            )
            new_meshes.append(p)
    if viz:
        visualization.draw_geometries(new_meshes)
    return new_meshes


def grid_meshes_lists_visulation(pcds, viz=False) -> None:
    """
    Every list contains a list of points clouds to be visualized.
    Every element of the list of list is a point cloud in pcd format
    """
    # First normalize them
    for pcd_list in pcds:
        for index, p in enumerate(pcd_list):
            maxx = np.max(np.array(p.vertices), 0)
            minn = np.min(np.array(p.vertices), 0)
            points = np.array(p.vertices) - np.mean(np.array(p.vertices), 0).reshape(1, 3)
            points = points / np.linalg.norm(maxx - minn)
            p.vertices = Vector3dVector(points)

    new_meshes = []
    for j in range(len(pcds)):
        for i in range(len(pcds[j])):
            p = pcds[j][i]
            shift_y = j * 1.2
            shift_x = i * 1.2
            p.vertices = Vector3dVector(
                np.array(p.vertices) + np.array([shift_x, shift_y, 0])
            )
            new_meshes.append(p)
    if viz:
        visualization.draw_geometries(new_meshes)
    return new_meshes


class MeshData:
    """
    Return the mesh data given the index of the test shape
    """

    def __init__(self):
        path_txt = "/mnt/gypsum/mnt/nfs/work1/kalo/gopalsharma/Projects/surfacefitting/dataset/filtered_data/points/new_test_all_disconnected.txt"
        self.path_meshes = "/mnt/gypsum/mnt/nfs/work1/kalo/gopalsharma/Projects/surfacefitting/dataset/filtered_data/points/mesh_data/meshes/"

        with open(path_txt, "r") as file:
            self.all_paths = file.readlines()
        self.all_paths = [a[0:-1] for a in self.all_paths]

    def return_open3d_mesh(self, vertices, triangles):
        mesh = geometry.TriangleMesh()
        mesh.vertices = utility.Vector3dVector(vertices)
        mesh.triangles = utility.Vector3iVector(triangles)
        return mesh

    def retrieve_mesh(self, index, viz=False):
        mesh = trimesh.load_mesh(self.path_meshes + index + ".obj")
        new_mesh = self.return_open3d_mesh(mesh.vertices, mesh.faces)
        if viz:
            #             visualization.draw_geometries([new_mesh])
            custom_draw_geometry_load_option([new_mesh])
        return new_mesh

    def custom_draw_geometry_load_option(self, pcds):
        vis = visualization.Visualizer()
        vis.create_window()
        for pcd in pcds:
            vis.add_geometry(pcd)
        vis.get_render_option().load_from_json("render_options.json")
        vis.run()
        vis.destroy_window()

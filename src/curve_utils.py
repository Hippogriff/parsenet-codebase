"""
This script contains utility function to draw surfaces
"""

import numpy as np
from geomdl import BSpline, NURBS
from geomdl import fitting
from geomdl import multi
from geomdl.visualization import VisMPL
from matplotlib import cm


class DrawSurfs:
    def __init__(self):
        """
        Given surfaces from features files from ABC dataset,
        load it into geomdl object or samples points on the surfaces
        of primitives, depending on the case. Defines utility to sample
        points form the surface of splines and primitives.
        """
        self.function_dict = {
            "Sphere": self.draw_sphere,
            "BSpline": self.draw_nurbspatch,
            "Cylinder": self.draw_cylinder,
            "Cone": self.draw_cone,
            "Torus": self.draw_torus,
            "Plane": self.draw_plane,
        }

    def load_shape(self, shape):
        """
        Takes a list containing surface in feature file format, and returns
        a list of sampled points on the surface of primitive/splines.
        """
        Points = []
        for surf in shape:
            function = self.function_dict[surf["type"]]
            points = function(surf)
            Points.append(points)
            Points = np.concatenate(Points, 0)
            return Points

    def draw_plane(self, surf):
        l = np.array(surf["location"])
        x = np.array(surf["x_axis"])
        y = np.array(surf["y_axis"])
        parameters = np.array(surf["vert_parameters"])
        u_min, v_min = np.min(parameters, 0)
        u_max, v_max = np.max(parameters, 0)
        u, v = np.meshgrid(np.arange(u_min, u_max, 0.1), np.arange(v_min, v_max, 0.1))
        plane = (
                l
                + np.expand_dims(u.flatten(), 1) * x.reshape((1, 3))
                + np.expand_dims(v.flatten(), 1) * y.reshape((1, 3))
        )
        return plane

    def draw_cylinder(self, surf):
        l = np.array(surf["location"])
        x = np.array(surf["x_axis"]).reshape((1, 3))
        y = np.array(surf["y_axis"]).reshape((1, 3))
        z = np.array(surf["z_axis"]).reshape((1, 3))
        r = np.array(surf["radius"])
        parameters = np.array(surf["vert_parameters"])
        u_min, v_min = np.min(parameters, 0)
        u_max, v_max = np.max(parameters, 0)
        u, v = np.meshgrid(np.arange(0, 3.14 * 2, 0.1), np.arange(v_min, v_max, 0.1))
        u = np.expand_dims(u.flatten(), 1)
        v = np.expand_dims(v.flatten(), 1)
        temp = np.cos(u) * r * x
        cylinder = l + np.cos(u) * r * x + np.sin(u) * r * y + v * z
        return cylinder

    def draw_sphere(self, surf):
        l = np.array(surf["location"])
        x = np.array(surf["x_axis"]).reshape((1, 3))
        y = np.array(surf["y_axis"]).reshape((1, 3))
        r = np.array(surf["radius"])
        z = np.cross(x, y)
        parameters = np.array(surf["vert_parameters"])
        u_min, v_min = np.min(parameters, 0)
        u_max, v_max = np.max(parameters, 0)
        u, v = np.meshgrid(np.arange(u_min, u_max, 0.3), np.arange(v_min, v_max, 0.3))
        u = np.expand_dims(u.flatten(), 1)
        v = np.expand_dims(v.flatten(), 1)

        sphere = l + r * np.cos(v) * (np.cos(u) * x + np.sin(u) * y) + r * np.sin(v) * z
        return sphere

    def draw_cone(self, surf):
        l = np.array(surf["location"])
        x = np.array(surf["x_axis"]).reshape((1, 3))
        y = np.array(surf["y_axis"]).reshape((1, 3))
        z = np.array(surf["z_axis"]).reshape((1, 3))
        r = np.array(surf["radius"])
        a = np.array(surf["angle"])

        parameters = np.array(surf["vert_parameters"])
        u_min, v_min = np.min(parameters, 0)
        u_max, v_max = np.max(parameters, 0)
        u, v = np.meshgrid(np.arange(u_min, u_max, 0.1), np.arange(v_min, v_max, 0.1))
        u = np.expand_dims(u.flatten(), 1)
        v = np.expand_dims(v.flatten(), 1)

        cone = (
                l
                + (r + v * np.sin(a)) * (np.cos(u) * x + np.sin(u) * y)
                + v * np.cos(a) * z
        )
        return cone

    def draw_torus(self, surf):
        l = np.array(surf["location"])
        x = np.array(surf["x_axis"]).reshape((1, 3))
        y = np.array(surf["y_axis"]).reshape((1, 3))
        z = np.array(surf["z_axis"]).reshape((1, 3))
        r_max = np.array(surf["max_radius"])
        r_min = np.array(surf["min_radius"])

        parameters = np.array(data["surfaces"][5]["vert_parameters"])
        u_min, v_min = np.min(parameters, 0)
        u_max, v_max = np.max(parameters, 0)
        u, v = np.meshgrid(np.arange(u_min, u_max, 0.3), np.arange(v_min, v_max, 0.3))
        u = np.expand_dims(u.flatten(), 1)
        v = np.expand_dims(v.flatten(), 1)
        cone = (
                l
                + (r_max + r_min * np.cos(v)) * (np.cos(u) * x + np.sin(u) * y)
                + (r_min) * np.sin(v) * z
        )
        return cone

    def load_spline_curve(self, spline):
        crv = BSpline.Curve()
        crv.degree = spline["degree"]
        crv.ctrlpts = spline["poles"]
        crv.knotvector = spline["knots"]
        return crv

    def load_spline_surf(self, spline):
        # Create a BSpline surface
        if spline["v_rational"] or spline["u_rational"]:
            surf = NURBS.Surface()
            control_points = np.array(spline["poles"])
            size_u, size_v = control_points.shape[0], control_points.shape[1]

            # Set degrees
            surf.degree_u = spline["u_degree"]
            surf.degree_v = spline["v_degree"]

            # Set control points
            surf.ctrlpts2d = np.concatenate([control_points,
                                             np.ones((size_u, size_v, 1))], 2).tolist()
            surf.knotvector_v = spline["v_knots"]
            surf.knotvector_u = spline["u_knots"]

            weights = spline["weights"]
            l = []
            for i in weights:
                l += i
            surf.weights = l
            return surf

        else:
            surf = BSpline.Surface()

            # Set degrees
            surf.degree_u = spline["u_degree"]
            surf.degree_v = spline["v_degree"]

            # Set control points
            surf.ctrlpts2d = spline["poles"]

            # Set knot vectors
            surf.knotvector_u = spline["u_knots"]
            surf.knotvector_v = spline["v_knots"]
            return surf

    def draw_nurbspatch(self, surf):
        surf = self.load_spline_surf(surf)
        return surf.evalpts

    def vis_spline_curve(self, crv):
        crv.vis = VisMPL.VisCurve3D()
        crv.render()

    def vis_spline_surf(self, surf):
        surf.vis = VisMPL.VisSurface()
        surf.render()

    def vis_multiple_spline_surf(self, surfs):
        mcrv = multi.SurfaceContainer([surf, surf1])
        mcrv.vis = VisMPL.VisSurface()
        mcrv.render()

    def sample_points_bspline_surface(self, spline, N):
        parameters = np.random.random((N, 2))
        points = spline.evaluate_list(parameters)
        return np.array(points)

    def regular_parameterization(self, grid_u, grid_v):
        nx, ny = (grid_u, grid_v)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv = np.expand_dims(xv.transpose().flatten(), 1)
        yv = np.expand_dims(yv.transpose().flatten(), 1)
        parameters = np.concatenate([xv, yv], 1)
        return parameters

    def boundary_parameterization(self, grid_u):
        u = np.arange(grid_u)
        zeros = np.zeros(grid_u)
        ones = np.ones(grid_u)

        parameters = [np.stack([zeros, u], 1)]
        parameters += [np.stack([np.arange(1, grid_u), np.zeros(grid_u - 1)], 1)]
        parameters += [np.stack([np.arange(1, grid_u), np.ones(grid_u - 1) * (grid_u - 1)], 1)]
        parameters += [np.stack([np.ones(grid_u - 2) * (grid_u - 1), np.arange(1, grid_u - 1)], 1)]
        parameters = np.concatenate(parameters, 0)
        return parameters / (grid_u - 1)


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


def fit_surface(points, size_u, size_v, degree_u=3, degree_v=3, regular_grids=False):
    fitted_surface = fitting.approximate_surface(
        points,
        size_u=size_u,
        size_v=size_v,
        degree_u=degree_u,
        degree_v=degree_v,
        ctrlpts_size_u=10,
        ctrlpts_size_v=10,
    )

    if regular_grids:
        parameters = regular_parameterization(25, 25)
    else:
        parameters = np.random.random((3000, 2))
    fitted_points = fitted_surface.evaluate_list(parameters)
    return fitted_surface, fitted_points


def regular_parameterization(grid_u, grid_v):
    nx, ny = (grid_u, grid_v)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = np.expand_dims(xv.transpose().flatten(), 1)
    yv = np.expand_dims(yv.transpose().flatten(), 1)
    parameters = np.concatenate([xv, yv], 1)
    return parameters

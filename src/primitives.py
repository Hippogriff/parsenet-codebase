"""
This defines the distance from a geometric primitive. The idea is to
sample points from the ground truth surface and find the distance of
these points from the predicted point cloud.
"""
import numpy as np
from src.utils import chamfer_distance_single_shape
import torch
from src.fitting_utils import match
import copy
from src.guard import guard_sqrt, guard_exp


EPS = np.finfo(np.float32).eps


class ResidualLoss:
    """
    Defines distance of points sampled on a patch with corresponding
    predicted patch for different primitives. There is a closed form
    formula for distance from geometric primitives, whereas for splines
    we use chamfer distance as an approximation.
    """
    def __init__(self, reduce=True, one_side=False):
        cp_distance = ComputePrimitiveDistance(reduce, one_side=one_side)
        self.routines = {"torus": cp_distance.distance_from_torus,
                         "sphere": cp_distance.distance_from_sphere,
                        "cylinder": cp_distance.distance_from_cylinder,
                        "cone": cp_distance.distance_from_cone,
                        "plane": cp_distance.distance_from_plane,
                         "closed-spline": cp_distance.distance_from_bspline,
                         "open-spline": cp_distance.distance_from_bspline}
        

    def residual_loss(self, Points, parameters, sqrt=False):
        distances = {}
        for k, v in parameters.items():
            if v is None:
                # degenerate case of primitives that are small
                continue
            dist = self.routines[v[0]](points=Points[k], params=v[1:], sqrt=sqrt)
            distances[k] = [v[0], dist]
        return distances


class ComputePrimitiveDistance:
    def __init__(self, reduce=True, one_side=False):
        """
        This defines a differentiable routines that gives
        distance of a point from a surface of a predicted geometric
        primitive.
        # TODO Define closed form distance of point from bspline surface.
        """
        self.reduce = reduce
        self.one_side = one_side

    def distance_from_torus(self, points, params, sqrt=False):
        """
        Distance of points from the torus
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, major_radius \in R+, minor_radius \in R+
        """
        axis, center, major_radius, minor_radius = params
        axis = axis.reshape((3, 1)) / torch.norm(axis, p=2)
        center = center.reshape((1, 3))
        
        center2points = points - center
        z_new = center2points @ axis # N x 1

        x_new = guard_sqrt(torch.sum(center2points ** 2, 1, keepdim=True) - z_new ** 2)  # N x 1

        # min distance for right circle
        right_dst = (guard_sqrt((x_new - major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        # min distance for left circle
        left_dst = (guard_sqrt((x_new + major_radius) ** 2 + z_new ** 2) - minor_radius) ** 2

        distance = torch.min(right_dst, left_dst)
        distance = distance.squeeze()

        if sqrt:
            distance = guard_sqrt(distance)

        if self.reduce:
            distance = torch.mean(distance)
        return distance


    def distance_from_plane(self, points, params, sqrt=False):
        """
        Distance of points from the plane
        :param points: N x 3
        :param params: a: 3 x 1, d \in R
        """
        a, d = params
        a = a.reshape((3, 1))

        # check for the orientation
        try:
            distance = torch.sum((points @ a - d) ** 2, 1)
        except:
            import ipdb; ipdb.set_trace()
            
        if sqrt:
            distance = guard_sqrt(distance)
        if self.reduce:
            distance = torch.mean(distance)
            
        # Note that this is distance square
        return distance

    def distance_from_sphere(self, points, params, sqrt=False):
        """
        Distance of points from the sphere
        :param points: N x 3
        :param params: c: 3 x 1, radius \in R
        """
        center, radius = params
        center = center.reshape((1, 3))
        distance = (torch.norm(points - center, p=2, dim=1) - radius) ** 2
        if sqrt:
            distance = guard_sqrt(distance)
            
        if self.reduce:
            distance = torch.mean(distance)
        return distance

    def distance_from_cylinder(self, points, params, sqrt=False):
        """
        Distance of points from the cylinder.
        :param points: N x 3
        :param params: axis: 3 x 1, center: 1 x 3, radius \in R
        """
        # axis: 3 x 1, center: 1 x 3
        axis, center, radius = params
        center = center.reshape((1, 3))
        axis = axis.reshape((3, 1))
        
        v = points - center
        prj = (v @ axis) ** 2

        # this is going negative at some point! fix it. Numerical issues.
        # voilating pythagoras
        dist_from_surface = torch.sum(v * v, 1) - prj[:, 0]
        dist_from_surface = torch.clamp(dist_from_surface, min=1e-5)

        distance = torch.sqrt(dist_from_surface) - radius
        # distance.register_hook(self.print_norm)
        distance = distance ** 2

        if sqrt:
            distance = guard_sqrt(distance)

        if torch.sum(torch.isnan(distance)):
            import ipdb; ipdb.set_trace()
        if self.reduce:
            distance = torch.mean(distance)

        return distance

    def print_norm(self, x):
        print ("printing norm 2", torch.norm(x))

    def distance_from_cone(self, points, params, sqrt=False):
        # axis: 3 x 1
        apex, axis, theta = params
        apex = apex.reshape((1, 3))
        axis = axis.reshape((3, 1))
        
        N = points.shape[0]

        # pi_2 = torch.ones(N).cuda()
        try:
            v = points - apex + 1e-8
        except:
            import ipdb; ipdb.set_trace()
        mod_v = torch.norm(v, dim=1, p=2)
        alpha_x = (v @ axis)[:, 0] / (mod_v + 1e-7)
        alpha_x = torch.clamp(alpha_x, min=-.999, max=0.999)

        # safe gaurding against arc cos derivate going at +1/-1.
        alpha = torch.acos(alpha_x)

        dist_angle = torch.clamp(torch.abs(alpha - theta), max=3.142 / 2.0)
        
        distance = (mod_v * torch.sin(dist_angle)) ** 2

        if sqrt:
            distance = guard_sqrt(distance)
        if self.reduce:
            distance = torch.mean(distance)
        return distance

    def distance_from_bspline(self, points, params, sqrt=False):
        """
        This is a rather approximation, where we sample points on the original
        bspline surface and store it in bspline_points, and we also sample
        points on the predicted bspline surface are store them in `points`
        """
        # Need to define weighted distance.
        bspline_points = params[0][0]
        return chamfer_distance_single_shape(bspline_points, points, one_side=self.one_side, sqrt=sqrt, reduce=self.reduce)


class SaveParameters:
    def __init__(self):
        """
        Defines protocol for saving and loading parameter per shape.
        """
        from src.primitive_forward import Fit
        self.fit = Fit()
        pass
    
    def save(self, parameters, labels, cluster_ids, primitives, pred_primitives, path, if_save=True):
        """
        Save parameters predicted by an algorithm.
        :param parameters: dictionary containing predicted parameters. Note
        that the key of these parameters is exactly the label id of that part
        produced by that algorithm.
        :param cluster_dsi: predicted labels per point
        :param matching: defines matching of how the predicted label ids (first column)
        matches with the ground truth label ids (second column). This is used for evaluation.
        :param path: path where this dictionary to be stored.
        :param if_save: whether to save the results or not.
        """
        out_dict = {}
        for k, v in parameters.items():
            if v is None:
                continue
            elif v[0] == "cylinder":
                axis = v[1].data.cpu().numpy()
                center = v[2].data.cpu().numpy()
                radius = v[3].item()
                out_dict[k] = ["cylinder", axis, center, radius]
                
            elif v[0] == "plane":
                normals = v[1].data.cpu().numpy()
                distance = v[2].item()
                out_dict[k] = ["plane", normals, distance]
                
            elif v[0] == "cone":
                apex = v[1].data.cpu().numpy()
                axis = v[2].data.cpu().numpy()
                theta = v[3].data.cpu().numpy()
                out_dict[k] = ["cone", apex, axis, theta]

            elif v[0] == "closed-spline":
                control_points = v[1].data.cpu().numpy()
                out_dict[k] = ["closed-spline", control_points]

            elif v[0] == "open-spline":
                control_points = v[1].data.cpu().numpy()
                out_dict[k] = ["open-spline", control_points]
            
            elif v[0] == "sphere":
                center = v[1].data.cpu().numpy()
                radius = v[2].item()
                out_dict[k] = ["sphere", center, radius]

        out_put = {"primitive_dict": out_dict}
        out_put["seg_id"] = cluster_ids.astype(np.float32)
        out_put["labels"] = labels.astype(np.float32)
        out_put["primitives"] = primitives.astype(np.float32)
        out_put["pred_primitives"] = pred_primitives.astype(np.float32)

        if if_save:
            np.save(path, out_dict)
        return out_put


    def load(self, data):
        """
        Loads the dataset in the format suitable for the evaluation.
        """
        points = data["points"]
        normals = data["normals"]
        labels = data["labels"]
        primitives = data["primitives"]
        primitives = copy.deepcopy(primitives)
        
        try:
            cluster_ids = data["seg_id_RANSAC"]
        except:
            cluster_ids = data["seg_id"]
        parameters = data["primitive_dict"]
        
        rows, cols, unique_target, unique_pred = match(labels, cluster_ids)
        gtpoints = {}
        for k in range(rows.shape[0]):
            if not (parameters.get(k) is None):
                v = parameters[k]
                for index, j in enumerate(v):
                    if index == 0:
                        continue
                    try:
                        v[index] = torch.from_numpy(j.astype(np.float32)).cuda()
                    except:
                        v[index] = torch.tensor(j).cuda()
                indices = labels == cols[k]
                # only include the surface patches that are matched
                if np.sum(indices) > 0:
                    gtpoints[k] = torch.from_numpy(points[indices].astype(np.float32)).cuda()
                else:
                    parameters.pop(k)
        return parameters, gtpoints


    def load_parameters(self, data, bit_mapping=False):
        """
        Samples points from the surface for the purpose of visualization.
        The sampled points are in grid and can be tesellated directly.
        """
        points = data["points"]
        normals = data["normals"]
        labels = data["labels"]

        primitives = data["primitives"]

        cluster_ids = data["seg_id"]
        primitive_dict = data["primitive_dict"]
        for k, v in primitive_dict.items():
            if v[0] == "cylinder":
                sampled_points, sampled_normals = self.fit.sample_cylinder_trim(v[3], v[2], v[1], points[cluster_ids == k])
                input_points = points[cluster_ids == k]
                if bit_mapping:
                    indices = self.bit_map(input_points, sampled_points)
                else:
                    indices = np.arange(sampled_points.shape[0])
                v.append(sampled_points[indices])
                v.append(sampled_normals[indices])
                
            if v[0] == "cone":
                sampled_points, sampled_normals = self.fit.sample_cone_trim(v[1], v[2], v[3], points[cluster_ids == k])
                # sampled_points, sampled_normals = self.fit.sample_cone(v[1], v[2], v[3])
                input_points = points[cluster_ids == k]
                # indices = self.bit_map(input_points, sampled_points)
                if bit_mapping:
                    indices = self.bit_map(input_points, sampled_points)
                else:
                    indices = np.arange(sampled_points.shape[0])
                v.append(sampled_points[indices])
                v.append(sampled_normals[indices])

            if v[0] == "plane":
                sampled_points = self.fit.sample_plane(v[2], v[1], np.mean(points[cluster_ids == k], 0))
                
                sampled_normals = np.concatenate([v[1].reshape((1, 3))] * sampled_points.shape[0], 0)
                input_points = points[cluster_ids == k]
                if bit_mapping:
                    indices = self.bit_map(input_points, sampled_points)
                else:
                    indices = np.arange(sampled_points.shape[0])
                
                v.append(sampled_points[indices])
                v.append(sampled_normals[indices])

            if v[0] == "sphere":
                sampled_points, sampled_normals = self.fit.sample_sphere(v[2], v[1])
                input_points = points[cluster_ids == k]
                if bit_mapping:
                    indices = self.bit_map(input_points, sampled_points)
                else:
                    indices = np.arange(sampled_points.shape[0])
                
                v.append(sampled_points[indices])
                v.append(sampled_normals[indices])

            if v[0] == "torus":
                axis, center, major_radius, minor_radius = v[1:]
                sampled_points = self.fit.sample_torus(major_radius, minor_radius, center, axis)
                v.append(sampled_points)
                v.append(None)
                
    def bit_map(self, input_points, output_points, thres=0.01):
        input_points = torch.from_numpy(input_points.astype(np.float32)).cuda()
        output_points = torch.from_numpy(output_points.astype(np.float32)).cuda()
        dist = torch.cdist(input_points, output_points).T
        try:
            indices = torch.min(dist, 1)[0] < thres
        except:
            import ipdb; ipdb.set_trace()
        return indices.data.cpu().numpy()
        

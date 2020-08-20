"""
This script defines dataset loading for the segmentation task on ABC dataset.
"""

import h5py
import numpy as np
from src.augment_utils import rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, random_scale_point_cloud, rotate_point_cloud


EPS = np.finfo(np.float32).eps


class Dataset:
    def __init__(self,
                 batch_size,
                 train_size=None,
                 val_size=None,
                 test_size=None,
                 normals=False,
                 primitives=False,
                 if_train_data=True):
        """
        Dataset of point cloud from ABC dataset.
        :param root_path:
        :param batch_size:
        :param if_train_data: since training dataset is large and consumes RAM,
        we can optionally choose to not load it.
        """
        self.batch_size = batch_size
        self.normals = normals
        self.primitives = primitives
        self.augment_routines = [rotate_perturbation_point_cloud, jitter_point_cloud, shift_point_cloud, random_scale_point_cloud, rotate_point_cloud]
        
        if if_train_data:
            with h5py.File("data/shapes/train_data.h5", "r") as hf:
                train_points = np.array(hf.get("points"))
                train_labels = np.array(hf.get("labels"))
                if normals:
                    train_normals = np.array(hf.get("normals"))
                if primitives:
                    train_primitives = np.array(hf.get("prim"))
            train_points = train_points[0:train_size].astype(np.float32)
            train_labels = train_labels[0:train_size]
            self.train_normals = train_normals[0:train_size].astype(np.float32)
            self.train_primitives = train_primitives[0:train_size]
            means = np.mean(train_points, 1)
            means = np.expand_dims(means, 1)
        
            self.train_points = (train_points - means)
            self.train_labels = train_labels

        
        with h5py.File("data/shapes/val_data.h5", "r") as hf:
            val_points = np.array(hf.get("points"))
            val_labels = np.array(hf.get("labels"))
            if normals:
                val_normals = np.array(hf.get("normals"))
            if primitives:
                val_primitives = np.array(hf.get("prim"))

        with h5py.File("data/shapes/test_data.h5", "r") as hf:
            test_points = np.array(hf.get("points"))
            test_labels = np.array(hf.get("labels"))
            if normals:
                test_normals = np.array(hf.get("normals"))
            if primitives:
                test_primitives = np.array(hf.get("prim"))
                
        val_points = val_points[0:val_size].astype(np.float32)
        val_labels = val_labels[0:val_size]

        test_points = test_points[0:test_size].astype(np.float32)
        test_labels = test_labels[0:test_size]

        if normals:
            self.val_normals = val_normals[0:val_size].astype(np.float32)
            self.test_normals = test_normals[0:test_size].astype(np.float32)

        if primitives:
            self.val_primitives = val_primitives[0:val_size]
            self.test_primitives = test_primitives[0:test_size]
        
        means = np.mean(test_points, 1)
        means = np.expand_dims(means, 1)
        self.test_points = (test_points - means)
        self.test_labels = test_labels
        
        means = np.mean(val_points, 1)
        means = np.expand_dims(means, 1)
        self.val_points = (val_points - means)
        self.val_labels = val_labels
        
    def get_train(self, randomize=False, augment=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        train_size = self.train_points.shape[0]
        while (True):
            l = np.arange(train_size)
            if randomize:
                np.random.shuffle(l)
            train_points = self.train_points[l]
            train_labels = self.train_labels[l]

            if self.normals:
                train_normals = self.train_normals[l]
            if self.primitives:
                train_primitives = self.train_primitives[l]
                
            for i in range(train_size // self.batch_size):
                points = train_points[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                if self.normals:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]
                
                if augment:
                    points = self.augment_routines[np.random.choice(np.arange(5))](points)

                if if_normal_noise:
                    normals = train_normals[i * self.batch_size:(i + 1) * self.batch_size]
                    
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)
                    
                labels = train_labels[i * self.batch_size:(i + 1) * self.batch_size]

                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T

                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                            # TODO make the same changes to normals also.
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)
                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = train_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)
                
                yield return_items

    def get_test(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        test_size = self.test_points.shape[0]
        batch_size = self.batch_size

        while (True):
            for i in range(test_size // batch_size):
                points = self.test_points[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                labels = self.test_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.test_normals[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                new_points = []
                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T
                        
                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.test_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)
                yield return_items
                
    def get_val(self, randomize=False, anisotropic=False, align_canonical=False, if_normal_noise=False):
        val_size = self.val_points.shape[0]
        batch_size = self.batch_size

        while (True):
            for i in range(val_size // batch_size):
                points = self.val_points[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                labels = self.val_labels[i * self.batch_size:(i + 1) * self.batch_size]
                if self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                if if_normal_noise and self.normals:
                    normals = self.val_normals[i * self.batch_size:(i + 1) *
                                   self.batch_size]
                    noise = normals * np.clip(np.random.randn(1, points.shape[1], 1) * 0.01, a_min=-0.01, a_max=0.01)
                    points = points + noise.astype(np.float32)

                new_points = []
                for j in range(self.batch_size):
                    if align_canonical:
                        S, U = self.pca_numpy(points[j])
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points[j] = (R @ points[j].T).T
                        if self.normals:
                            normals[j] = (R @ normals[j].T).T
                        
                        std = np.max(points[j], 0) - np.min(points[j], 0)
                        if anisotropic:
                            points[j] = points[j] / (std.reshape((1, 3)) + EPS)
                        else:
                            points[j] = points[j] / (np.max(std) + EPS)

                return_items = [points, labels]
                if self.normals:
                    return_items.append(normals)
                else:
                    return_items.append(None)

                if self.primitives:
                    primitives = self.val_primitives[i * self.batch_size:(i + 1) * self.batch_size]
                    return_items.append(primitives)
                else:
                    return_items.append(None)
                yield return_items

    def normalize_points(self, points, normals, anisotropic=False):
        points = points - np.mean(points, 0, keepdims=True)
        noise = normals * np.clip(np.random.randn(points.shape[0], 1) * 0.01, a_min=-0.01, a_max=0.01)
        points = points + noise.astype(np.float32)

        S, U = self.pca_numpy(points)
        smallest_ev = U[:, np.argmin(S)]
        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
        # rotate input points such that the minor principal
        # axis aligns with x axis.
        points = (R @ points.T).T
        normals = (R @ normals.T).T
        std = np.max(points, 0) - np.min(points, 0)
        if anisotropic:
            points = points / (std.reshape((1, 3)) + EPS)
        else:
            points = points / (np.max(std) + EPS)        
        return points.astype(np.float32), normals.astype(np.float32)


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

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U

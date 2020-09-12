import h5py
import numpy as np
from torch.utils.data import Dataset

from src.augment_utils import Augment
from src.curve_utils import DrawSurfs

augment = Augment()

EPS = np.finfo(np.float32).eps


class generator_iter(Dataset):
    """This is a helper function to be used in the parallel data loading using Pytorch
    DataLoader class"""

    def __init__(self, generator, train_size):
        self.generator = generator
        self.train_size = train_size

    def __len__(self):
        return self.train_size

    def __getitem__(self, idx):
        return next(self.generator)


class DataSetControlPointsPoisson:
    def __init__(self, path, batch_size, size_u=20, size_v=20, splits={}, closed=False):
        """
        :param path: path to h5py file that stores the dataset
        :param batch_size: batch size
        :param num_points: number of 
        :param size_u:
        :param size_v:
        :param splits:
        """
        self.path = path
        self.batch_size = batch_size
        self.size_u = size_u
        self.size_v = size_v
        all_files = []

        count = 0
        self.train_size = splits["train"]
        self.val_size = splits["val"]
        self.test_size = splits["test"]

        # load the points and control points
        with h5py.File(path, "r") as hf:
            points = np.array(hf.get(name="points")).astype(np.float32)
            control_points = np.array(hf.get(name="controlpoints")).astype(np.float32)

        np.random.seed(0)
        List = np.arange(points.shape[0])
        np.random.shuffle(List)
        points = points[List]
        control_points = control_points[List]
        if closed:
            # closed spline has different split
            self.train_points = points[0:28000]
            self.val_points = points[28000:31000]
            self.test_points = points[31000:]

            self.train_control_points = control_points[0:28000]
            self.val_control_points = control_points[28000:31000]
            self.test_control_points = control_points[31000:]
        else:
            self.train_points = points[0:50000]
            self.val_points = points[50000:60000]
            self.test_points = points[60000:]

            self.train_control_points = control_points[0:50000]
            self.val_control_points = control_points[50000:60000]
            self.test_control_points = control_points[60000:]

        self.draw = DrawSurfs()

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

    def load_train_data(self, if_regular_points=False, align_canonical=False, anisotropic=False, if_augment=False):
        while True:
            for batch_id in range(self.train_size // self.batch_size - 1):
                Points = []
                Parameters = []
                controlpoints = []
                scales = []
                RS = []

                for i in range(self.batch_size):
                    points = self.train_points[batch_id * self.batch_size + i]
                    mean = np.mean(points, 0)
                    points = points - mean

                    if align_canonical:
                        S, U = self.pca_numpy(points)
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = R @ points.T
                        points = points.T
                        RS.append(R)

                    if anisotropic:
                        std = np.abs(np.max(points, 0) - np.min(points, 0))
                        std = std.reshape((1, 3))
                        points = points / (std + EPS)
                    else:
                        std = np.max(np.max(points, 0) - np.min(points, 0))
                        points = points / std

                    scales.append(std)
                    Points.append(points)
                    cntrl_point = self.train_control_points[batch_id * self.batch_size + i]
                    cntrl_point = cntrl_point - mean.reshape((1, 1, 3))

                    if align_canonical:
                        cntrl_point = cntrl_point.reshape((self.size_u * self.size_v, 3))
                        cntrl_point = R @ cntrl_point.T
                        cntrl_point = np.reshape(cntrl_point.T, (self.size_u, self.size_v, 3))

                    if anisotropic:
                        cntrl_point = cntrl_point / (std.reshape((1, 1, 3)) + EPS)
                    else:
                        cntrl_point = cntrl_point / std
                    controlpoints.append(cntrl_point)
                controlpoints = np.stack(controlpoints, 0)
                Points = np.stack(Points, 0)
                if if_augment:
                    Points = augment.augment(Points)
                    Points = Points.astype(np.float32)
                yield [Points, None, controlpoints, scales, RS]

    def load_val_data(self, if_regular_points=False, align_canonical=False, anisotropic=False, if_augment=False):
        while True:
            for batch_id in range(self.val_size // self.batch_size - 1):
                Points = []
                Parameters = []
                controlpoints = []
                scales = []
                RS = []
                for i in range(self.batch_size):
                    points = self.val_points[batch_id * self.batch_size + i]
                    mean = np.mean(points, 0)
                    points = points - mean

                    if align_canonical:
                        S, U = self.pca_numpy(points)
                        smallest_ev = U[:, np.argmin(S)]
                        R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                        # rotate input points such that the minor principal
                        # axis aligns with x axis.
                        points = R @ points.T
                        points = points.T
                        RS.append(R)

                    if anisotropic:
                        std = np.abs(np.max(points, 0) - np.min(points, 0))
                        std = std.reshape((1, 3))
                        points = points / (std + EPS)
                    else:
                        std = np.max(np.max(points, 0) - np.min(points, 0))
                        points = points / std

                    scales.append(std)
                    Points.append(points)
                    cntrl_point = self.val_control_points[batch_id * self.batch_size + i]
                    cntrl_point = cntrl_point - mean.reshape((1, 1, 3))

                    if align_canonical:
                        cntrl_point = cntrl_point.reshape((self.size_u * self.size_v, 3))
                        cntrl_point = R @ cntrl_point.T
                        cntrl_point = np.reshape(cntrl_point.T, (self.size_u, self.size_v, 3))

                    if anisotropic:
                        cntrl_point = cntrl_point / (std.reshape((1, 1, 3)) + EPS)
                    else:
                        cntrl_point = cntrl_point / std
                    controlpoints.append(cntrl_point)
                controlpoints = np.stack(controlpoints, 0)
                Points = np.stack(Points, 0)
                if if_augment:
                    Points = augment.augment(Points)
                    Points = Points.astype(np.float32)
                yield [Points, None, controlpoints, scales, RS]

    def load_test_data(self, if_regular_points=False, align_canonical=False, anisotropic=False, if_augment=False):
        for batch_id in range(self.test_size // self.batch_size):
            Points = []
            controlpoints = []
            scales = []
            RS = []
            for i in range(self.batch_size):
                points = self.test_points[batch_id * self.batch_size + i]
                mean = np.mean(points, 0)
                points = points - mean

                if align_canonical:
                    S, U = self.pca_numpy(points)
                    smallest_ev = U[:, np.argmin(S)]
                    R = self.rotation_matrix_a_to_b(smallest_ev, np.array([1, 0, 0]))
                    # rotate input points such that the minor principal
                    # axis aligns with x axis.
                    points = R @ points.T
                    points = points.T
                    RS.append(R)

                if anisotropic:
                    std = np.abs(np.max(points, 0) - np.min(points, 0))
                    std = std.reshape((1, 3))
                    points = points / (std + EPS)
                else:
                    std = np.max(np.max(points, 0) - np.min(points, 0))
                    # points = points / std

                scales.append(std)
                Points.append(points)
                cntrl_point = self.test_control_points[batch_id * self.batch_size + i]
                cntrl_point = cntrl_point - mean.reshape((1, 1, 3))

                if align_canonical:
                    cntrl_point = cntrl_point.reshape((self.size_u * self.size_v, 3))
                    cntrl_point = R @ cntrl_point.T
                    cntrl_point = np.reshape(cntrl_point.T, (self.size_u, self.size_v, 3))
                if anisotropic:
                    cntrl_point = cntrl_point / (std.reshape((1, 1, 3)) + EPS)
                else:
                    cntrl_point = cntrl_point / std
                controlpoints.append(cntrl_point)

            controlpoints = np.stack(controlpoints, 0)
            Points = np.stack(Points, 0)
            if if_augment:
                Points = augment.augment(Points)
                Points = Points.astype(np.float32)
            yield [Points, None, controlpoints, scales, RS]

    def pca_torch(self, X):
        covariance = torch.transpose(X, 1, 0) @ X
        S, U = torch.eig(covariance, eigenvectors=True)
        return S, U

    def pca_numpy(self, X):
        S, U = np.linalg.eig(X.T @ X)
        return S, U

import numpy as np
from src.VisUtils import tessalate_points
from src.loss import basis_function_one
from src.approximation import generate_bezier_surface_on_grid, uniform_knot_bspline
import h5py


class Zitter:
    def __init__(self):
        nx, ny = (3, 3)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(x, y)

        parameters = np.stack([xv, yv], 2)
        self.control_points = np.concatenate([parameters, np.zeros((3, 3, 1))], 2)
        self.control_points -= np.mean(self.control_points, (0, 1))

    def zitter_middle_z(self):
        zitter = self.control_points.copy()
        zitter[1, 1, -1] = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        return zitter

    def zitter_corner_z_same(self):
        zitter = self.control_points.copy()
        value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        zitter[0, 0, -1] = value
        zitter[2, 2, -1] = value
        zitter[0, 2, -1] = value
        zitter[2, 0, -1] = value
        return zitter

    def zitter_corner_z_diff(self):
        zitter = self.control_points.copy()
        zitter[0, 0, -1] = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        zitter[2, 2, -1] = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        zitter[0, 2, -1] = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        zitter[2, 0, -1] = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        return zitter

    def zitter_one_boundary(self):
        zitter = self.control_points.copy()
        value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
        zitter[0, 0, -1]  = value
        zitter[0, 1, -1] = value
        zitter[0, 2, -1] = value
        return zitter

    def zitter_two_boundaries(self):
        zitter = self.control_points.copy()
        if np.random.random() > 0.5:
            value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
            zitter[2, 0, -1]  = value
            zitter[2, 1, -1] = value
            zitter[2, 2, -1] = value

            value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
            zitter[0, 0, -1]  = value
            zitter[0, 1, -1] = value
            zitter[0, 2, -1] = value
        else:
            value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
            zitter[0, 2, -1]  = value
            zitter[1, 2, -1] = value
            zitter[2, 2, -1] = value

            value = np.clip(np.random.randn(), a_min=-1.0, a_max=1.0)
            zitter[0, 0, -1]  = value
            zitter[1, 0, -1] = value
            zitter[2, 0, -1] = value
        return zitter


class DataSet:
    def __init__(self, path, train_size, test_size, val_size):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size

        # dataset is assumed to be normalized before.
        with h5py.File(path, "r") as hf:
            points = np.array(hf.get("points"))
            control_points = np.array(hf.get("control_points"))

        N = points.shape[0]
        mean = np.mean(points, 1)
        points = points - mean.reshape((N, 1, 3))
        std = np.max(np.max(points, 1) - np.min(points, 1), 1)
        points = points / std.reshape((N, 1, 1))

        control_points = control_points - mean.reshape((N, 1, 1, 3))
        control_points = control_points / std.reshape((N, 1, 1, 1))

        self.train_points = points[0:train_size]
        self.train_cps = control_points[0:train_size]
        l = np.arange(train_size)
        np.random.shuffle(l)
        self.train_points = self.train_points[l]
        self.train_cps = self.train_cps[l]

        self.val_point = points[train_size:train_size + val_size]
        self.val_cps = control_points[train_size:train_size + val_size]

        self.test_points = points[train_size + val_size:train_size + val_size + test_size]
        self.test_cps = control_points[train_size + val_size:train_size + val_size + test_size]

    def get_train_data(self, batch_size):
        while True:
            for i in range(self.train_size // batch_size):
                yield [self.train_points[i * batch_size: (i + 1) * batch_size], self.train_cps[i * batch_size: (i + 1) * batch_size]]
        
    def get_val_data(self, batch_size):
        for i in range(self.val_size // batch_size):
            yield [self.val_points[i * batch_size: (i + 1) * batch_size], self.val_cps[i * batch_size: (i + 1) * batch_size]]

    def get_test_data(self, batch_size):
        for i in range(self.test_size // batch_size):
            yield [self.test_points[i * batch_size: (i + 1) * batch_size], self.test_cps[i * batch_size: (i + 1) * batch_size]]

import numpy as np
import torch


def test_cone():
    points, normals = fitting.sample_cone(np.array([0.0, 0.0, 0]),
                                          np.array([1, 1, 0]), np.pi / 3)

    apex, axis, theta = fitting.fit_cone_torch(torch.from_numpy(points),
                                               torch.from_numpy(normals),
                                               torch.from_numpy(np.ones((1000, 1))))

    visualize_point_cloud(points, normals=normals, viz=True)

    new_points, new_normals = fitting.sample_cone(apex.data.numpy().reshape(3),
                                                  axis.data.numpy().reshape(3), theta.item())
    visualize_point_cloud(np.concatenate([points, new_points], 0), normals=np.concatenate([normals, new_normals], 0),
                          viz=True)


def test_cylinder():
    points, normals = fitting.sample_cylinder(1, np.array([0, 0, 0]), np.array([1, 2, 0]) / np.sqrt(5))

    points = points.astype(np.float32)
    normals = normals.astype(np.float32)
    weights = np.ones((100, 1), dtype=np.float32)
    axis, center, radius = fitting.fit_cylinder_torch(torch.from_numpy(points),
                                                      torch.from_numpy(normals), torch.from_numpy(weights))

    new_points, new_normals = fitting.sample_cylinder(1, center.data.numpy().reshape(3), axis.data.numpy().reshape(3))

    visualize_point_cloud(points, normals=normals, viz=True)
    colors = np.zeros((200, 3))
    colors[0:100, 0] = 1
    print(center, radius, axis)
    visualize_point_cloud(np.concatenate([points, new_points], 0), normals=np.concatenate([normals, new_normals], 0),
                          colors=colors, viz=True)


def test_sphere():
    points, normals = fitting.sample_sphere(1, np.array([0, 0, 0]))
    print(np.mean(points, 0))
    points = points.astype(np.float32)
    # normals = normals.astype(np.float32)
    weights = np.ones((1000, 1), dtype=np.float32)
    center, radius = fitting.fit_sphere_torch(torch.from_numpy(points), None, torch.from_numpy(weights))
    # center, radius = fitting.fit_sphere_numpy(points, weights)
    print(center, radius)

    new_points, new_normals = fitting.sample_sphere(radius.item(), center.data.numpy().reshape(3))

    visualize_point_cloud(points, normals=normals, viz=True)
    colors = np.zeros((200, 3))
    colors[0:100, 0] = 1

    visualize_point_cloud(np.concatenate([points, new_points], 0), normals=np.concatenate([normals, new_normals], 0),
                          colors=colors, viz=True)


# normals = normals.astype(np.float32)
def grad_check_sphere():
    points, normals = fitting.sample_sphere(1, np.array([0, 0, 0]))
    points = points.astype(np.float64)
    weights = torch.from_numpy(np.ones((100, 1), dtype=np.float64))
    weights.requires_grad = True

    def func(weights):
        center, radius = fitting.fit_sphere_torch(torch.from_numpy(points), weights)
        return torch.mean(center)

    print(gradcheck(func, weights))


def grad_check_cone():
    points, normals = fitting.sample_cone(np.array([0.10, 1.0, 2.0]),
                                          np.array([1, 1, 0]), np.pi / 3)

    weights = torch.from_numpy(np.ones((1000, 1), dtype=np.float64))
    weights.requires_grad = True

    def func_apex(weights):
        apex, axis, theta = fitting.fit_cone_torch(torch.from_numpy(points),
                                                   torch.from_numpy(normals),
                                                   weights)
        return torch.mean(theta)

    print(gradcheck(func_apex, weights))


def grad_check_cone():
    points, normals = fitting.sample_cone(np.array([0.10, 1.0, 2.0]),
                                          np.array([1, 1, 0]), np.pi / 3)

    weights = torch.from_numpy(np.ones((1000, 1), dtype=np.float64))
    weights.requires_grad = True

    def func_apex(weights):
        apex, axis, theta = fitting.fit_cone_torch(torch.from_numpy(points),
                                                   torch.from_numpy(normals),
                                                   weights)
        return torch.mean(theta)

    print(gradcheck(func_apex, weights))


def grad_check_cylinder():
    points, normals = fitting.sample_cylinder(1, np.array([1, 2, 0.3]), np.array([1, 0, 1]) / np.sqrt(2))

    points = points.astype(np.float64)
    normals = normals.astype(np.float64)
    weights = torch.from_numpy(np.ones((100, 1), dtype=np.float64))
    weights.requires_grad = True

    def func_axis(weights):
        axis, center, radius = fitting.fit_cylinder_torch(torch.from_numpy(points),
                                                          torch.from_numpy(normals), weights)
        # print(axis, center, radius)
        return torch.mean(axis)

    print(gradcheck(func_axis, weights))

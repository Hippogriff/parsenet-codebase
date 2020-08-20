import numpy as np
from scipy.special import comb


def bernstein_polynomial(n):
    """
    n: degree of the polynomial
    """
    N = np.ones(n + 1, dtype=np.int32) * n
    K = np.arange(n + 1)
    basis = comb(N, K)
    return basis.reshape((1, n + 1))

def bernstein_tensor(t, basis):
    """
    t: L x 1
    basis: 1 x n + 1
    """
    n = basis.shape[1] - 1
    T = []
    for i in range(n + 1):
        T.append((t ** i) * ((1.0 - t) ** (n-i)))
    T = np.concatenate(T, 1)
    basis_tensor = T * basis
    return basis_tensor

basis = bernstein_polynomial(3)
t = np.random.random((100, 1))
basis_u = bernstein_tensor(t, basis)

t = np.random.random((100, 1))
basis_v = bernstein_tensor(t, basis)


p = np.array([[0, 0, 0],
        [0.33, 0, 0.5],
        [0.66, 0, 0.5],
        [1, 0, 0],

        [0, 0.33, 0.5],
        [0.33, 0.33, 1],
        [0.66, 0.33, 1],
        [1, 0.33, 0.5],

        [0, 0.66, -0.5],
        [0.33, 0.66, -1],
        [0.66, 0.66, -1],
        [1, 0.66, -0.5],

        [0, 1, 0],
        [0.33, 1, 0.5],
        [0.66, 1, 0.5],
        [1, 1, 0]])

cp = p.reshape((4, 4, 3))


points = []
for i in range(3):
    points.append(np.matmul(np.matmul(basis_u, cp[:, :, i]), np.transpose(basis_v)))

points = np.stack(points, 2)

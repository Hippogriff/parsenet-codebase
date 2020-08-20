import numpy as np
import geomdl
from scipy.special import comb
import time


def fit_curve(points, degree, num_ctrls):
    num_points, dim = points.shape
    num_cpts = num_ctrls

    # Get uk
    uk = compute_params_curve(points, use_centripetal=False)

    # Compute knot vector
    kv = compute_knot_vector2(
        degree, num_points=num_points, num_cpts=num_ctrls, params=uk
    )

    matrix_n = []
    for i in range(0, num_points):
        m_temp = []
        for j in range(0, num_cpts):
            m_temp.append(basis_function_one(degree, kv, j, uk[i]))
        matrix_n.append(m_temp)

    matrix_n = np.array(matrix_n)
    #     t = np.linalg.lstsq(matrix_n, points)
    ps_inv = np.linalg.inv(np.matmul(np.transpose(matrix_n), matrix_n))
    ps_inv = np.matmul(ps_inv, np.transpose(matrix_n))
    result = np.matmul(ps_inv, points)
    return result


class BSpline:
    def __init__(self):
        pass

    def evaluate_param(self, param, control_points, knot_vectors_u, knot_vectors_v, degree_u, degree_v):
        control_points_u, control_points_v = control_points.shape[0], control_points.shape[1]
        nu = []
        for j in range(0, control_points_u):
            nu.append(self.basis_function_one(degree_u, knot_vectors_u, j, param[0]))
        nu = np.array(nu).reshape(control_points_u, 1)

        nv = []
        for j in range(0, control_points_v):
            nv.append(self.basis_function_one(degree_v, knot_vectors_v, j, param[1]))
        nv = np.array(nv).reshape(control_points_v, 1)

        points = []
        for i in range(3):
            points.append(np.matmul(np.matmul(nu.T, control_points[:, :, i]), nv))
        points = np.array(points).reshape(1, 3)
        return points

    def basis_functions(self, param, control_points_u, control_points_v, knot_vectors_u, knot_vectors_v, degree_u, degree_v):
        """
        Returns the basis function in u and v direction to be used to compute the
        renormalization factor for the shifting control point grids.
        """
        nu = []
        for j in range(0, control_points_u):
            nu.append(self.basis_function_one(degree_u, knot_vectors_u, j, param[0]))
        nu = np.array(nu).reshape(control_points_u, 1)

        nv = []
        for j in range(0, control_points_v):
            nv.append(self.basis_function_one(degree_v, knot_vectors_v, j, param[1]))
        nv = np.array(nv).reshape(control_points_v, 1)
        return nu, nv

    def create_geomdl_surface(self, control_points, ku, kv, degree_u, degree_v):
        bspline = geomdl.BSpline.Surface()
        cu = control_points.shape[0]
        cv = control_points.shape[1]
        bspline.degree_u = degree_u
        bspline.ctrlpts_size_u = cu
        bspline.ctrlpts_size_v = cv
        bspline.degree_v = degree_v
        bspline.knotvector_u = ku.tolist()
        bspline.knotvector_v = kv.tolist()
        bspline.ctrlpts2d = control_points.tolist()
        return bspline
    
    def evaluate_params(self, params, control_points, knot_vectors_u, knot_vectors_v, degree_u, degree_v):
        control_points_u, control_points_v = control_points.shape[0], control_points.shape[1]
        num_points = params.shape[0]

        nu = []
        nu = np.zeros((num_points, control_points_u))
        for i in range(num_points):
            basis = []
            for j in range(0, control_points_u):
                nu[i, j] = self.basis_function_one(degree_u, knot_vectors_u, j, params[i, 0])
        nu = np.expand_dims(nu, 2)

        nv = np.zeros((num_points, control_points_v))
        for i in range(num_points):
            for j in range(0, control_points_v):
                nv[i, j] = self.basis_function_one(degree_v, knot_vectors_v, j, params[i, 1])
        nv = np.expand_dims(nv, 1)

        points = []
        basis = np.matmul(nu, nv)
        basis = np.reshape(basis, (num_points, control_points_v * control_points_u))
        for i in range(3):
            cntrl_pts = np.reshape(control_points[:, :, i], (control_points_u * control_points_v, 1))
            points.append(np.matmul(basis, cntrl_pts))
        points = np.stack(points, 1)
        return points

    def fit_surface(self, points, size_u, size_v, degree_u=2, degree_v=2, control_points_u=None, control_points_v=None):
        """
        Given points in grid format, this function performs a least square fitting
        to fit bspline of given degree. This involves first computing u,v for each
        input points along with knot vectors.
        :param points: points of size Nx3, note that they are gridded originally of size
        N^(1/2) x N^(1/2) x 3
        :param size_u: u size of the grid
        :param size_v: v size of the grid, note that size_u x size_v = N
        :param control_points_u: control points in u direction
        :param control_points_v: control points in v direction
        """
        points = np.array(points)
        points_ = points.reshape((size_u, size_v, 3))

        if (not control_points_u):
            control_points_u = size_u - 1
        if (not control_points_v):
            control_points_v = size_v - 1
        uk, vl = self.compute_params_surface(points_, size_u=control_points_u, size_v=control_points_v)
    
        # Set up knot vectors depending on the number of control points
        kv_u = self.compute_knot_vector2(degree_u, size_u, control_points_u, uk)
        kv_v = self.compute_knot_vector2(degree_v, size_v, control_points_v, vl)

        nu = []
        for i in range(0, size_u):
            m_temp = []
            for j in range(0, control_points_u):
                m_temp.append(self.basis_function_one(degree_u, kv_u, j, uk[i]))
            nu.append(m_temp)

        nu = np.array(nu)

        nv = []
        for i in range(0, size_v):
            m_temp = []
            for j in range(0, control_points_v):
                m_temp.append(self.basis_function_one(degree_v, kv_v, j, vl[i]))
            nv.append(m_temp)
        nv = np.array(nv)

        ut_u_inv = np.linalg.inv(np.matmul(np.transpose(nu), nu))
        ut_u_inv_u = np.matmul(ut_u_inv, np.transpose(nu))

        vt_v_inv = np.linalg.inv(np.matmul(np.transpose(nv), nv))
        vt_v_inv_v = np.matmul(nv, vt_v_inv)

        cntrlpts = []
        # use the pseudo inverse formulation
        for i in range(3):
            points_cntrl = list(np.matmul(np.matmul(ut_u_inv_u, points_[:, :, i]), vt_v_inv_v))
            cntrlpts.append(points_cntrl)

        ctrl = np.array(cntrlpts)
        ctrl = np.transpose(ctrl, (1, 2, 0))
        return ctrl, kv_u, kv_v

    def compute_knot_vector2(self, degree, num_points, num_cpts, params):
        """
        Computes a knot vector ensuring that every knot span has at least one
        :param degree:
        :param num_points:
        :param num_cpts:
        :param params:
        :return:
        """
        d = num_points / (num_cpts - degree)
        j = np.arange(1, num_cpts - degree)
        I = np.floor(j * d)
        alpha = j * d - I
        params_dash_small = params[I.astype(np.int32) - 1]
        params_dash_large = params[I.astype(np.int32)]

        kv = alpha * params_dash_large + (1.0 - alpha) * params_dash_small
        extra_1 = np.array([1.0] * (degree + 1))
        extra_0 = np.array([0.0] * (degree + 1))
        kv = np.concatenate([extra_0, kv, extra_1])
        return kv

    def basis_function_one(self, degree, knot_vector, span, knot):
        """ Computes the value of a basis function for a single parameter.

        Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.
        :param degree: degree, :math:`p`
        :type degree: int
        :param knot_vector: knot vector
        :type knot_vector: list, tuple
        :param span: knot span, :math:`i`
        :type span: int
        :param knot: knot or parameter, :math:`u`
        :type knot: float
        :return: basis function, :math:`N_{i,p}`
        :rtype: float
        """
        # Special case at boundaries
        if (
            (span == 0 and knot == knot_vector[0])
            or (span == len(knot_vector) - degree - 2)
            and knot == knot_vector[len(knot_vector) - 1]
        ):
            return 1.0

        # Knot is outside of span range
        if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
            return 0.0

        N = [0.0 for _ in range(degree + span + 1)]

        # Initialize the zeroth degree basis functions
        for j in range(0, degree + 1):
            if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
                N[j] = 1.0

        # Computing triangular table of basis functions
        for k in range(1, degree + 1):
            # Detecting zeros saves computations
            saved = 0.0
            if N[0] != 0.0:
                saved = ((knot - knot_vector[span]) * N[0]) / (
                    knot_vector[span + k] - knot_vector[span]
                )

            for j in range(0, degree - k + 1):
                Uleft = knot_vector[span + j + 1]
                Uright = knot_vector[span + j + k + 1]

                # Zero detection
                if N[j + 1] == 0.0:
                    N[j] = saved
                    saved = 0.0
                else:
                    temp = N[j + 1] / (Uright - Uleft)
                    N[j] = saved + (Uright - knot) * temp
                    saved = (knot - Uleft) * temp
        return N[0]

    def compute_params_surface(self, points, size_u, size_v):
        # finding params in v direction
        size_u, size_v = points.shape[0:2]
        params_v = []
        for u in range(size_u):
            temp = self.compute_params_curve(points[u]).reshape((1, size_v))
            params_v.append(temp)
        params_v = np.concatenate(params_v, 0)

        params_v = np.mean(params_v, 0)
        params_u = []
        for v in range(size_v):
            temp = self.compute_params_curve(points[:, v]).reshape((size_u, 1))
            params_u.append(temp)
        params_u = np.concatenate(params_u, 1)

        params_u = np.mean(params_u, 1)
        return params_u, params_v

    def compute_params_curve(self, points, use_centripetal=False):
        """
        Given gridded points, the surface needs to be
        """
        num_points, dim = points.shape
        num_points = points.shape[0]
        points_dash = np.square(points[0:-1] - points[1:])
        points_dash = np.sqrt(np.sum(points_dash, 1))

        # Find the total chord length
        d = np.sum(points_dash)
        points_dash = points_dash / d

        # Divide individual chord lengths by the total chord length
        uk = np.zeros((num_points))
        for i in range(num_points - 1):
            uk[i + 1] = np.sum(points_dash[0 : i + 1])
        return uk


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


def fit_bezier_surface(points, basis_u, basis_v):
    """
    Given basis function basis_u, basis_v, find the control points.
    This is applicable for the gridded points of size N x N x 3.
    basis functions are of size N x (n + 1)
    """
    # N x (n + 1)
    nu = basis_u
    nv = basis_v

    ut_u_inv = np.linalg.inv(np.matmul(np.transpose(nu), nu))
    ut_u_inv_u = np.matmul(ut_u_inv, np.transpose(nu))

    vt_v_inv = np.linalg.inv(np.matmul(np.transpose(nv), nv))
    vt_v_inv_v = np.matmul(nv, vt_v_inv)

    cntrlpts = []
    # use the pseudo inverse formulation
    for i in range(3):
        points_cntrl = list(np.matmul(np.matmul(ut_u_inv_u, points[:, :, i]), vt_v_inv_v))
        cntrlpts.append(points_cntrl)
    ctrl = np.array(cntrlpts)
    ctrl = np.transpose(ctrl, (1, 2, 0))
    return ctrl


def fit_bezier_surface_fit_kronecker(points, basis_u, basis_v):
    """
    Given basis function basis_u, basis_v, find the control points.
    This is applicable for non gridded points of size N x 3 and
    the basis functions are of size N x (n + 1) corresponding to N number
    of points. Also, n + 1 is the number of control points in u direction.
    Note that to get better fitting, points at the boundary should be sampled.
    :param basis_u: bernstein polynomial of size N x (n + 1)
    :param basis_v: bernstein polynomial of size N x (m + 1)
    :return ctrl: control points of size (n + 1) x (m + 1)
    """
    # converts the problem of U x C x V^t = P to U^T x V x C = P
    # that is A^t x X = b form
    A = []
    N = basis_u.shape[0]
    n = basis_v.shape[1] - 1
    for i in range(N):
        A.append(np.matmul(np.transpose(basis_u[i:i+1, :]), basis_v[i:i+1, :]))
    A = np.stack(A, 0)
    A = np.reshape(A, (N, -1))

    cntrl = []
    for i in range(3):
        t = np.linalg.lstsq(A, points[:, i])
        cntrl.append(t[0].reshape((n + 1, n + 1)))
    cntrl = np.stack(cntrl, 2)
    return cntrl


def generate_bezier_surface_on_grid(cp, basis_u, basis_v):
    """
    evaluates the bezier curve with give control points on a grid defined
    by basis_u x basis_v. Only suitable if the points are required to on the grid.
    """
    points = []
    for i in range(3):
        points.append(np.matmul(np.matmul(basis_u, cp[:, :, i]), np.transpose(basis_v)))
    points = np.stack(points, 2)
    return points


def generate_bezier_surface_using_cp_on_grid(cp, n, num_points):
    """
    evaluates the bezier curve with give control points on a grid defined
    by basis_u x basis_v. Only suitable if the points are required to on the grid.
    """
    basis = bernstein_polynomial(n)
    t = np.random.random((num_points, 1))
    basis_v = bernstein_tensor(t, basis)
    basis_u = bernstein_tensor(t, basis)
    points = []
    for i in range(3):
        points.append(np.matmul(np.matmul(basis_u, cp[:, :, i]), np.transpose(basis_v)))
    points = np.stack(points, 2)
    points = np.reshape(points, (num_points ** 2, 3))
    return points


def compute_params_curve(points, use_centripetal=False):
    """
    Given gridded points, the surface needs to be 
    """
    num_points, dim = points.shape
    num_points = points.shape[0]
    points_dash = np.square(points[0:-1] - points[1:])
    points_dash = np.sqrt(np.sum(points_dash, 1))

    # Find the total chord length
    d = np.sum(points_dash)
    points_dash = points_dash / d

    # Divide individual chord lengths by the total chord length
    uk = np.zeros((num_points))
    for i in range(num_points - 1):
        uk[i + 1] = np.sum(points_dash[0 : i + 1])
    return uk

def uniform_knot_bspline(control_points_u, control_points_v, degree_u, degree_v, grid_size=30):
    u = v = np.arange(0., 1, 1 / grid_size)
    
    knots_u = [0.0] * degree_u + np.arange(0, 1.01, 1 / (control_points_u - degree_u)).tolist() + [1.0] * degree_u
    knots_v = [0.0] * degree_v + np.arange(0, 1.01, 1 / (control_points_v - degree_v)).tolist() + [1.0] * degree_v
    
    nu = []
    nu = np.zeros((u.shape[0], control_points_u))
    for i in range(u.shape[0]):
        basis = []
        for j in range(0, control_points_u):
            nu[i, j] = basis_function_one(degree_u, knots_u, j, u[i])
            
    nv = np.zeros((v.shape[0], control_points_v))
    for i in range(v.shape[0]):
        for j in range(0, control_points_v):
            nv[i, j] = basis_function_one(degree_v, knots_v, j, v[i])
    return nu, nv


def basis_function_one(degree, knot_vector, span, knot):
    """ Computes the value of a basis function for a single parameter.

    Implementation of Algorithm 2.4 from The NURBS Book by Piegl & Tiller.
    :param degree: degree, :math:`p`
    :type degree: int
    :param knot_vector: knot vector
    :type knot_vector: list, tuple
    :param span: knot span, :math:`i`
    :type span: int
    :param knot: knot or parameter, :math:`u`
    :type knot: float
    :return: basis function, :math:`N_{i,p}`
    :rtype: float
    """
    # Special case at boundaries
    if (
        (span == 0 and knot == knot_vector[0])
        or (span == len(knot_vector) - degree - 2)
        and knot == knot_vector[len(knot_vector) - 1]
    ):
        return 1.0

    # Knot is outside of span range
    if knot < knot_vector[span] or knot >= knot_vector[span + degree + 1]:
        return 0.0

    N = [0.0 for _ in range(degree + span + 1)]

    # Initialize the zeroth degree basis functions
    for j in range(0, degree + 1):
        if knot_vector[span + j] <= knot < knot_vector[span + j + 1]:
            N[j] = 1.0

    # Computing triangular table of basis functions
    for k in range(1, degree + 1):
        # Detecting zeros saves computations
        saved = 0.0
        if N[0] != 0.0:
            saved = ((knot - knot_vector[span]) * N[0]) / (
                knot_vector[span + k] - knot_vector[span]
            )

        for j in range(0, degree - k + 1):
            Uleft = knot_vector[span + j + 1]
            Uright = knot_vector[span + j + k + 1]

            # Zero detection
            if N[j + 1] == 0.0:
                N[j] = saved
                saved = 0.0
            else:
                temp = N[j + 1] / (Uright - Uleft)
                N[j] = saved + (Uright - knot) * temp
                saved = (knot - Uleft) * temp
    return N[0]


def uniform_knot_bspline_(control_points_u, control_points_v, degree_u, degree_v, grid_size=30):
    """
    Returns uniform knots, given the number of control points in u and v directions and 
    their degrees.
    """
    u = v = np.arange(0., 1, 1 / grid_size)

    knots_u = [0.0] * degree_u + np.arange(0, 1.01, 1 / (control_points_u - degree_u)).tolist() + [1.0] * degree_u
    knots_v = [0.0] * degree_v + np.arange(0, 1.01, 1 / (control_points_v - degree_v)).tolist() + [1.0] * degree_v

    nu = []
    nu = np.zeros((u.shape[0], control_points_u))
    for i in range(u.shape[0]):
        for j in range(0, control_points_u):
            nu[i, j] = basis_function_one(degree_u, knots_u, j, u[i])

    nv = np.zeros((v.shape[0], control_points_v))
    for i in range(v.shape[0]):
        for j in range(0, control_points_v):
            nv[i, j] = basis_function_one(degree_v, knots_v, j, v[i])
    return nu, nv, knots_u, knots_v

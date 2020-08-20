import numpy as np

def compute_stats(data, max_surfaces, max_control_points):
    # see if the number of surfaces are less than the threshold
    if len(data) > max_surfaces:
        return [0, None]

    contain_spline = False
    types = []
    for surf in data:
        types.append(surf["type"])
    if "BSpline" in types:
        contain_spline = True
    else:
        return[0, None]

    # remove extra meta data
    for d in data:
        for key in ["vert_parameters", "face_indices", "coefficients", "vert_indices"]:
            if key in d.keys():
                del d[key]

    new_data = []
    for index, surf in enumerate(data):
        # removing the unnecessary information about exact poles and keeping just the counts
        new_data.append(surf)
        if surf["type"] == "BSpline":
            new_data[-1]["poles"] = np.array(surf["poles"]).shape
            new_data[-1]["u_knots"] = np.array(surf["u_knots"]).shape
            new_data[-1]["v_knots"] = np.array(surf["v_knots"]).shape
            new_data[-1]["weights"] = np.array(surf["weights"]).shape

    ctrl_p_shape = []
    for surf in data:
        if surf["type"] == "BSpline":
            ctrl_p_shape.append(np.array(surf["weights"]).reshape(1, 2))

    ctrl_p_shape = np.concatenate(ctrl_p_shape, 0)
    valid_splines = np.sum((ctrl_p_shape < max_control_points))

    valid_shapes = False
    if valid_splines == ctrl_p_shape.shape[0] * ctrl_p_shape.shape[1]:
        valid_shapes = True

    return [valid_shapes, new_data]
import math
import torch
import warnings
import numpy as np

from scipy import interpolate


PI = 2 * torch.acos(torch.zeros(1))


def _ax_angle_to_mat(axis, angle):
    n = torch.norm(axis)
    x, y, z = axis / n
    c = torch.cos(angle)
    s = torch.sin(angle)

    xs = x*s
    ys = y*s
    zs = z*s
    xC = x*(1-c)
    yC = y*(1-c)
    zC = z*(1-c)

    return torch.tensor(
        [
            [x*xC + c, x*yC - zs, x*zC + ys],
            [y*xC + zs, y*yC + c, y*zC - xs],
            [z*xC - ys, z*yC + xs, z*zC + c]
        ]
    )


def _bernstein(n, i, t):
    """
    Bernstein polynomial
    :param n: (int) polynomial degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    binomial_coeff = torch.combinations(torch.arange(n), i).shape[0] if i > 0 else torch.ones(1)
    return binomial_coeff * t.pow(i) * (1 - t).pow(n - i)


def _bezier_point(t, control_points):
    """
    Return one point on the bezier curve.
    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return torch.stack([_bernstein(n, i, t) * control_point for i, control_point in enumerate(control_points)]).sum(0)


def _bezier_path(control_points, n_points=4):
    """
    Compute bezier path (trajectory) given control points.
    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = [_bezier_point(t, control_points) for t in torch.linspace(0, 1, n_points)]
    return torch.stack(traj)


def _get_bezier_len(n_points, complexity):
    return int(2 * (complexity - 1) * (n_points - 1))


def _get_bezier_n_points(x_len, complexity):
    return math.ceil((x_len / (2 * (complexity - 1))) + 1)


def _generate_bezier_wave(x_len, mean=1.0, std=0.2, complexity=5, controls=[0.25, 0.75], **kwargs):
    n_points = _get_bezier_n_points(x_len, complexity)
    bezier_len = _get_bezier_len(n_points, complexity)
    step = round(bezier_len / complexity)
    points_x = torch.tensor(
        [
            0,
            *[
                torch.normal(idx, std, size=(1,)) for idx in torch.linspace(step, bezier_len - step, complexity - 2)
            ],
            bezier_len
        ]
    )
    points_y = torch.normal(mean, std, size=(complexity,))
    points = torch.stack([points_x, points_y]).T

    wave = []
    current_point = points[0, ...]
    for i in range(complexity - 1):
        for j, control in enumerate(controls):
            mid_point = (points[i, ...] + (points[i + 1, ...] - points[i, ...]) * control)
            if j == 0:
                points_input = torch.stack(
                    [
                        current_point,
                        points[i, ...],
                        mid_point
                    ]
                )
            else:
                points_input = torch.stack(
                    [
                        current_point,
                        mid_point
                    ]
                )

            bezier_line = _bezier_path(
                points_input,
                n_points
            )
            wave.append(bezier_line[:-1, :])
            current_point = mid_point
    
    return torch.cat(wave)[:x_len, :]


def _generate_cubic_spline_wave(x_len, mean=1.0, std=0.2, complexity=5, **kwargs):
    x = np.arange(0, x_len, (x_len - 1) / (complexity - 1))
    y = np.random.normal(loc=mean, scale=std, size=(complexity, ))
    cs = interpolate.CubicSpline(x, y)
    return torch.as_tensor(cs(np.arange(x_len)))


def _interp1d(x, xp, yp):
    is_singles = {}
    eps = torch.finfo(yp.dtype).eps

    for name, vec in {'x': x, 'xp': xp, 'yp': yp}.items():
        assert vec.ndim == 2, "_interp1d expects all input to have 2D shape"
        is_singles[name] = vec.shape[0] == 1

    # Checking for the dimensions
    assert (
        xp.size(1) == yp.size(1) and
        (
            xp.size(0) == yp.size(0) or
            xp.size(0) == 1 or
            yp.size(0) == 1
        )
    ), "x and y must have the same number of columns, and either the same number of row or one of them having only one row."

    reshaped_x = False
    if (
        (xp.size(0) == 1) and
        (yp.size(0) == 1) and
        (x.size(0) > 1)
    ):
        original_x_shape = x.shape
        x = x.contiguous().view(1, -1)
        reshaped_x = True

    max_dim = max(xp.size(0), x.size(0))
    shape_y = (max_dim, x.size(-1))
    y = torch.zeros(*shape_y)

    ind = y.long()

    if x.size(0) == 1:
        x = x.expand(xp.size(0), -1)

    torch.searchsorted(
        xp.contiguous(),
        x.contiguous(),
        out=ind
    )

    ind -= 1
    ind = torch.clamp(ind, 0, xp.size(1) - 1 - 1)

    def sel(vec, is_single):
        if is_single:
            return vec.contiguous().view(-1)[ind]
        return torch.gather(vec, 1, ind)

    is_singles['slopes'] = is_singles['xp']
    
    slopes = (
        (yp[:, 1:] - yp[:, :-1])
        /
        (eps + (xp[:, 1:] - xp[:, :-1]))
    )

    y = sel(
        yp,
        is_singles["yp"]
    ) + sel(
        slopes,
        is_singles["slopes"]
    ) * (
        x - sel(
            xp,
            is_singles["xp"]
        )
    )

    if reshaped_x:
        y = y.view(original_x_shape)

    return y


def scale(x, mean=1.0, std=0.1, axis=0):
    noise = torch.normal(mean, std, x.shape[axis])
    return x * noise


def jitter(x, mean=0.0, std=0.05):
    noise = torch.normal(mean, std, x.shape)
    return x + noise


def rotate(x):
    axis = 2 * torch.rand(3) - 1
    angle = 2 * PI * torch.rand(1) - PI
    rot = _ax_angle_to_mat(axis, angle)
    return torch.matmul(rot, x)


def time_warp(x, axis=0, backend="cubic_spline", **kwargs):
    assert x.ndim == 2, "This operation only supports 2D Tensor"
    if not axis:
        spatial_len, temporal_len = x.shape
    else:
        temporal_len, spatial_len = x.shape
    
    if backend == "cubic_spline":
        waves = torch.stack([_generate_cubic_spline_wave(temporal_len, **kwargs) for _ in range(spatial_len)], dim=1)
    elif backend == "bezier":
        warnings.warn("\"bezier\" backend is vastly slower than \"cubic_spline\" backend.")
        waves = torch.stack([_generate_bezier_wave(temporal_len, **kwargs) for _ in range(spatial_len)], dim=1)
    else:
        raise ValueError(f"The chosen backend {backend} is not supported. Supported backend: \"cubic_spline\", \"bezier\")

    cumsum_waves = torch.cumsum(waves, 0)
    cumsum_waves *= torch.true_divide(temporal_len - 1, cumsum_waves[-1, :])
    cumsum_waves = cumsum_waves.t()

    x_range = torch.arange(temporal_len)
    return _interp1d(x_range, cumsum_waves, x)


def magnitude_warp(x, axis=0, backend="cubic_spline", **kwargs):
    assert x.ndim == 2, "This operation only supports 2D Tensor"
    if not axis:
        spatial_len, temporal_len = x.shape
    else:
        temporal_len, spatial_len = x.shape
    
    if backend == "cubic_spline":
        waves = torch.stack([_generate_cubic_spline_wave(temporal_len, **kwargs) for _ in range(spatial_len)], dim=1)
    elif backend == "bezier":
        warnings.warn("\"bezier\" backend is vastly slower than \"cubic_spline\" backend.")
        waves = torch.stack([_generate_bezier_wave(temporal_len, **kwargs) for _ in range(spatial_len)], dim=1)
    else:
        raise ValueError(f"The chosen backend {backend} is not supported. Supported backend: \"cubic_spline\", \"bezier\")

    return x  * waves
import math
import torch
import numpy as np


PI = 2 * torch.acos(torch.zeros(1))


# def _interp_1d(x, xp, fp):


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
    return int(((2 * complexity) - 1) * n_points)


def _get_bezier_n_points(x_len, complexity):
    return math.ceil(x_len / ((2 * complexity) - 1))


def _generate_wave(x_len, mean=1.0, std=0.2, complexity=6, controls=[0.25, 0.57]):
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
            wave.append(bezier_line)
            current_point = mid_point
    
    bezier_line = _bezier_path(
        torch.stack(
            [
                current_point,
                points[-1, ...]
            ]
        ),
        n_points
    )
    wave.append(bezier_line)
    return torch.cat(wave)[:x_len, :]


def scale(x, mean=1.0, std=0.1, axis=0):
    noise = torch.normal(mean, std, x.shape[axis])
    return x * noise


def jitter(x, mean=0.0, std=0.05):
    noise = torch.normal(mean, std, x.shape)
    return x + noise


def rotate(x, axis=0):
    axis = 2 * torch.rand(x.shape[axis]) - 1
    angle = 2 * PI * torch.rand(1) - PI
    rot = _ax_angle_to_mat(axis, angle)
    return torch.matmul(rot, x)


def time_warp(x):


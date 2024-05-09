import os
from errno import EEXIST

import torch
from jaxtyping import Float
from torch import Tensor


def strip_lowerdiag(L: Float[Tensor, "b 3 3"]) -> Float[Tensor, "b 6"]:
    """
    Strip lower diagonal of a 3x3 matrix.
    """
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym: Float[Tensor, "b 3 3"]) -> Float[Tensor, "b 6"]:
    """ Strip symmetric matrix."""
    return strip_lowerdiag(sym)


def build_scaling_rotation(s: Float[Tensor, "b 3"], r: Float[Tensor, "b 4"]) -> Float[Tensor, "b 3 3"]:
    """
    Build scaling rotation matrix.
    s: scaling matrix of bX3
    r: rotation matrix of bX4
    r should be qw, qx, qy, qz.
    """

    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_rotation(r: Float[Tensor, "b 4"]) -> Float[Tensor, "b 3 3"]:
    """
    build rotation from quaternion to rotation matrix.
    arguments:
    r: quaternion with 4 elements.
    return: Tensor, rotation matrix of bX3X3
    """
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        os.makedirs(folder_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and os.path.isdir(folder_path):
            pass
        else:
            raise


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

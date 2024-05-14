#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import numpy as np
import torch
from jaxtyping import Float
from typing import NamedTuple


class BasicPointCloud(NamedTuple):
    points: Float[np.ndarray, "b 3"]
    colors: Float[np.ndarray, "b 3"]
    normals: Float[np.ndarray, "b 3"]


def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)


def getWorld2View2(
    R, t, translate: np.ndarray | None = None, scale: float | None = None
) -> np.ndarray:
    """
    this gives the world2camera matrix which already takes the camera pose transformation,
    such as centering and scaling.
    return world to camera matrix of 4X4
    """
    if translate is None:
        translate = np.array([0.0, 0.0, 0.0])
    if scale is None:
        scale = 1.0

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # R for w2c
    Rt[:3, 3] = t  # T for w2c
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return Rt.astype(np.float32)  # type: ignore


def getWorld2View_torch(R, t):
    Rt = torch.zeros(4, 4, device=R.device, dtype=R.dtype)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getProjectionMatrix(*, znear, zfar, fovX, fovY, **kwargs):
    """
    todo: normalized device coordinates?
    GPT4:
    Yes, the function getProjectionMatrix(znear, zfar, fovX, fovY) you've provided is indeed creating a projection
    matrix that transforms 3D coordinates into a normalized space, often referred to as Normalized Device Coordinates
    (NDC).

    In the context of 3D graphics, a projection matrix is used to transform the 3D coordinates of objects in a scene
    into the 2D coordinates of the viewport. The projection matrix created by this function is a perspective projection
    matrix, which means it also takes into account the perspective (or field of view) of the camera.

    The parameters znear and zfar define the near and far clipping planes of the view frustum, which is the region of
    space in the modeled world that may appear on the screen. fovX and fovY are the horizontal and vertical field
     of view angles.

    The resulting matrix will transform 3D points in the world such that points within the view frustum are mapped to
    a cube of size 2x2x2 centered at the origin. These are the Normalized Device Coordinates.
    The coordinates are then typically transformed again to map to the actual pixel coordinates of the viewport.

    So, while the term "Normalized Device Coordinates" is not explicitly mentioned in the code,
    the concept is indeed being used.
    """

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def getProjectionMatrixShift(
    *,
    znear,
    zfar,
    focal_x,
    focal_y,
    cx,
    cy,
    width,
    height,
    fovX,
    fovY,
    device: str | torch.device
):
    # fov is a report between the pixel and the true z focal length in meter.
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # the origin at center of image plane
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    # shift the frame window due to the non-zero principle point offsets
    # todo: remains to verify.
    offset_x = cx - (width / 2)
    offset_x = (offset_x / focal_x) * znear
    offset_y = cy - (height / 2)
    offset_y = (offset_y / focal_y) * znear

    top = top + offset_y
    left = left + offset_x
    right = right + offset_x
    bottom = bottom + offset_y

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

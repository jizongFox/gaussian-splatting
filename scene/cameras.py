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
import numpy as np
import torch
import typing as t
from jaxtyping import Float
from torch import nn, Tensor

from utils.graphics_utils import (
    getWorld2View2,
    getProjectionMatrixShift,
)


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id: int,
        R: Float[np.ndarray, "3 3"],
        T: Float[np.ndarray, "3"],
        FoVx: float,
        FoVy: float,
        focal_x: float,
        focal_y: float,
        cx: float,
        cy: float,
        image: Float[Tensor, "3 h w"],
        gt_alpha_mask: t.Optional[Float[Tensor, "h w"]],
        image_name: str,
        uid: int,
        trans: Float[np.ndarray, "3 "] = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
    ):
        """
        Camera class for storing camera information and image data.

        :param colmap_id: Camera ID in COLMAP
        :param R: Rotation matrix
        :param T: Translation vector
        :param FoVx: Field of view in x direction
        :param FoVy: Field of view in y direction
        :param cx: Principal point in x direction
        :param cy: Principal point in y direction
        :param image: Image data
        :param gt_alpha_mask: Ground truth alpha mask
        :param image_name: Image name
        :param uid: Unique ID for each image.
        :param trans: additional translation vector
        :param scale: scale factor
        :param data_device: device to store data

        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx: float = cx
        self.cy: float = cy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(
                f"[Warning] Custom device {data_device} failed, fallback to default cuda device"
            )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones(
                (1, self.image_height, self.image_width), device=self.data_device
            )

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.focal_x = focal_x
        self.focal_y = focal_y

        # to ndc matrix?
        # todo: this is to change where we take consideration of cx and cy
        self._projection_matrix = (
            getProjectionMatrixShift(
                znear=self.znear,
                zfar=self.zfar,
                fovX=self.FoVx,
                fovY=self.FoVy,
                focal_x=self.focal_x,
                focal_y=self.focal_y,
                cx=self.cx,
                cy=self.cy,
                width=self.image_width,
                height=self.image_height,
            )
            .transpose(0, 1)
            .cuda()
        )
        # TODO _projection_matrix is the P matrix with transpose (P^{T}).

        self.world_view_transform = (
            torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        )
        # todo  world_view_transform records the w2c matrix with transpose (w2c^{T}).

        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self._projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        # this means (proj@world2cam)^{T}, should be left multiplied with the xyz.

        self.cam2world = torch.inverse(
            torch.tensor(getWorld2View2(R, T, trans, scale))
        ).cuda()
        # self.world2cam = torch.tensor(getWorld2View2(R, T, trans, scale))
        self.camera_center = self.cam2world[:3, 3].cuda()

    def extra_repr(self) -> str:
        return (
            f"name={self.image_name}, image_id={self.uid}, "
            f"c2w={self.cam2world.cpu().numpy().tolist()}, "
            f"center={self.camera_center}"
        )


class MiniCam:
    def __init__(
        self,
        width,
        height,
        fovy,
        fovx,
        znear,
        zfar,
        world_view_transform,
        full_proj_transform,
    ):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

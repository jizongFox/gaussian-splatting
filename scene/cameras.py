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

from gaussian_renderer.finetune_utils import initialize_quat_delta, quat2rotation
from utils.graphics_utils import (
    getProjectionMatrixShift,
    getWorld2View_torch,
)

context: str | None = None


def set_context(ctx: str):
    global context
    context = ctx


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
        :param R: Rotation matrix in c2w
        :param T: Translation vector in w2c
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
        self.R = R.astype(np.float32)
        self.T = T.astype(np.float32)
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.cx: float = cx
        self.cy: float = cy
        self.image_name = image_name

        self.data_device = torch.device(data_device)

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
        self.znear = 0.0000001

        self.trans = trans
        self.scale = scale
        self.focal_x = focal_x
        self.focal_y = focal_y

        enable_delta: bool = False
        if context == "icomma":
            enable_delta: bool = True

        # add delta
        self.delta_quat: Float[Tensor, "1 4"] = nn.Parameter(
            initialize_quat_delta(1, device=data_device),
            requires_grad=enable_delta,
        )
        self.delta_t: Float[Tensor, "1 3"] = nn.Parameter(
            torch.zeros(1, 3, device=data_device),
            requires_grad=enable_delta,
        )
        #
        # to ndc matrix?
        # todo: this is to change where we take consideration of cx and cy

        # self.world2cam = torch.tensor(getWorld2View2(R, T, trans, scale))

    def extra_repr(self) -> str:
        return (
            f"name={self.image_name}, image_id={self.uid}, "
            f"c2w={self.cam2world.detach().cpu().numpy().tolist()}, "
            f"center={self.camera_center}"
        )

    @property
    def projection_matrix(self) -> Float[Tensor, "4 4"]:
        # TODO _projection_matrix is the P matrix with transpose (P^{T}).

        return getProjectionMatrixShift(
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
            device=self.data_device,
        ).transpose(0, 1)

    @property
    def world_view_transform(self) -> Float[Tensor, "4 4"]:
        # todo  world_view_transform records the w2c matrix with transpose (w2c^{T}).

        R_new = quat2rotation(self.delta_quat)[0] @ torch.tensor(
            self.R, device=self.data_device
        )

        t_new = (
            quat2rotation(self.delta_quat)[0] @ self.cam2world_ori[:3, 3] + self.delta_t
        )  # camtoworld_translation

        new_matrix = torch.zeros(4, 4, device=self.data_device)
        new_matrix[:3, :3] = R_new
        new_matrix[:3, 3] = t_new
        new_matrix[3, 3] = 1.0
        return new_matrix.inverse().transpose(0, 1)

    @property
    def world_view_transform_old_method(self) -> Float[Tensor, "4 4"]:
        # todo  world_view_transform records the w2c matrix with transpose (w2c^{T}).

        R_new = quat2rotation(self.delta_quat)[0] @ torch.tensor(
            self.R, device=self.data_device
        )
        T_new = torch.tensor(self.T, device=self.data_device) + self.delta_t

        # R_new is the c2w matrix with delta rotation.
        # T_new is the w2c matrix with delta translation.
        # return getWorld2View_torch(R_new, T_new).transpose(0, 1)
        return getWorld2View_torch(R_new, T_new).transpose(0, 1)

    @property
    def cam2world(self) -> Float[Tensor, "4 4"]:
        return torch.inverse(self.world_view_transform.transpose(0, 1))

    @property
    def camera_center(self):
        return self.cam2world[:3, 3].cuda()

    @property
    def cam2world_ori(self) -> Float[Tensor, "4 4"]:
        return torch.inverse(self.world_view_transform_ori.transpose(0, 1))

    @property
    def world_view_transform_ori(self) -> Float[Tensor, "4 4"]:
        R = torch.tensor(self.R, device=self.data_device)
        T = torch.tensor(self.T, device=self.data_device)
        return getWorld2View_torch(R, T).transpose(0, 1)

    @property
    def full_proj_transform(self) -> Float[Tensor, "4 4"]:
        # this means (proj@world2cam)^{T}, should be left multiplied with the xyz.

        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def image_id(self) -> int:
        return self.uid

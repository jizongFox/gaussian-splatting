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
from pathlib import Path
from torch import nn, Tensor

from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
from utils.general_utils import run_resize
from utils.graphics_utils import (
    getProjectionMatrixShift,
    getWorld2View_torch,
)


class Camera(nn.Module):
    def __init__(
        self,
        colmap_id: int,
        R: Float[np.ndarray, "3 3"] | Float[Tensor, "3 3"],
        T: Float[np.ndarray, "3"] | Float[Tensor, "3 3"],
        FoVx: float,
        FoVy: float,
        focal_x: float,
        focal_y: float,
        cx: float,
        cy: float,
        image: Float[Tensor, "3 h w"] | str | Path,
        gt_alpha_mask: t.Optional[Float[Tensor, "h w"]],
        image_name: str,
        uid: int,
        trans: Float[np.ndarray, "3 "] = np.array([0.0, 0.0, 0.0]),
        scale: float = 1.0,
        data_device: str = "cuda",
        image_width: int = None,
        image_height: int = None,
        camera_extrinsic: Float[Tensor, "6"] | None = None,
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
        :param downsample: downsample factor

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

        self.data_device = torch.device(data_device)

        self.image_width = image_width
        self.image_height = image_height

        self.image = image

        self.zfar = 100.0
        self.znear = 0.0000001

        self.trans = trans
        self.scale = scale
        self.focal_x = focal_x
        self.focal_y = focal_y

        self.delta_quat: Float[Tensor, "1 3"] = torch.zeros(
            1, 3, device=data_device, requires_grad=False
        )
        self.delta_t: Float[Tensor, "1 3"] = torch.zeros(
            1, 3, device=data_device, requires_grad=False
        )

        # self.world2cam = torch.tensor(getWorld2View2(R, T, trans, scale))

        self.camera_extrinsic = camera_extrinsic

    @property
    def _pose_delta(self) -> Float[Tensor, "1 6"]:
        return torch.cat(
            [
                self.delta_t,
                self.delta_quat,
            ],
            dim=1,
        )

    def extra_repr(self) -> str:
        return (
            f"name={self.image_name}, image_id={self.uid}, "
            f"c2w={self.cam2world.detach().cpu().numpy().tolist()}, "
            f"center={self.camera_center}, "
            # f"camera_extrinsic={self.camera_extrinsic.tolist()}"
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
        origin_c2w = self.cam2world_ori

        c2w_delta = exp_map_SO3xR3(self._pose_delta)[0]  # 3X4
        c2w_delta = torch.cat(
            [c2w_delta, torch.tensor([[0, 0, 0, 1]], device=self.data_device)], dim=0
        )

        new_c2w = c2w_delta @ origin_c2w
        w2c = torch.inverse(new_c2w)

        return w2c.transpose(0, 1)

    @property
    def cam2world(self) -> Float[Tensor, "4 4"]:
        return torch.inverse(self.world_view_transform.transpose(0, 1))

    @property
    def world2cam(self) -> Float[Tensor, "4 4"]:
        return self.world_view_transform.transpose(0, 1)

    @property
    def camera_center(self):
        return self.cam2world[:3, 3].cuda()

    @property
    def cam2world_ori(self) -> Float[Tensor, "4 4"]:
        return torch.inverse(self.world_view_transform_ori.transpose(0, 1))

    @property
    def world_view_transform_ori(self) -> Float[Tensor, "4 4"]:
        R = self.R
        T = self.T
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

    @property
    def original_image(self) -> Tensor:
        try:
            return run_resize(
                image_path=self.image, width=self.image_width, height=self.image_height
            )
        except AttributeError as e:
            raise RuntimeError(e)  # if isinstance(self.image, (str, Path)):

        # return run_resize(  #     image_path=self.image, width=self.image_width, height=self.image_height  # )  # elif isinstance(self.image, Tensor):  #     return self.image  # else:  #     raise ValueError("Image type not supported")

    @property
    def camera_name(self) -> str:
        return str(Path(self.image_name).parents[0])

    @property
    def time_slot(self) -> str:
        return str(Path(self.image_name).stem)

    @property
    def R(self) -> torch.Tensor:
        return self._R

    @R.setter
    def R(self, value):
        if isinstance(value, np.ndarray):
            self._R = torch.from_numpy(value).float().cuda()
        elif isinstance(value, Tensor):
            self._R = value
        else:
            raise NotImplementedError("R should be either np.ndarray or torch.Tensor")

    @property
    def T(self) -> torch.Tensor:
        return self._T

    @T.setter
    def T(self, value):
        if isinstance(value, np.ndarray):
            self._T = torch.from_numpy(value).float().cuda()
        elif isinstance(value, Tensor):
            self._T = value
        else:
            raise NotImplementedError("T should be either np.ndarray or torch.Tensor")


def create_pose_optimizer(
    cameras: t.List[Camera], lr: float = 1e-8, scene_scale: float = 20
) -> torch.optim.Optimizer:
    quats, translates = [], []
    for cur_camera in cameras:
        cur_camera.delta_quat.requires_grad = True
        cur_camera.delta_t.requires_grad = True
        quats.append(cur_camera.delta_quat)
        translates.append(cur_camera.delta_t)
    params = [
        {"params": quats, "lr": lr, "name": "quat"},
        {"params": translates, "lr": lr * scene_scale, "name": "translate"},
    ]

    pose_optimizer = torch.optim.Adam(params, lr=lr)
    return pose_optimizer

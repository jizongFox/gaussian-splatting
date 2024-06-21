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
from collections import defaultdict

import numpy as np
import torch
import typing as t
from jaxtyping import Float
from pathlib import Path
from torch import nn, Tensor

from gaussian_renderer.finetune_utils import quat2rotation
from nerfstudio.cameras.lie_groups import exp_map_SO3xR3
from scene.cameras import Camera
from utils.general_utils import run_resize
from utils.graphics_utils import (
    getProjectionMatrixShift,
    getWorld2View_torch,
)

context: str | None = None


def set_context(ctx: str):
    global context
    context = ctx


def delta_parameter_to_matrix(delta: Float[Tensor, "6"]) -> Float[Tensor, "4 4"]:
    delta_matrix_3_4 = exp_map_SO3xR3(delta[None, ...])[0]
    result = torch.eye(4, device="cuda", dtype=torch.float32)
    result[:3, :4] = delta_matrix_3_4[:3, :]
    return result


class CameraRig(nn.Module):
    def __init__(
        self,
        *,
        colmap_id: int,
        camera2center: Float[Tensor, "4 4"],
        center2world: Float[Tensor, "4 4"],
        camera2center_delta_quat: Float[Tensor, "6"] | None = None,
        camera2center_delta_trans: Float[Tensor, "6"] | None = None,
        center2world_delta_quat: Float[Tensor, "6"] | None = None,
        center2world_delta_trans: Float[Tensor, "6"] | None = None,
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
        super().__init__()
        self.cam2center = camera2center
        self.center2world = center2world
        self.cam2center_delta_quat = camera2center_delta_quat
        self.cam2center_delta_trans = camera2center_delta_trans
        self.center2world_delta_quat = center2world_delta_quat
        self.center2world_delta_trans = center2world_delta_trans

        self.uid = uid
        self.colmap_id = colmap_id
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

        self.camera_extrinsic = camera_extrinsic

    def extra_repr(self) -> str:
        return (
            f"name={self.image_name}, image_id={self.uid}, "
            f"c2w={self.cam2world.detach().cpu().numpy().tolist()}, "
            f"center={self.camera_center}, "
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

        R_new = self.R

        t_new = self.cam2world_ori[:3, 3]  # camtoworld_translation

        new_matrix = torch.zeros(4, 4, device=self.data_device)
        new_matrix[:3, :3] = R_new
        new_matrix[:3, 3] = t_new
        new_matrix[3, 3] = 1.0
        return new_matrix.inverse().transpose(0, 1)

    @property
    def world_view_transform_old_method(self) -> Float[Tensor, "4 4"]:
        # todo  world_view_transform records the w2c matrix with transpose (w2c^{T}).

        R_new = self.R
        T_new = self.T

        # R_new is the c2w matrix with delta rotation.
        # T_new is the w2c matrix with delta translation.
        # return getWorld2View_torch(R_new, T_new).transpose(0, 1)
        return getWorld2View_torch(R_new, T_new).transpose(0, 1)

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

    @property
    def camera_name(self) -> str:
        return str(Path(self.image_name).parents[0])

    @property
    def time_slot(self) -> str:
        return str(Path(self.image_name).stem)

    @property
    def R(self) -> torch.Tensor:
        if self.cam2center_delta is not None:
            cam2world = (
                delta_parameter_to_matrix(self.center2world_delta)
                @ self.center2world
                @ delta_parameter_to_matrix(self.cam2center_delta)
                @ self.cam2center
            )
        else:
            cam2world = self.center2world @ self.cam2center
        return cam2world[:3, :3]

    @property
    def T(self) -> torch.Tensor:
        if self.cam2center_delta is not None:
            cam2world = (
                delta_parameter_to_matrix(self.center2world_delta)
                @ self.center2world
                @ delta_parameter_to_matrix(self.cam2center_delta)
                @ self.cam2center
            )
        else:
            cam2world = self.center2world @ self.cam2center
        return torch.inverse(cam2world)[:3, 3]

    @property
    def center2world_delta(self) -> Float[Tensor, "6"]:
        return torch.cat([self.center2world_delta_trans, self.center2world_delta_quat])

    @property
    def cam2center_delta(self) -> Float[Tensor, "6"]:
        return torch.cat([self.cam2center_delta_trans, self.cam2center_delta_quat])


def to_rig_cameras(cameras: t.List[Camera]) -> t.List[CameraRig]:
    def create_cam2center(extrinsic: Float[np.ndarray, "6"]):
        cam_rot = quat2rotation(
            torch.from_numpy(extrinsic[:4].astype(np.float32))[None, ...]
        )[0].cuda()
        cam_translate = torch.from_numpy(extrinsic[4:].astype(np.float32)).cuda()
        cam2center = torch.eye(4, device="cuda", dtype=torch.float32)
        cam2center[:3, :3] = cam_rot
        cam2center[:3, 3] = cam_translate
        return cam2center

    center_to_worlds = defaultdict(list)

    for cur_camera in cameras:
        cur_camera: Camera
        cam2world = cur_camera.cam2world
        cam2center = create_cam2center(cur_camera.camera_extrinsic)
        center2world = cam2world @ torch.inverse(cam2center)
        time_slot_name = cur_camera.time_slot
        center_to_worlds[time_slot_name].append(center2world)

    center_to_worlds = {
        k: nn.Parameter(sum(v) / len(v), requires_grad=False)
        for k, v in center_to_worlds.items()
    }
    camera_to_center = {
        c.camera_name: nn.Parameter(
            create_cam2center(c.camera_extrinsic), requires_grad=False
        )
        for c in cameras
    }

    center_to_worlds_delta_quat = {
        k: nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=True)
        for k in center_to_worlds
    }
    center_to_worlds_delta_trans = {
        k: nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=True)
        for k in center_to_worlds
    }
    camera_to_center_delta_quat = {
        k: nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=True)
        for k in camera_to_center
    }
    camera_to_center_delta_trans = {
        k: nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=True)
        for k in camera_to_center
    }

    # convert camera to rig camera

    rig_cameras = []

    for cur_camera in cameras:
        cur_camera: Camera
        rig_camera = CameraRig(
            colmap_id=cur_camera.colmap_id,
            camera2center=camera_to_center[cur_camera.camera_name],
            center2world=center_to_worlds[cur_camera.time_slot],
            camera2center_delta_quat=camera_to_center_delta_quat[
                cur_camera.camera_name
            ],
            camera2center_delta_trans=camera_to_center_delta_trans[
                cur_camera.camera_name
            ],
            center2world_delta_quat=center_to_worlds_delta_quat[cur_camera.time_slot],
            center2world_delta_trans=center_to_worlds_delta_trans[cur_camera.time_slot],
            FoVx=cur_camera.FoVx,
            FoVy=cur_camera.FoVy,
            focal_x=cur_camera.focal_x,
            focal_y=cur_camera.focal_y,
            cx=cur_camera.cx,
            cy=cur_camera.cy,
            image=cur_camera.image,
            gt_alpha_mask=None,
            image_name=cur_camera.image_name,
            uid=cur_camera.uid,
            image_width=cur_camera.image_width,
            image_height=cur_camera.image_height,
            camera_extrinsic=None,
        )
        rig_cameras.append(rig_camera)
    return rig_cameras


def create_pose_optimizer(
    camera_rig: t.List[CameraRig], lr, scene_scale: float = 20
) -> torch.optim.Adam:
    cam2center_quat_parameters = []
    cam2center_trans_parameters = []
    center2world_quat_parameters = []
    center2world_trans_parameters = []

    for cur_camera in camera_rig:
        cam2center_quat_parameters.append(cur_camera.cam2center_delta_quat)
        cam2center_trans_parameters.append(cur_camera.cam2center_delta_trans)
        center2world_quat_parameters.append(cur_camera.center2world_delta_quat)
        center2world_trans_parameters.append(cur_camera.center2world_delta_trans)
    """
    exmaple 
     [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]
    """

    params = [
        {"params": cam2center_quat_parameters, "lr": lr, "name": "cam2center_quat"},
        {
            "params": cam2center_trans_parameters,
            "lr": lr * scene_scale,
            "name": "cam2center_trans",
        },
        {"params": center2world_quat_parameters, "lr": lr, "name": "center2world_quat"},
        {
            "params": center2world_trans_parameters,
            "lr": lr * scene_scale,
            "name": "center2world_trans",
        },
    ]
    pose_optimizer = torch.optim.Adam(params, lr=lr)
    return pose_optimizer


def camera_metrics(cameras: t.List[CameraRig]):
    cam2center_translation = torch.stack(
        tuple(set([x.cam2center_delta for x in cameras]))
    )[:, :3]
    cam2center_rotation = torch.stack(
        tuple(set([x.cam2center_delta for x in cameras]))
    )[:, 3:]

    center2world_translation = torch.stack(
        tuple(set([x.center2world_delta for x in cameras]))
    )[:, :3]
    center2world_rotation = torch.stack(
        tuple(set([x.center2world_delta for x in cameras]))
    )[:, 3:]

    cam2center_translation_max = cam2center_translation.norm(dim=-1).max()
    cam2center_translation_mean = cam2center_translation.norm(dim=-1).mean()

    center2world_translation_max = center2world_translation.norm(dim=-1).max()
    center2world_translation_mean = center2world_translation.norm(dim=-1).mean()

    cam2center_rotation_mean = np.rad2deg(cam2center_rotation.mean().cpu().detach())
    cam2center_rotation_max = np.rad2deg(cam2center_rotation.max().cpu().detach())

    center2world_rotation_mean = np.rad2deg(center2world_rotation.mean().cpu().detach())
    center2world_rotation_max = np.rad2deg(center2world_rotation.max().cpu().detach())

    return {
        "cam2center_translation_max": float(cam2center_translation_max),
        "cam2center_translation_mean": float(cam2center_translation_mean),
        "center2world_translation_max": float(center2world_translation_max),
        "center2world_translation_mean": float(center2world_translation_mean),
        "cam2center_rotation_mean": float(cam2center_rotation_mean),
        "cam2center_rotation_max": float(cam2center_rotation_max),
        "center2world_rotation_mean": float(center2world_rotation_mean),
        "center2world_rotation_max": float(center2world_rotation_max),
    }

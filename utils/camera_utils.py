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
from tqdm import tqdm

from scene.cameras import Camera
from scene.dataset_readers import CameraInfo
from utils.graphics_utils import fov2focal


def loadCam(downsample: int, id: int, cam_info: CameraInfo, resolution_scale) -> Camera:
    (cx, cy) = (cam_info.cx / downsample, cam_info.cy / downsample)
    focal_length_x, focal_length_y = (
        cam_info.focal_x / downsample,
        cam_info.focal_y / downsample,
    )
    image_width = int(cam_info.width / downsample)
    image_height = int(cam_info.height / downsample)

    loaded_mask = None
    camera = Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx,
        cy=cy,
        image=cam_info.image_path,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device="cuda",
        focal_x=focal_length_x,
        focal_y=focal_length_y,
        image_width=image_width,
        image_height=image_height,
        camera_extrinsic=cam_info.camera_extrinsic,
    )
    return camera


def cameraList_from_camInfos(cam_infos, resolution_scale, downscale) -> t.List[Camera]:
    # with Pool(os.cpu_count() * 2) as pool:
    #     camera_list = pool.starmap(
    #         loadCam,
    #         [(downscale, x, y, resolution_scale) for x, y in enumerate(cam_infos)],
    #     )

    camera_list = []

    for num, camera_info in tqdm(enumerate(cam_infos), total=len(cam_infos)):
        camera_list.append(loadCam(downscale, num, camera_info, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: CameraInfo):
    """
    this function takes camera and save c2w matrix into json.
    Some notations are wrong in this script.

    The generated json should be the same, compared with nerfstudio,
    except for opengl and manual rotation.

    """

    c2w = camera.c2w
    pos = c2w[:3, 3]
    rot = c2w[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]

    camera_entry = {
        "id": id,
        "img_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "position": pos.tolist(),
        "rotation": serializable_array_2d,
        "fy": fov2focal(camera.FovY, camera.height),
        "fx": fov2focal(camera.FovX, camera.width),
        "cx": camera.cx,
        "cy": camera.cy,
    }
    return camera_entry


import math


def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z  # in radians


def camera_metrics(cameras: t.List[Camera]):
    translation = torch.cat([x.delta_t for x in cameras])
    rotation = torch.cat([x.delta_quat for x in cameras])

    transalation_max = translation.norm(dim=-1).max()
    transalation_mean = translation.norm(dim=-1).mean()

    rotation_mean = np.rad2deg(rotation.mean().cpu().detach())
    rotation_max = np.rad2deg(rotation.max().cpu().detach())

    return {
        "translation_max": float(transalation_max),
        "translation_mean": float(transalation_mean),
        "rotation_mean": float(rotation_mean),
        "rotation_max": float(rotation_max),
    }

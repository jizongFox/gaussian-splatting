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
from multiprocessing.dummy import Pool

import os
import typing as t

from scene.cameras import Camera
from scene.dataset_readers import _read_image, CameraInfo
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False


def loadCam(downsample: int, id: int, cam_info: CameraInfo, resolution_scale) -> Camera:
    if cam_info.image is None:
        cam_info.image = _read_image(cam_info.image_path)

    orig_w, orig_h = cam_info.image.size

    if downsample in [1, 2, 4, 8]:
        scale = downsample
        resolution = round(orig_w / (resolution_scale * downsample)), round(
            orig_h / (resolution_scale * downsample)
        )
    else:  # should be a type that converts to float
        if downsample == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print(
                        "[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                    )
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / downsample

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    (cx, cy) = (cam_info.cx / scale, cam_info.cy / scale)
    focal_length_x, focal_length_y = cam_info.focal_x / scale, cam_info.focal_y / scale

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        cx=cx,
        cy=cy,
        image=gt_image,
        gt_alpha_mask=loaded_mask,
        image_name=cam_info.image_name,
        uid=id,
        data_device="cuda",
        focal_x=focal_length_x,
        focal_y=focal_length_y,
    )


def cameraList_from_camInfos(cam_infos, resolution_scale, downscale) -> t.List[Camera]:
    # camera_list = []
    with Pool(os.cpu_count() * 2) as pool:
        camera_list = pool.starmap(
            loadCam,
            [(downscale, x, y, resolution_scale) for x, y in enumerate(cam_infos)],
        )
    # for id, c in enumerate(cam_infos):
    #     camera_list.append(loadCam(args, id, c, resolution_scale))

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

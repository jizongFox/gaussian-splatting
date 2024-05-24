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
from __future__ import annotations

import json
import numpy as np
import os
import sys
import torch
import typing as t
from PIL import Image
from dataclasses import dataclass
from functools import lru_cache
from jaxtyping import Float
from loguru import logger
from pathlib import Path
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    read_extrinsics_text,
    read_intrinsics_text,
    qvec2rotmat,
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
    read_points3D_text,
    Colmap_Camera,
    Image as Colmap_Image,
)
from scene.gaussian_model import BasicPointCloud
from scene.helper import SE3_to_quaternion_and_translation_torch
from utils.graphics_utils import getWorld2View2, focal2fov


def _read_image(filename):
    with Image.open(filename) as f:
        return f.copy()


@dataclass
class CameraInfo:
    uid: int
    """camera id in colmap"""
    R: np.ndarray
    """R for c2w """
    T: np.ndarray
    """T for w2c """
    FovY: float
    FovX: float
    image: np.ndarray | None
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float
    cy: float
    focal_x: float
    focal_y: float

    @property
    def w2c(self) -> Float[np.ndarray, "4 4"]:
        return getWorld2View2(self.R, self.T)

    @property
    def c2w(self) -> Float[np.ndarray, "4 4"]:
        return np.linalg.inv(self.w2c)


@dataclass(slots=True)
class SceneInfo:
    point_cloud: BasicPointCloud | None
    train_cameras: t.List[CameraInfo]
    test_cameras: t.List[CameraInfo]
    nerf_normalization: dict
    ply_path: str | None | Path


def get_center_and_diag(cam_centers):
    """
    this function returns the center and the maximum norm of the camera poses.
    """
    cam_centers = np.hstack(cam_centers)
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    return center.flatten(), diagonal


def getNerfppNorm(cam_info):

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(
    cam_extrinsics: t.Dict[int, Colmap_Image],
    cam_intrinsics: t.Dict[int, Colmap_Camera],
    *,
    images_folder: str,
    force_centered_pp: bool = False,
) -> t.List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))  # R for c2w
        T = np.array(extr.tvec)  # here the T is not -RT, T for w2c

        if intr.model == "SIMPLE_PINHOLE":
            raise RuntimeError("SIMPLE_PINHOLE cameras are not supported!")
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            if force_centered_pp:
                cx = width / 2
                cy = height / 2
            else:
                cx = intr.params[2]
                cy = intr.params[3]
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            if force_centered_pp:
                cx = width / 2
                cy = height / 2
            else:
                cx = intr.params[2]
                cy = intr.params[3]
        else:
            assert False, (
                "Colmap camera model not handled: "
                "only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
            )

        image_path = os.path.join(images_folder, extr.name)
        image_name = "/".join(
            [Path(x).stem for x in Path(image_path).relative_to(images_folder).parts]
        )
        image = None

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            cx=cx,
            cy=cy,
            focal_y=focal_length_y,
            focal_x=focal_length_x,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos


def fetchPly(path: str | Path, remove_rgb_color: bool = False):
    plydata = PlyData.read(path)
    vertices = plydata["vertex"]
    positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    colors = np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
    if remove_rgb_color:
        logger.warning("Removing RGB colors from the point cloud.")
        colors = np.zeros_like(colors)
    try:
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
    except ValueError:
        normals = None

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, "vertex")
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(
    path: str | Path,
    image_dir: str | Path,
    eval_mode: bool,
    llffhold: int = 8,
    load_pcd: bool = False,
    force_centered_pp: bool = False,
):
    cam_intrinsics: t.Dict[int, Colmap_Camera]
    cam_extrinsics: t.Dict[int, Colmap_Image]

    if Path(os.path.join(path, "images.bin")).exists():
        cameras_extrinsic_file = os.path.join(path, "images.bin")
        cameras_intrinsic_file = os.path.join(path, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    elif Path(os.path.join(path, "images.txt")).exists():
        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    else:
        raise ValueError("No cameras file found.")

    cam_infos_unsorted: t.List[CameraInfo] = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=image_dir,
        force_centered_pp=force_centered_pp,
    )
    cam_infos: t.List[CameraInfo] = sorted(
        cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )
    cam_infos = [x for x in cam_infos if Path(x.image_path).exists()]

    if os.environ.get("DEBUG", "0") == "1":
        cam_infos = cam_infos[::2]

    train_cam_infos: t.List[CameraInfo]
    test_cam_infos: t.List[CameraInfo]

    if eval_mode:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    logger.info(f"nerf_normalization: {nerf_normalization}")

    ply_path = os.path.join(path, "points3D.ply")

    bin_path = os.path.join(path, "points3D.bin")
    txt_path = os.path.join(path, "points3D.txt")
    if not os.path.exists(ply_path):
        logger.info(
            "Converting point3d.bin to .ply, will happen only the first time you open the scene."
        )
        assert Path(bin_path).exists() or Path(txt_path).exists()
        if Path(bin_path).exists():
            xyz, rgb, _ = read_points3D_binary(bin_path)
        else:
            xyz, rgb, _ = read_points3D_text(txt_path)
        mask = np.linalg.norm(xyz, axis=-1) <= 50
        xyz, rgb = xyz[mask], rgb[mask]
        storePly(ply_path, xyz, rgb)

    if load_pcd:
        pcd = fetchPly(ply_path)
    else:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info


def _read_slam_intrinsic_and_extrinsic(
    json_path: Path | str,
    image_folder: Path | str,
    output_convention: t.Literal["opencv", "slam"] = "opencv",
) -> t.Tuple[t.Dict[int, Colmap_Camera], t.Dict[int, Colmap_Image]]:
    json_path = Path(json_path)
    image_folder = Path(image_folder)
    assert json_path.exists(), f"Path {json_path} does not exist."
    assert (
        image_folder.exists() and image_folder.is_dir()
    ), f"Path {image_folder} does not exist."

    with open(json_path, "r") as f:
        meta_file = json.load(f)

    camera_calibrations = meta_file["calibrationInfo"]
    available_cam_list = list(camera_calibrations.keys())
    cameras = {}

    for camera_id, (cur_camera_name, camera_detail) in enumerate(
        camera_calibrations.items()
    ):
        model = "PINHOLE"
        width = camera_detail["intrinsics"]["width"]
        height = camera_detail["intrinsics"]["height"]
        params = [
            camera_detail["intrinsics"]["camera_matrix"][0],
            camera_detail["intrinsics"]["camera_matrix"][4],
            camera_detail["intrinsics"]["camera_matrix"][2],
            camera_detail["intrinsics"]["camera_matrix"][5],
        ]
        params = np.array(params).astype(float)

        cameras[camera_id] = Colmap_Camera(
            id=camera_id, model=model, width=width, height=height, params=params
        )

    # del camera_id

    def iterate_word2cam_matrix(meta_file, image_folder):
        available_image_names = [x.name for x in image_folder.glob("*.png")]

        for cur_frame in meta_file["data"]:
            for cur_camera_name, cur_c2w in cur_frame["worldTcam"].items():
                if cur_camera_name in cur_frame["imgName"]:
                    if cur_frame["imgName"][cur_camera_name] in available_image_names:
                        yield cur_frame["imgName"][
                            cur_camera_name
                        ], cur_camera_name, cur_c2w

    def quaternion_to_rotation_matrix(q):
        w, x, y, z = q
        return np.array(
            [
                [
                    1 - 2 * y ** 2 - 2 * z ** 2,
                    2 * x * y - 2 * z * w,
                    2 * x * z + 2 * y * w,
                ],
                [
                    2 * x * y + 2 * z * w,
                    1 - 2 * x ** 2 - 2 * z ** 2,
                    2 * y * z - 2 * x * w,
                ],
                [
                    2 * x * z - 2 * y * w,
                    2 * y * z + 2 * x * w,
                    1 - 2 * x ** 2 - 2 * y ** 2,
                ],
            ]
        )

    S = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
    )

    images = {}

    extrinsic_raw = list(iterate_word2cam_matrix(meta_file, image_folder))
    logger.info(f"Found {len(extrinsic_raw)} images in the meta file.")

    for image_id, (cur_name, cur_camera_name, cur_frame) in enumerate(extrinsic_raw):
        px = cur_frame["px"]
        py = cur_frame["py"]
        pz = cur_frame["pz"]
        qw = cur_frame["qw"]
        qx = cur_frame["qx"]
        qy = cur_frame["qy"]
        qz = cur_frame["qz"]
        R = np.zeros((4, 4))
        qvec = np.array([qw, qx, qy, qz])
        norm = np.linalg.norm(qvec)
        qvec /= norm
        R[:3, :3] = quaternion_to_rotation_matrix(qvec)
        R[:3, 3] = np.array([px, py, pz])
        R[3, 3] = 1.0
        # todo: check if normalized. If not, normalize it.
        # Q, T = SE3_to_quaternion_and_translation_torch(torch.from_numpy(R).unsqueeze(0).double())
        # assert torch.allclose(Q.float(), torch.from_numpy(qvec).float(), rtol=1e-3, atol=1e-3), (
        #     Q.float(), torch.from_numpy(qvec).float())
        # assert torch.allclose(T.float(), torch.from_numpy([px, py, pz]).float(), rtol=1e-3, atol=1e-3)
        # here the world coordinate is defined in robotics space, where z is up, x is left and y is right.
        # the camera coordinate is defined in opencv convention, where the camera is looking down the z axis,
        # y is down and x is right.

        # convert the world coordinate to camera coordinate.
        if output_convention == "opencv":
            R = S.T.dot(R)  # this is the c2w in opencv convention.

        world2cam = np.linalg.inv(R)  # this is the w2c in opencv convention.

        world2cam = torch.tensor(world2cam)
        Q, T = SE3_to_quaternion_and_translation_torch(
            world2cam.unsqueeze(0)
        )  # this is the w2c
        qx, qy, qz, qw = Q.numpy().flatten().tolist()
        px, py, pz = T.numpy().flatten().tolist()

        qvec_w2c = np.array([qw, qx, qy, qz])
        assert np.linalg.norm(qvec_w2c) - 1 < 1e-3, np.linalg.norm(qvec_w2c)
        tvec_w2c = np.array([px, py, pz])

        # get the camera_id:
        camera_id = available_cam_list.index(cur_camera_name)

        images[image_id] = Colmap_Image(
            id=image_id,
            qvec=qvec_w2c,
            tvec=tvec_w2c,
            camera_id=camera_id,
            name=cur_name,
            xys=None,
            point3D_ids=None,
        )
    return cameras, images


def readSlamSceneInfo(
    json_path: str | Path,
    image_dir: str | Path,
    eval_mode: bool = True,
    llffhold=8,
    load_pcd: bool = False,
    force_centered_pp: bool = False,
) -> SceneInfo:
    assert Path(json_path).exists(), f"Path {json_path} does not exist."

    cam_intrinsics, cam_extrinsics = _read_slam_intrinsic_and_extrinsic(
        json_path=Path(json_path),
        image_folder=Path(image_dir),
        output_convention="slam",
    )
    cam_infos_unsorted: t.List[CameraInfo] = readColmapCameras(
        cam_extrinsics=cam_extrinsics,
        cam_intrinsics=cam_intrinsics,
        images_folder=image_dir,
        force_centered_pp=force_centered_pp,
    )
    cam_infos: t.List[CameraInfo] = sorted(
        cam_infos_unsorted.copy(), key=lambda x: x.image_name
    )

    if os.environ.get("DEBUG", "0") == "1":
        cam_infos = cam_infos[::2]

    train_cam_infos: t.List[CameraInfo]
    test_cam_infos: t.List[CameraInfo]

    if eval_mode:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )
    return scene_info


@lru_cache()
def _preload():
    torch.inverse(torch.randn((2, 2, 2), device="cuda"))

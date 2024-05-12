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

import json
import os
import random
import typing as t
from loguru import logger

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
        pcd_path=None,
        global_args=None,
    ):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            logger.info(
                "Loading trained model at iteration {}".format(self.loaded_iter)
            )

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(
            os.path.join(args.source_path, "cameras.bin")
        ) or os.path.exists(os.path.join(args.source_path, "cameras.txt")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                global_args.image_dir,
                args.eval,
                force_cxcy_center=global_args.force_cxcy_center,
            )
        elif global_args.meta_file is not None:
            print("Found slam file, assuming slam data set!")
            scene_info = sceneLoadTypeCallbacks["Slam"](
                global_args.meta_file,
                global_args.image_dir,
                args.eval,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )

        else:
            assert False, "Could not recognize scene type!"

        if pcd_path is not None:
            logger.warning(f"using {pcd_path}")
            scene_info.point_cloud = fetchPly(pcd_path)
            scene_info.ply_path = pcd_path

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            logger.debug("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            logger.debug(
                f"Loaded {len(self.train_cameras[resolution_scale])} images at resolution scale {resolution_scale}"
            )
            logger.debug("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )
            logger.debug(
                f"Loaded {len(self.test_cameras[resolution_scale])} images at resolution scale {resolution_scale}"
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                ),
            )
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0) -> t.List[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> t.List[Camera]:
        return self.test_cameras[scale]

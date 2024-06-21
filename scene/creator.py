import json
import os
import typing as t
from loguru import logger
from pathlib import Path

from configs.base import SlamDatasetConfig, ColmapDatasetConfig
from scene.cameras import Camera
from scene.dataset_readers import readColmapSceneInfo, readSlamSceneInfo
from scene.gaussian_model import GaussianModel
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        gaussians: GaussianModel,
        dataset: t.Union["SlamDatasetConfig", "ColmapDatasetConfig"],
        save_dir: Path,
        load_iteration: int = None,
        resolution_scales=(1,),
    ):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = save_dir.as_posix()
        self.gaussians = gaussians
        self.loaded_iter = None

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

        if isinstance(dataset, ColmapDatasetConfig):

            scene_info = readColmapSceneInfo(
                dataset.sparse_dir.as_posix(),
                dataset.image_dir.as_posix(),
                eval_mode=dataset.eval_mode,
                load_pcd=False,
                force_centered_pp=dataset.force_centered_pp,
                llffhold=dataset.eval_every_n_frame,
            )
        elif isinstance(dataset, SlamDatasetConfig):
            scene_info = readSlamSceneInfo(
                image_dir=dataset.image_dir.as_posix(),
                json_path=dataset.meta_file.as_posix(),
                eval_mode=dataset.eval_mode,
                load_pcd=False,
                force_centered_pp=dataset.force_centered_pp,
                llffhold=dataset.eval_every_n_frame,
            )
        else:
            raise ValueError("Unknown dataset type")

        if not self.loaded_iter:
            # todo: to check this part.
            with open(dataset.pcd_path, "rb") as src_file, open(
                os.path.join(save_dir, "input.ply"), "wb"
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

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            assert resolution_scale == 1
            logger.debug("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, dataset.resolution
            )
            logger.debug(
                f"Loaded {len(self.train_cameras[resolution_scale])} images at resolution scale {resolution_scale}"
            )
            logger.debug("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, dataset.resolution
            )
            logger.debug(
                f"Loaded {len(self.test_cameras[resolution_scale])} images at resolution scale {resolution_scale}"
            )

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0) -> t.List[Camera]:
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0) -> t.List[Camera]:
        return self.test_cameras[scale]

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
import rich
import torch
import typing as t
import tyro
import yaml
from loguru import logger
from pathlib import Path
from tqdm import tqdm

from configs.base import (
    ExperimentConfig,
    ModelConfig,
    ControlConfig,
    FinetuneOptimizerConfig,
    SlamDatasetConfig,
)
from gaussian_renderer import pose_depth_render
from scene.cameras import Camera
from scene.creator import Scene, GaussianModel
from scene.dataset_readers import _preload, fetchPly  # noqa
from utils.system_utils import get_hash
from utils.train_utils import (
    prepare_output_and_logger,
    _iterate_over_cameras,
)


@torch.no_grad()
def training(
    *,
    config: ExperimentConfig,
    gaussians: GaussianModel,
    camera_iters: t.Generator,
    save_dir: Path,
):

    point_cloud_path = (
        config.save_dir / f"point_cloud/iteration_00000" / "point_cloud.ply"
    )
    gaussians.save_ply(point_cloud_path)

    visibility_list = set()

    for iteration in tqdm(range(0, config.optimizer.iterations + 1)):
        try:
            cur_camera = next(camera_iters)
        except StopIteration:
            break

        viewpoint_cam: Camera = cur_camera["camera"]
        gt_image = cur_camera["target"]
        mask = cur_camera["mask"]
        gt_depth = cur_camera["depth"]

        render_pkg = pose_depth_render(
            viewpoint_cam,
            model=gaussians,
            bg_color=background,
        )

        image, depth, viewspace_point_tensor, visibility_filter, radii, accum_alphas = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["alphas"],
        )

        if accum_alphas.mean() > 0.6:
            visibility_list.add(viewpoint_cam.image_name)
            with open(save_dir / "visibility.json", "w") as f:
                json.dump(sorted(visibility_list), f)
            tb_writer.add_images(
                f"{viewpoint_cam.image_name}/render",
                image[None],
                global_step=0,
            )
            tb_writer.add_images(
                f"{viewpoint_cam.image_name}/gt",
                gt_image[None],
                global_step=0,
            )

    return visibility_list


#%% configuration
slam_data_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/orchard_tilted_rich.dslam/"
)
save_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/orchard_tilted_rich.dslam/outputs/3dgs-subset/"
)

slam_config = SlamDatasetConfig(
    image_dir=slam_data_dir,
    mask_dir=None,
    depth_dir=None,
    meta_file=slam_data_dir / "meta.json",
    pcd_path=slam_data_dir / "subregion1-downsampled.ply",
    remove_pcd_color=False,
    resolution=8,
    max_sphere_distance=0.01,
    eval_mode=False,
    pcd_start_opacity=1.0,
)


finetuneConfig = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=slam_config,
    optimizer=FinetuneOptimizerConfig(),
    control=ControlConfig(
        save_dir=save_dir,
        num_evaluations=0,
    ),
)

finetuneConfig.optimizer.iterations = 40000

config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig}))

_hash = get_hash()
config.control.save_dir = config.control.save_dir / ("git_" + _hash)

print("Optimizing " + config.control.save_dir.as_posix())

config.save_dir.mkdir(parents=True, exist_ok=True)

Path(config.save_dir, "config.yaml").write_text(yaml.dump(vars(config)))

gaussians = GaussianModel(config.model.sh_degree)
update_lr_gs_callback = lambda iteration: gaussians.update_learning_rate(  # noqa
    iteration
)
scene = Scene(
    gaussians,
    config.dataset,
    save_dir=config.save_dir,
    load_iteration=None,
    resolution_scales=(1,),
)
gaussians.create_from_pcd(
    fetchPly(
        config.dataset.pcd_path.as_posix(),
        remove_rgb_color=config.dataset.remove_pcd_color,
    ),
    spatial_lr_scale=scene.cameras_extent,
    max_sphere=config.dataset.max_sphere_distance,
    start_opacity=config.dataset.pcd_start_opacity,
)
logger.info(
    f"Loaded {len(gaussians)} points from {config.dataset.pcd_path}, opacity: {config.dataset.pcd_start_opacity}"
)


camera_iterator = _iterate_over_cameras(
    cameras=scene.getTrainCameras(),
    data_conf=config.dataset,
    shuffle=True,
    infinite=False,
)
bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
tb_writer = prepare_output_and_logger(config)
rich.print(config)
visibility = training(
    config=config,
    gaussians=gaussians,
    camera_iters=camera_iterator,
    save_dir=config.save_dir,
)

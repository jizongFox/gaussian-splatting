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

import open3d as o3d
import rich
import torch
import typing as t
import yaml
from functools import partial
from itertools import chain
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.base import (
    ExperimentConfig,
    OptimizerConfig,
)
from gaussian_renderer import pose_depth_render_unified
from scene.cameras import Camera
from scene.creator import Scene, GaussianModel
from scene.dataset_readers import fetchPly
from scene.gaussian_model import merge_gaussian_models
from utils.background_helper import BackgroundPCDCreator
from utils.exposure_utils import ExposureManager, ExposureGrid
from utils.loss_utils import (
    ssim,
    yiq_color_space_loss,
)
from utils.system_utils import get_hash
from utils.train_utils import (
    prepare_output_and_logger,
    iterate_over_cameras,
    report_status,
)


def densification(
        *,
        gaussians: GaussianModel,
        iteration: int,
        scene: Scene,
        optim_conf: OptimizerConfig,
        radii: Tensor,
        visibility_filter: Tensor,
        viewspace_point_tensor: Tensor,
        size_threshold: int = 12,
        min_opacity: float = 0.001,
):
    # Densification
    if iteration < optim_conf.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        # size_threshold = (
        #     size_threshold if iteration > optim_conf.opacity_reset_interval else None
        # )

        if (
                iteration > optim_conf.densify_from_iter
                and iteration % optim_conf.densification_interval == 0
        ):
            logger.trace(f"calling densify_and_prune at iteration {iteration}")
            gaussians.densify_and_prune(
                optim_conf.densify_grad_threshold,
                min_opacity,
                scene.cameras_extent,
                size_threshold,
            )

        if iteration % optim_conf.opacity_reset_interval == 0:
            logger.trace("calling reset_opacity")
            gaussians.reset_opacity()
    else:
        # after having densified the pcd, we should prune the invisibile 3d gaussians.
        if iteration % 500 == 0:
            opacity_mask = t.cast(Tensor, gaussians.opacity <= min_opacity)
            logger.trace(f"pruned {opacity_mask.sum()} points at iteration {iteration}")
            gaussians.prune_points(opacity_mask.squeeze(-1))


def training(
        *,
        config: ExperimentConfig,
        gaussians: GaussianModel,
        bkg_gaussian: GaussianModel | None = None,
        scene: Scene,
        tra_cameras: t.List[Camera],
        test_cameras: t.List[Camera],
        writer: SummaryWriter,
        optimizers: t.List[t.Union[torch.optim.Optimizer, None]],
        end_of_iter_cbs: t.List[t.Union[t.Callable, None]],
        loss_cbs: t.List[t.Callable, None] | None = None,
        background: Tensor,
        exposure_manager: ExposureManager | None = None
):
    camera_iterator = iterate_over_cameras(
        cameras=tra_cameras,
        data_conf=config.dataset,
        shuffle=True,
        num_threads=6,
    )

    indicator = tqdm(range(1, config.iterations + 1))

    for iteration in indicator:
        cur_camera = next(camera_iterator)
        viewpoint_cam = cur_camera["camera"]
        gt_image = cur_camera["target"]
        mask = cur_camera["mask"]

        render_pkg, *_ = pose_depth_render_unified(
            viewpoint_cam,
            model=gaussians,
            background_model=bkg_gaussian,
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
        # accum_alpha_mask = t.cast(Tensor, accum_alphas > 0.5).float()
        if exposure_manager is not None:
            image = exposure_manager(image, viewpoint_cam.image_name)
        gt_image = gt_image * mask
        image = image * mask

        Ll1 = (1.0 - config.optimizer.lambda_dssim) * yiq_color_space_loss(
            image[None, ...], gt_image[None, ...], channel_weight=(1, 1, 1)
        ) + config.optimizer.lambda_dssim * (
                      1.0
                      - ssim(
                  image,
                  gt_image)
              )

        # if iteration > opt.densify_until_iter:
        loss = Ll1

        writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)

        if iteration % 10 == 0:
            gaussians.optimizer = t.cast(torch.optim.Adam, gaussians.optimizer)

            for cur_group in gaussians.optimizer.param_groups:
                writer.add_scalar(
                    f"train/{cur_group['name']}", cur_group["lr"], iteration
                )

        if iteration % 50 == 0 and exposure_manager is not None:
            exposure_manager.record_exps(writer, iteration=iteration)

        for cur_loss_cb in loss_cbs:
            if cur_loss_cb is None:
                continue
            cur_loss = cur_loss_cb(iteration)
            loss = loss + cur_loss

        loss.backward()

        # Optimizer step
        for cur_optimizer in optimizers:
            if cur_optimizer is None:
                continue
            cur_optimizer.step()
            cur_optimizer.zero_grad(set_to_none=True)

        for cb in end_of_iter_cbs:
            if cb is None:
                continue
            cb(iteration)

        indicator.set_postfix(
            {"loss": f"{loss.item():.4f}", "pcd": f"{len(gaussians):.2e}"}
        )

        if iteration in config.control.test_iterations:
            logger.info(f"Testing at iteration {iteration}")
            tra_result, test_result = report_status(
                train_cameras=tra_cameras[::100],
                test_cameras=test_cameras,
                data_config=config.dataset,
                render_func=lambda camera: pose_depth_render_unified(
                    camera,
                    model=gaussians,
                    bg_color=background,
                    background_model=bkg_gaussian,
                )[0],
                tb_writer=writer,
                iteration=iteration,
            )
            logger.info(f"Train: {tra_result}, Test: {test_result}")
            logger.info(f"Saving at iteration {iteration}")

            point_cloud_path = (
                    config.save_dir
                    / f"point_cloud/iteration_{iteration:06d}"
                    / "point_cloud.ply"
            )
            new_gaussian = merge_gaussian_models(gaussians, bkg_gaussian)
            new_gaussian.save_ply(point_cloud_path)

        densification(
            iteration=iteration,
            optim_conf=config.optimizer,
            radii=radii,
            visibility_filter=visibility_filter,
            viewspace_point_tensor=viewspace_point_tensor,
            size_threshold=5,
            gaussians=gaussians,
            scene=scene,
            min_opacity=config.optimizer.min_opacity,
        )


def main(
        config: ExperimentConfig,
        pose_checkpoint_path: Path | None = None,
        use_bkg_model: bool = False,
):
    rich.print(config)
    _hash = get_hash()
    config.control.save_dir = config.control.save_dir / ("git_" + _hash)

    print("Optimizing " + config.save_dir.as_posix())

    config.save_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        config.save_dir / "log.log", level="TRACE", backtrace=True, diagnose=True
    )

    Path(config.save_dir, "config.yaml").write_text(yaml.dump(vars(config)))

    gaussians = GaussianModel(config.model.sh_degree)

    scene = Scene(
        gaussians,
        config.dataset,
        save_dir=config.save_dir,
        load_iteration=None,
    )
    logger.info(f"scene scale: {scene.cameras_extent}")

    if pose_checkpoint_path is not None:
        poses_checkpoint = torch.load(pose_checkpoint_path)
        for poses in chain(scene.getTrainCameras(), scene.getTestCameras()):
            try:
                poses.load_state_dict(poses_checkpoint[poses.image_name])
            except KeyError as e:
                logger.warning(e)
                continue
        logger.info(f"Loaded poses from {pose_checkpoint_path}")
        del poses_checkpoint

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

    optimizer = gaussians.training_setup(config.optimizer)

    pose_optimizer = scene.pose_optimizer(lr=config.control.pose_lr_init)
    pose_scheduler = torch.optim.lr_scheduler.StepLR(
        pose_optimizer, step_size=config.iterations // 4, gamma=0.5
    )

    bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    tb_writer = prepare_output_and_logger(config)
    if use_bkg_model:
        logger.info("Using background model")
        # background model
        bkg_pcd: o3d.geometry.PointCloud = BackgroundPCDCreator(
            gaussian_model=gaussians,
            background=background,
            cameras=scene.getTrainCameras()[::10],
            data_config=config.dataset,
            alpha_threshold=0.1,
            num_points=int(5e4),
        ).main()
        o3d.io.write_point_cloud(str(config.save_dir / "background_pcd.ply"), bkg_pcd)
        bkg_gaussians = GaussianModel(1)
        bkg_gaussians.create_from_pcd(
            bkg_pcd,
            spatial_lr_scale=scene.cameras_extent,
            max_sphere=1.0,
            start_opacity=0.1,
        )
        bkg_optimizer = bkg_gaussians.training_setup(config.bkg_optimizer)
        update_lr_bkg_gs_callback = (
            lambda iteration: bkg_gaussians.update_learning_rate(iteration)  # noqa
        )
    else:
        bkg_gaussians = None
        bkg_optimizer = None
        update_lr_bkg_gs_callback = None
    # ============================ exposure issue =========================
    exposure_manager = ExposureGrid(cameras=scene.getTrainCameras())
    exposure_manager.cuda()
    exposure_optimizer = exposure_manager.setup_optimizer(lr=5e-2, wd=1e-4)
    exposure_scheduler = torch.optim.lr_scheduler.StepLR(
        exposure_optimizer, step_size=config.iterations // 3, gamma=0.5
    )

    # ============================ callbacks ===============================
    update_lr_gs_callback = lambda iteration: gaussians.update_learning_rate(  # noqa
        iteration
    )

    update_lr_pose_callback = lambda iteration: pose_scheduler.step()  # noqa

    update_lr_exp_callback = lambda iteration: exposure_scheduler.step()  # noqa

    def upgrade_sh_degree_callback(iteration):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            if bkg_gaussians is not None:
                bkg_gaussians.oneupSHdegree()

    # ============================ callbacks ===============================
    training(
        config=config,
        gaussians=gaussians,
        bkg_gaussian=bkg_gaussians,
        scene=scene,
        background=background,
        tra_cameras=scene.getTrainCameras(),
        test_cameras=scene.getTestCameras(),
        writer=tb_writer,
        optimizers=[pose_optimizer, optimizer, bkg_optimizer, exposure_optimizer],
        exposure_manager=exposure_manager,
        end_of_iter_cbs=[
            update_lr_gs_callback,
            update_lr_bkg_gs_callback,
            update_lr_pose_callback,
            upgrade_sh_degree_callback,
            update_lr_exp_callback
        ],
        loss_cbs=[partial(exposure_manager.loss_callback, writer=tb_writer)],
    )

    camera_checkpoint = {x.image_id: x.state_dict() for x in scene.getTrainCameras()}
    torch.save(camera_checkpoint, config.save_dir / "camera_checkpoint.pth")
    if exposure_manager is not None:
        torch.save(exposure_manager.state_dict(), config.save_dir / "exposure_manager.pth")

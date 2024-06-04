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
import open3d as o3d
import rich
import torch
import typing as t
import tyro
import yaml
from itertools import chain
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.base import (
    ExperimentConfig,
    OptimizerConfig,
    ModelConfig,
    ControlConfig,
    ColmapDatasetConfig,
)
from gaussian_renderer import pose_depth_render_unified
from scene.cameras import Camera
from scene.creator import Scene, GaussianModel
from scene.dataset_readers import _preload, fetchPly  # noqa
from scene.gaussian_model import merge_gaussian_models
from utils.background_helper import BackgroundPCDCreator
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
    optimizers: t.List[torch.optim.Optimizer],
    end_of_iter_cbs: t.List[t.Callable],
    background: Tensor,
):
    camera_iterator = iterate_over_cameras(
        cameras=tra_cameras,
        data_conf=config.dataset,
        shuffle=True,
        num_threads=6,
    )

    indicator = tqdm(range(1, config.optimizer.iterations + 1))

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

        gt_image = gt_image * mask
        image = image * mask

        Ll1 = (1.0 - config.optimizer.lambda_dssim) * yiq_color_space_loss(
            image[None, ...], gt_image[None, ...], channel_weight=(1, 1, 1)
        ) + config.optimizer.lambda_dssim * (
            1.0
            - ssim(
                image,
                gt_image,
            )
        )

        # if iteration > opt.densify_until_iter:
        loss = Ll1

        writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)

        loss.backward()

        # Optimizer step
        for cur_optimizer in optimizers:
            cur_optimizer.step()
            cur_optimizer.zero_grad(set_to_none=True)

        for cb in end_of_iter_cbs:
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
            min_opacity=0.001,
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

    config.control.save_dir.mkdir(parents=True, exist_ok=True)

    Path(config.control.save_dir, "config.yaml").write_text(
        yaml.dump(vars(config))
    )

    gaussians = GaussianModel(config.model.sh_degree)

    scene = Scene(
        gaussians,
        config.dataset,
        save_dir=config.save_dir,
        load_iteration=None,
    )

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

    pose_optimizer = scene.pose_optimizer(lr=config.optimizer.pose_lr_init)
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
        o3d.io.write_point_cloud(
            str(config.save_dir / "background_pcd.ply"), bkg_pcd
        )
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

    # ============================ callbacks ===============================
    update_lr_gs_callback = lambda iteration: gaussians.update_learning_rate(  # noqa
        iteration
    )

    update_lr_pose_callback = lambda iteration: pose_scheduler.step()  # noqa

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
        optimizers=[
            x for x in [pose_optimizer, optimizer, bkg_optimizer] if x is not None
        ],
        end_of_iter_cbs=[x for x in [
            update_lr_gs_callback,
            update_lr_bkg_gs_callback,
            update_lr_pose_callback,
            upgrade_sh_degree_callback,
        ] if x is not None],
    )

    camera_checkpoint = {x.image_id: x.state_dict() for x in scene.getTrainCameras()}
    torch.save(camera_checkpoint, config.save_dir / "camera_checkpoint.pth")


if __name__ == "__main__":
    save_dir = Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/outputs/3dgs-align-normals/try2"
    )
    # slam_dir = Path(
    #     "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion1"
    # )
    #
    # slam_config = SlamDatasetConfig(
    #     image_dir=slam_dir / "images",
    #     mask_dir=slam_dir / "masks",
    #     depth_dir=None,
    #     resolution=2,
    #     pcd_path=slam_dir
    #     / "outputs/test_depth_loss/pretrained-poses/test-depth-loss-depth-scale-shift-depth/git_2ab75d2/input.ply",
    #     pcd_start_opacity=0.99,
    #     remove_pcd_color=False,
    #     max_sphere_distance=1e-3,
    #     force_centered_pp=False,
    #     eval_every_n_frame=60,
    #     eval_mode=True,
    #     meta_file=slam_dir / "meta_updated.json",
    # )
    # pose_checkpoint_path = Path(
    #     "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion1/outputs/test_depth_loss/"
    #     "pretrained-poses/test-depth-loss-depth-sparse-nerf-2/git_f47faa1/camera_checkpoint.pth"
    # )
    #
    # colmap_dir = Path(
    #     "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/"
    #     "colmap-simpler-alignment/sparse/0"
    # )
    colmap_config = ColmapDatasetConfig(
        image_dir=Path(
            "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/images-nerfacto-keep-pp"
        ),
        mask_dir=Path(
            "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/masks"
        ),
        depth_dir=Path(
            "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/depths"
        ),
        resolution=2,
        pcd_path=Path(
            "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside_normal-opencv.ply"
        ),
        pcd_start_opacity=1.0,
        max_sphere_distance=1e-2,
        sparse_dir=Path(
            "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/colmap-simpler-alignment/sparse/0"
        ),
        force_centered_pp=False,
        eval_every_n_frame=45,
    )

    optimizer_config = OptimizerConfig(
        iterations=16_000,
        position_lr_init=0.00000016,
        position_lr_final=0.000000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.005,
        opacity_lr=0.02,
        scaling_lr=0.002,
        rotation_lr=0.002,
        percent_dense=0.01,
        lambda_dssim=0.1,
        densification_interval=200,
        opacity_reset_interval=3000000,
        densify_from_iter=5000,
        densify_until_iter=10_000,
        densify_grad_threshold=0.0005,
        pose_lr_init=0e-5,
    )
    bkg_optimizer_config = OptimizerConfig(
        iterations=16_000,
        position_lr_init=0.00016,
        position_lr_final=0.000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.005,
        opacity_lr=0.02,
        scaling_lr=0.002,
        rotation_lr=0.002,
        percent_dense=0.01,
        lambda_dssim=0.1,
        densification_interval=200,
        opacity_reset_interval=300000000,
        densify_from_iter=50000000000,
        densify_until_iter=10_000000,
        densify_grad_threshold=1,
        pose_lr_init=0e-5,
    )

    finetuneConfig = ExperimentConfig(
        model=ModelConfig(sh_degree=1, white_background=True),
        dataset=colmap_config,
        optimizer=optimizer_config,
        bkg_optimizer=bkg_optimizer_config,
        control=ControlConfig(
            save_dir=save_dir, num_evaluations=16, include_0_epoch=True
        ),
    )
    config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig}))
    main(config, pose_checkpoint_path=None)

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
import rich
import torch
import typing as t
import tyro
import yaml
from argparse import Namespace
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.base import (
    ColmapDatasetConfig,
    ExperimentConfig,
    OptimizerConfig,
    ModelConfig,
    ControlConfig,
)
from gaussian_renderer import pose_depth_render
from scene.cameras import Camera
from scene.creator import Scene, GaussianModel
from scene.dataset_readers import _preload  # noqa
from utils.loss_utils import (
    l1_loss,
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
):
    # Densification
    if iteration < optim_conf.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(
            gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
        )
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
        size_threshold = (
            size_threshold if iteration > optim_conf.opacity_reset_interval else None
        )

        if (
            iteration > optim_conf.densify_from_iter
            and iteration % optim_conf.densification_interval == 0
        ):
            logger.trace(f"calling densify_and_prune at iteration {iteration}")
            gaussians.densify_and_prune(
                optim_conf.densify_grad_threshold,
                0.0001,
                scene.cameras_extent,
                size_threshold,
            )

        if iteration % optim_conf.opacity_reset_interval == 0:
            logger.trace("calling reset_opacity")
            gaussians.reset_opacity()
    # else:
    #     # after having densified the pcd, we should prune the invisibile 3d gaussians.
    #     if iteration % 500 == 0 and optim_conf.prune_after_densification:
    #         opacity_mask = gaussians.opacity <= 0.005
    #         logger.trace(f"pruned {opacity_mask.sum()} points at iteration {iteration}")
    #         gaussians.prune_points(opacity_mask.squeeze(-1))


def training(
    *,
    config: ExperimentConfig,
    gaussians: GaussianModel,
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

    for iteration in tqdm(range(0, config.optimizer.iterations + 1)):

        cur_camera = next(camera_iterator)
        viewpoint_cam = cur_camera["camera"]
        gt_image = cur_camera["target"]
        mask = cur_camera["mask"]

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
        accum_alpha_mask = t.cast(Tensor, accum_alphas > 0.5).float()

        gt_image = gt_image * mask * accum_alpha_mask
        image = image * mask * accum_alpha_mask

        Ll1 = l1_loss(image, gt_image)

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

        if iteration in config.control.test_iterations:
            logger.info(f"Testing at iteration {iteration}")
            report_status(
                train_cameras=tra_cameras[::100],
                test_cameras=test_cameras,
                data_config=config.dataset,
                render_func=lambda camera: pose_depth_render(
                    camera, model=gaussians, bg_color=background
                ),
                tb_writer=writer,
                iteration=iteration,
            )
            logger.info(f"Saving at iteration {iteration}")

            point_cloud_path = (
                config.save_dir
                / f"point_cloud/iteration_{iteration:06d}"
                / "point_cloud.ply"
            )
            gaussians.save_ply(point_cloud_path)

        densification(
            iteration=iteration,
            optim_conf=config.optimizer,
            radii=radii,
            visibility_filter=visibility_filter,
            viewspace_point_tensor=viewspace_point_tensor,
            size_threshold=12,
            gaussians=gaussians,
            scene=scene,
        )


def main(config: ExperimentConfig):
    _hash = get_hash()
    config.control.save_dir = config.control.save_dir / ("git_" + _hash)

    print("Optimizing " + config.save_dir.as_posix())

    config.control.save_dir.mkdir(parents=True, exist_ok=True)

    Path(config.control.save_dir, "config.yaml").write_text(yaml.dump(vars(config)))

    gaussians = GaussianModel(config.model.sh_degree)

    scene = Scene(
        gaussians,
        config.dataset,
        save_dir=config.save_dir,
        load_iteration=None,
        shuffle=True,
    )
    optimizer = gaussians.training_setup(config.optimizer)

    pose_optimizer = scene.pose_optimizer(lr=config.optimizer.pose_lr_init)
    pose_scheduler = torch.optim.lr_scheduler.StepLR(
        pose_optimizer, step_size=5000, gamma=0.75
    )

    bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    tb_writer = prepare_output_and_logger(
        config.save_dir.as_posix(), Namespace(**vars(config))
    )
    rich.print(config)

    # ============================ callbacks ===============================
    update_lr_gs_callback = lambda iteration: gaussians.update_learning_rate(  # noqa
        iteration
    )

    update_lr_pose_callback = lambda iteration: pose_scheduler.step()  # noqa

    def upgrade_sh_degree_callback(iteration):
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 2000 == 0:
            gaussians.oneupSHdegree()

    # ============================ callbacks ===============================

    training(
        config=config,
        gaussians=gaussians,
        scene=scene,
        background=background,
        tra_cameras=scene.getTrainCameras(),
        test_cameras=scene.getTestCameras(),
        writer=tb_writer,
        optimizers=[pose_optimizer, optimizer],
        end_of_iter_cbs=[
            update_lr_gs_callback,
            update_lr_pose_callback,
            upgrade_sh_degree_callback,
        ],
    )

    camera_checkpoint = {x.image_id: x.state_dict() for x in scene.getTrainCameras()}
    torch.save(camera_checkpoint, config.save_dir / "camera_checkpoint.pth")


if __name__ == "__main__":
    save_dir = Path(
        "/home/jizong/Workspace/gaussian-splatting/output/verify_train_template"
    )

    colmap_dir = Path(
        "/home/jizong/Workspace/dConstruct/nerfstudio/data/1037-1039-single-cam-OPENCV-undistort"
    )
    colmap_config = ColmapDatasetConfig(
        image_dir=colmap_dir / "images",
        mask_dir=None,
        depth_dir=None,
        resolution=2,
        pcd_path=colmap_dir / "sparse" / "0" / "points3D.ply",
        sparse_dir=colmap_dir / "sparse/0",
        force_centered_pp=False,
        eval_every_n_frame=100,
    )

    optimizer_config = OptimizerConfig(
        iterations=15_000,
        position_lr_init=0.00016,
        position_lr_final=0.000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.025,
        opacity_lr=0.05,
        scaling_lr=0.005,
        rotation_lr=0.005,
        percent_dense=0.01,
        lambda_dssim=0.2,
        densification_interval=500,
        opacity_reset_interval=10000,
        densify_from_iter=4000,
        densify_until_iter=12_000,
        densify_grad_threshold=0.00001,
    )

    finetuneConfig = ExperimentConfig(
        model=ModelConfig(sh_degree=3, white_background=True),
        dataset=colmap_config,
        optimizer=optimizer_config,
        control=ControlConfig(save_dir=save_dir, test_iterations=5),
    )
    config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig}))
    main(config)

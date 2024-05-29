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
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.base import (
    ExperimentConfig,
    ModelConfig,
    ControlConfig,
    FinetuneOptimizerConfig,
    SlamDatasetConfig,
)
from gaussian_renderer import pose_depth_render
from scene.creator import Scene, GaussianModel
from scene.dataset_readers import _preload, fetchPly  # noqa
from utils.debug_utils import save_images
from utils.depth_related import SobelDepthLoss
from utils.system_utils import get_hash
from utils.train_utils import prepare_output_and_logger, iterate_over_cameras


def training(
    *,
    config: ExperimentConfig,
    gaussians: GaussianModel,
    camera_iters: t.Generator,
    tb_writer: SummaryWriter,
    optimizers: t.List[torch.optim.Optimizer],
    end_of_iter_cbs: t.List[t.Callable],
):
    # depth_criterion = SparseNerfDepthLoss(patch_size=48, stride=48)
    depth_criterion = SobelDepthLoss().cuda()
    # depth_criterion = ScaleAndShiftInvariantLoss()

    for iteration in tqdm(range(0, config.optimizer.iterations + 1)):

        cur_camera = next(camera_iters)

        viewpoint_cam = cur_camera["camera"]
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
        accum_alpha_mask = t.cast(Tensor, accum_alphas > 0.5).float()

        if not torch.any((accum_alpha_mask * mask).bool()):
            continue
        depth_loss = depth_criterion(
            depth, gt_depth, mask=(accum_alpha_mask * mask).bool()
        )

        # # Loss
        # gt_image = gt_image * mask * accum_alpha_mask
        # image = image * mask * accum_alpha_mask
        #
        # Ll1 = l1_loss(
        #     image,
        #     gt_image,
        # )
        #
        # loss = Ll1 + l1_loss(
        #     sober_filter(image[None, ...]), sober_filter(gt_image[None, ...])
        # )
        #
        # tb_writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        # tb_writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)
        loss = depth_loss

        loss.backward()

        save_images(
            camera=viewpoint_cam,
            image=image,
            gt_image=gt_image,
            depth_image=depth,
            gt_depth_image=gt_depth,
            accum_alpha_mask=accum_alpha_mask * mask,
            iteration=iteration,
            save_dir=config.save_dir,
            tb_writer=tb_writer,
        )

        # Optimizer step
        for cur_optimizer in optimizers:
            cur_optimizer.step()
            cur_optimizer.zero_grad(set_to_none=True)

        for cb in end_of_iter_cbs:
            cb(iteration)


#%% configuration
slam_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion1"
)
save_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion1/"
    "outputs/test_depth_loss/pretrained-poses/test-depth-loss-depth-sorbel-depth-loss-2"
)


slam_config = SlamDatasetConfig(
    image_dir=slam_dir / "images",
    mask_dir=slam_dir / "masks",
    depth_dir=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/depths"
    ),
    resolution=2,
    pcd_path=slam_dir
    / "outputs/test_depth_loss/pretrained-poses/test-depth-loss-depth-scale-shift-depth/git_2ab75d2/input.ply",
    pcd_start_opacity=1.0,
    remove_pcd_color=False,
    max_sphere_distance=1e-3,
    force_centered_pp=False,
    eval_mode=False,
    meta_file=slam_dir / "meta_updated.json",
)

optimizer_config = FinetuneOptimizerConfig(
    iterations=12_000,
    position_lr_init=0.0000,
    position_lr_final=0.00000,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.00000,
    opacity_lr=0.00000,
    scaling_lr=0.00000,
    rotation_lr=0.000000,
    percent_dense=0.01,  # this is to reduce the size of the eclipse
    lambda_dssim=0.2,
    densification_interval=5000000,
    opacity_reset_interval=500000000,
    densify_from_iter=4000,
    densify_until_iter=12_000,
    densify_grad_threshold=0.00001,  # this is to split more
    pose_lr_init=2e-4,
)

finetuneConfig = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=slam_config,
    optimizer=optimizer_config,
    control=ControlConfig(save_dir=save_dir, num_evaluations=10),
)
config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig}))

_hash = get_hash()
config.control.save_dir = config.control.save_dir / ("git_" + _hash)

print("Optimizing " + config.control.save_dir.as_posix())

config.control.save_dir.mkdir(parents=True, exist_ok=True)

Path(config.control.save_dir, "config.yaml").write_text(yaml.dump(vars(config)))

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


optimizer = gaussians.training_setup(config.optimizer)

pose_optimizer = scene.pose_optimizer(lr=config.optimizer.pose_lr_init)
pose_scheduler = torch.optim.lr_scheduler.StepLR(
    pose_optimizer, step_size=config.iterations // 4, gamma=0.75
)
update_lr_pose_callback = lambda iteration: pose_scheduler.step()  # noqa

camera_iterator = iterate_over_cameras(
    scene=scene,
    data_conf=config.dataset,
    shuffle=True,
)
bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
tb_writer = prepare_output_and_logger(
    config
)
rich.print(config)
training(
    config=config,
    gaussians=gaussians,
    camera_iters=camera_iterator,
    tb_writer=tb_writer,
    optimizers=[pose_optimizer, optimizer],
    end_of_iter_cbs=[update_lr_gs_callback, update_lr_pose_callback],
)

camera_checkpoint = {x.image_name: x.state_dict() for x in scene.getTrainCameras()}
torch.save(camera_checkpoint, config.save_dir / "camera_checkpoint.pth")

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
import rich
import torch
import typing as t
import tyro
import yaml
from PIL import Image, ImageDraw
from argparse import Namespace
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
    ColmapDatasetConfig,
    _DatasetConfig,
)
from dctoolbox.utils import run_in_thread
from gaussian_renderer import pose_depth_render
from gaussian_renderer.finetune_utils import (
    GradLayer,
)
from scene.creator import Scene, GaussianModel, Camera
from scene.dataset_readers import _preload  # noqa
from utils.loss_utils import l1_loss
from utils.system_utils import get_hash
from utils.train_utils import prepare_output_and_logger

_preload()

sober_filter = GradLayer().cuda()


def iterate_over_cameras(
    scene: Scene, data_conf: _DatasetConfig
) -> t.Generator[t.Dict[str, t.Union[str, torch.Tensor, Camera]], None, None]:
    while True:
        for camera in scene.getTrainCameras():
            image_name = camera.image_name
            gt_image = camera.original_image.cuda()
            mask_torch = torch.ones_like(gt_image)

            if data_conf.mask_dir is not None:
                mask_path = Path(config.dataset.mask_dir) / f"{image_name}.png.png"
                with Image.open(mask_path) as fmask:
                    fmask = fmask.convert("L").resize(
                        (gt_image.shape[2], gt_image.shape[1]), Image.NEAREST
                    )

                mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
                mask_torch = torch.from_numpy(mask).cuda()[None, ...]

            yield dict(
                image_name=image_name,
                target=gt_image,
                mask=mask_torch,
                camera=camera,
            )


@run_in_thread()
def save_images(
    camera: Camera,
    image: Tensor,
    gt_image: Tensor,
    Ll1: Tensor,
    iteration: int,
    save_dir: Path,
    tb_writer: SummaryWriter,
):
    if int(camera.image_name.split("_")[-1]) in list(range(1435, 1613)):
        tb_writer.add_images(
            f"train/{camera.image_name}/render", image[None], iteration
        )
        tb_writer.add_images(f"train/{camera.image_name}/gt", gt_image[None], iteration)
        tb_writer.add_images(
            f"train/{camera.image_name}/diff",
            torch.abs(image - gt_image)[None],
            iteration,
        )
        save_path = (
            Path(save_dir)
            / f"train/{camera.image_name}/render_{iteration:07d}_pred.png"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            (image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        ).save(save_path)

        save_path = Path(save_dir) / f"train/{camera.image_name}/gt_{iteration:07d}.png"

        Image.fromarray(
            (gt_image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        ).save(save_path)

        save_path = (
            Path(save_dir) / f"train/{camera.image_name}/diff_{iteration:07d}.png"
        )

        diff_pil = Image.fromarray(
            (torch.abs(image - gt_image).detach().cpu().numpy() * 255)
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )
        diff_pil_w_text = ImageDraw.Draw(diff_pil)

        # Add Text to an image
        diff_pil_w_text.text(
            (50, 75), f"l1 loss: {Ll1.item():.5f}", fill=(255, 255, 255)
        )
        diff_pil.save(save_path)


def training(
    *,
    config: ExperimentConfig,
    gaussians: GaussianModel,
    camera_iters: t.Generator,
    tb_writer: SummaryWriter,
    optimizers: t.List[torch.optim.Optimizer],
    end_of_iter_cbs: t.List[t.Callable],
):
    first_iter = 0

    if config.dataset.pcd_path is not None:
        logger.warning(f"using pcd_path = {config.dataset.pcd_path}")

    first_iter += 1

    for iteration in tqdm(range(first_iter, config.optimizer.iterations + 1)):

        cur_camera = next(camera_iters)

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

        # Loss
        gt_image = gt_image * mask
        image = image * mask

        gt_image = gt_image * accum_alpha_mask
        image = image * accum_alpha_mask

        Ll1 = l1_loss(
            image,
            gt_image,
        )

        loss = Ll1 + l1_loss(
            sober_filter(image[None, ...]), sober_filter(gt_image[None, ...])
        )

        tb_writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        tb_writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)

        loss.backward()

        save_images(
            viewpoint_cam,
            image,
            gt_image,
            Ll1,
            iteration,
            config.control.save_dir,
            tb_writer,
        )

        if torch.isnan(loss):
            logger.error(f"loss is NaN at iteration {iteration}")
            raise RuntimeError("loss is NaN")

        # Optimizer step
        for cur_optimizer in optimizers:
            cur_optimizer.step()
            cur_optimizer.zero_grad(set_to_none=True)

        for cb in end_of_iter_cbs:
            cb(iteration)


#%% configuration
slam_data_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion1/"
)
save_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/"
    "subregion1/outputs/test-new-interface-with-pcd-moving"
)

slam_config = SlamDatasetConfig(
    image_dir=slam_data_dir / "image-subset",
    mask_dir=slam_data_dir / "mask",
    meta_file=slam_data_dir.parent / "meta_update.json",
    pcd_path=slam_data_dir.parent / "korea_accoms_outside.ply",
    remove_pcd_color=False,
)
colmap_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/"
)
colmap_config = ColmapDatasetConfig(
    image_dir=colmap_dir / "images",
    mask_dir=colmap_dir / "masks",
    resolution=1,
    pcd_path=colmap_dir
    / "outputs/unnamed/nerfacto/2024-05-06_080057/point_cloud_world_true.ply",
    sparse_dir=colmap_dir / "colmap-simpler-alignment/sparse/0",
    force_centered_pp=False,
)

optimizer_config = FinetuneOptimizerConfig(
    iterations=15_000,
    position_lr_init=0.0001,
    position_lr_final=0.00001,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.00001,
    opacity_lr=0.00001,
    scaling_lr=0.00001,
    rotation_lr=0.000001,
    percent_dense=0.01,  # this is to reduce the size of the eclipse
    lambda_dssim=0.2,
    densification_interval=5000000,
    opacity_reset_interval=500000000,
    densify_from_iter=4000,
    densify_until_iter=12_000,
    densify_grad_threshold=0.00001,  # this is to split more
)


finetuneConfig = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=colmap_config,
    optimizer=optimizer_config,
    control=ControlConfig(save_dir=save_dir),
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
    shuffle=True,
    resolution_scales=(1,),
)
optimizer = gaussians.training_setup(config.optimizer)

pose_optimizer = scene.pose_optimizer(lr=0e-3)
pose_scheduler = torch.optim.lr_scheduler.StepLR(
    pose_optimizer, step_size=1000, gamma=0.5
)
update_lr_pose_callback = lambda iteration: pose_scheduler.step()  # noqa

camera_iterator = iterate_over_cameras(scene, config.dataset)

bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
tb_writer = prepare_output_and_logger(
    config.save_dir.as_posix(), Namespace(**vars(config))
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

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
import yaml
from PIL import Image, ImageDraw
from argparse import Namespace
from loguru import logger
from pathlib import Path
from random import randint
from torch import Tensor
from tqdm import tqdm

from configs.base import (
    ExperimentConfig,
    ModelConfig,
    SlamDatasetConfig,
    FinetuneOptimizerConfig,
    ControlConfig,
)
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

TENSORBOARD_FOUND = True
sober_filter = GradLayer().cuda()


def training(
    config: ExperimentConfig,
    checkpoint: str | None = None,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(
        config.save_dir.as_posix(), Namespace(**vars(config))
    )
    gaussians = GaussianModel(config.model.sh_degree)
    if config.dataset.pcd_path is not None:
        logger.warning(f"using pcd_path = {config.dataset.pcd_path}")
    scene = Scene(
        gaussians,
        config.dataset,
        save_dir=config.save_dir,
        load_iteration=None,
        shuffle=True,
        resolution_scales=(1,),
    )
    gaussians.training_setup(config.optimizer)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, config.optimizer)

    bg_color = [1, 1, 1] if config.model.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # initialize poses.
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack: t.List[Camera]
    viewpoint_stack = scene.getTrainCameras().copy()

    first_iter += 1

    for iteration in tqdm(range(first_iter, config.optimizer.iterations + 1)):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and not gaussians.is_max_sh():
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        viewpoint_cam: Camera
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

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

        image_name = viewpoint_cam.image_name
        if config.dataset.mask_dir is not None:
            mask_path = Path(config.dataset.mask_dir) / f"{image_name}.png.png"
            with Image.open(mask_path) as fmask:
                fmask = fmask.convert("L").resize(
                    (image.shape[2], image.shape[1]), Image.NEAREST
                )

            mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
            mask_torch = torch.from_numpy(mask).cuda()[None, ...]

        else:
            mask_torch = torch.ones_like(image)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image * mask_torch
        image = image * mask_torch

        gt_image = gt_image * accum_alpha_mask
        image = image * accum_alpha_mask

        Ll1 = l1_loss(
            image,
            gt_image,
        )

        if int(viewpoint_cam.image_name.split("_")[-1]) in list(range(1435, 1613)):
            tb_writer.add_images(
                f"train/{viewpoint_cam.image_name}/render", image[None], iteration
            )
            tb_writer.add_images(
                f"train/{viewpoint_cam.image_name}/gt", gt_image[None], iteration
            )
            tb_writer.add_images(
                f"train/{viewpoint_cam.image_name}/diff",
                torch.abs(image - gt_image)[None],
                iteration,
            )
            save_path = (
                Path(scene.model_path)
                / f"train/{viewpoint_cam.image_name}/render_{iteration:07d}_pred.png"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                (image.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            ).save(save_path)

            save_path = (
                Path(scene.model_path)
                / f"train/{viewpoint_cam.image_name}/gt_{iteration:07d}.png"
            )

            Image.fromarray(
                (gt_image.detach().cpu().numpy() * 255)
                .astype(np.uint8)
                .transpose(1, 2, 0)
            ).save(save_path)

            save_path = (
                Path(scene.model_path)
                / f"train/{viewpoint_cam.image_name}/diff_{iteration:07d}.png"
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

        loss = Ll1 + l1_loss(
            sober_filter(image[None, ...]), sober_filter(gt_image[None, ...])
        )

        tb_writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        tb_writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)

        loss.backward()

        iter_end.record()
        if torch.isnan(loss):
            logger.error(f"loss is NaN at iteration {iteration}")
            breakpoint()

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    import tyro

    finetuneConfig = ExperimentConfig(
        model=ModelConfig(sh_degree=1, white_background=True),
        dataset=SlamDatasetConfig(
            image_dir=Path(
                "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion3/images"
            ),
            mask_dir=Path(
                "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion3/masks"
            ),
            meta_file=Path(
                "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/meta_updated.json"
            ),
            pcd_path=Path(
                "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside.ply"
            ),
            remove_pcd_color=False,
        ),
        optimizer=FinetuneOptimizerConfig(),
        control=ControlConfig(
            save_dir=Path(
                "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/"
                "subregion3/outputs/test-new-interface"
            )
        ),
    )
    exp_config = tyro.cli(
        tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig})
    )

    _hash = get_hash()
    exp_config.control.save_dir = exp_config.control.save_dir / ("git_" + _hash)

    print("Optimizing " + exp_config.control.save_dir.as_posix())

    exp_config.control.save_dir.mkdir(parents=True, exist_ok=True)

    Path(exp_config.control.save_dir, "config.yaml").write_text(
        yaml.dump(vars(exp_config))
    )
    rich.print(exp_config)
    training(config=exp_config)

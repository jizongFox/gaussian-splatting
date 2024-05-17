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
import os
import random
import sys
import torch
import typing as t
import yaml
from PIL import Image, ImageDraw
from argparse import ArgumentParser
from itertools import chain
from loguru import logger
from pathlib import Path
from random import randint
from torch import nn, Tensor
from tqdm import tqdm

from arguments import (
    ModelParams,
    PipelineParams,
    OptimizationFinetuneParams,
)
from gaussian_renderer import icomma_render
from gaussian_renderer.finetune_utils import (
    apply_affine,
    GradLayer,
)
from scene import Scene, GaussianModel, Camera
from scene.cameras import set_context
from scene.dataset_readers import _preload  # noqa
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, Entropy
from utils.system_utils import get_hash
from utils.train_utils import prepare_output_and_logger

_preload()

TENSORBOARD_FOUND = True
sober_filter = GradLayer().cuda()


def training(
    dataset,
    opt,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    *,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if args.pcd_path is not None:
        logger.warning(f"using pcd_path = {args.pcd_path}")
    set_context("icomma")
    scene = Scene(dataset, gaussians, pcd_path=args.pcd_path, global_args=args)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # initialize poses.
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack: t.List[Camera]
    viewpoint_stack = scene.getTrainCameras().copy()

    pose_optimizer = torch.optim.Adam(
        chain(*[x.parameters() for x in viewpoint_stack]), lr=5e-5
    )
    pose_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        pose_optimizer, T_max=opt.iterations, eta_min=1e-6
    )
    # initialize poses.
    if args.activate_pose_grad:
        logger.info("activating pose grad")

    # cx and cy
    cxcy = nn.Embedding(4, 2).cuda()
    cxcy.weight.data = nn.Parameter(
        torch.zeros_like(cxcy.weight.data), requires_grad=args.activate_pose_grad
    )
    cxcy_optimizer = torch.optim.Adam(cxcy.parameters(), lr=2e-5)
    cxcy_scheduler = torch.optim.lr_scheduler.StepLR(
        cxcy_optimizer, step_size=opt.iterations // 4, gamma=0.1
    )

    fxfy = nn.Embedding(4, 2).cuda()
    fxfy.weight.data = nn.Parameter(
        torch.zeros_like(cxcy.weight.data), requires_grad=args.activate_pose_grad
    )
    fxfy_optimizer = torch.optim.Adam(fxfy.parameters(), lr=2e-5)
    fxfy_scheduler = torch.optim.lr_scheduler.StepLR(
        fxfy_optimizer, step_size=opt.iterations // 4, gamma=0.1
    )

    first_iter += 1

    ent_criterion = Entropy()
    for iteration in tqdm(range(first_iter, opt.iterations + 1)):
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

        cur_camera_id = torch.tensor(
            viewpoint_cam.colmap_id, dtype=torch.long, device=torch.device("cuda")
        )

        render_pkg = icomma_render(
            viewpoint_cam,
            model=gaussians,
            bg_color=background,
        )

        image, viewspace_point_tensor, visibility_filter, radii, accum_alphas = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["accum_alphas"],
        )
        cur_cxcy = cxcy(cur_camera_id)
        cur_fxfy_delta = fxfy(cur_camera_id)
        image = apply_affine(
            image[None, ...],
            cur_cxcy[0][None, ...],
            cur_cxcy[1][None, ...],
            padding_value=background,
            fx_delta=cur_fxfy_delta[0][None, ...],
            fy_delta=cur_fxfy_delta[1][None, ...],
        )[0]
        accum_alphas = apply_affine(
            accum_alphas[None, None, ...],
            cur_cxcy[0][None, ...],
            cur_cxcy[1][None, ...],
            padding_value=0,
            fx_delta=cur_fxfy_delta[0][None, ...],
            fy_delta=cur_fxfy_delta[1][None, ...],
        )[0]

        accum_alpha_mask = t.cast(Tensor, accum_alphas > 0.5).float()

        image_name = viewpoint_cam.image_name
        if args.mask_dir is not None:
            mask_path = Path(args.mask_dir) / f"{image_name}.png.png"
            with Image.open(mask_path) as fmask:
                fmask = fmask.convert("L").resize(
                    (image.shape[2], image.shape[1]), Image.NEAREST
                )

            mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
            mask_torch = torch.from_numpy(mask).cuda()[None, ...]
            mask_torch2 = apply_affine(
                torch.ones_like(mask_torch)[None, ...],
                cur_cxcy[0][None, ...],
                cur_cxcy[1][None, ...],
                padding_value=0,
                mode="nearest",
            )[0]
            mask_torch = mask_torch * mask_torch2
            #
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

        # loss += l1_loss(
        #     sober_filter(image[None, ...]), sober_filter(gt_image[None, ...])
        # )

        with torch.set_grad_enabled(True):
            opacity = gaussians.opacity[visibility_filter]
            if len(opacity) == 0:
                ent_loss = torch.tensor(0.0, device=visibility_filter.device)
            else:
                opacity_dist = torch.cat([opacity, 1 - opacity], dim=1)
                assert opacity_dist.shape[1] == 2, opacity_dist.shape

                ent_loss = ent_criterion(opacity_dist)

        # if iteration > opt.densify_until_iter:
        loss = loss + ent_loss * args.ent_weight

        tb_writer.add_scalar("train/entropy", ent_loss.item(), iteration)
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

        cxcy_optimizer.step()
        cxcy_optimizer.zero_grad(set_to_none=True)
        fxfy_optimizer.step()
        fxfy_optimizer.zero_grad(set_to_none=True)
        cxcy_scheduler.step()

        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        pose_scheduler.step()

        fxfy_scheduler.step()

        if iteration in checkpoint_iterations:
            camera_poses_state = {
                x.image_id: x.state_dict() for x in scene.getTrainCameras()
            }
            camera_poses_state.update(
                {"fxfy": fxfy.state_dict(), "cxcy": cxcy.state_dict()}
            )
            torch.save(
                camera_poses_state,
                Path(scene.model_path) / f"camera_poses_{iteration:08d}.pth",
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationFinetuneParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=random.randint(3000, 65535))
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations",
        nargs="+",
        type=int,
        default=[1, 2000, 5000, 7_000, 10000, 12_500, 15_000, 30_000, 100_000, 150_000]
        + [i for i in range(2000, 150_000, 2000)],
    )
    parser.add_argument(
        "--save_iterations",
        nargs="+",
        type=int,
        default=[1, 2000, 5000, 7_000, 10000, 12_500, 15_000, 30_000, 100_000, 150_000]
        + [i for i in range(2000, 150_000, 2000)],
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    jizong_parser = parser.add_argument_group("jizong_test")
    jizong_parser.add_argument(
        "--loss-config",
        choices=[
            "naive",
            "ssim",
            "l1",
            "l2",
            "tv",
            "ssim_21",
            "ssim_5",
            "ssim+hsv",
            "hsv",
            "yiq",
            "ssim+yiq",
            "ssim+yiq+",
            "ssim-mres+yiq+",
        ],
        type=str,
        help="jizong's loss configuration",
    )
    jizong_parser.add_argument(
        "--ent-weight", type=float, default=1e-4, help="entropy on opacity"
    )
    jizong_parser.add_argument(
        "--pcd-path", type=str, default=None, help="load pcd file"
    )
    jizong_parser.add_argument(
        "--prune-after-densification",
        default=False,
        action="store_true",
        help="prune after densification",
    )
    jizong_parser.add_argument("--no-hash", default=False, action="store_true")
    jizong_parser.add_argument("--meta-file", type=Path, help="meta file for the scene")
    jizong_parser.add_argument("--image-dir", type=Path, help="images directory")
    jizong_parser.add_argument(
        "--mask-dir", type=Path, help="mask directory, where 0 is ignored, 1 is visible"
    )
    jizong_parser.add_argument(
        "--activate-pose-grad",
        default=False,
        action="store_true",
        help="activate pose grad",
    )

    args = parser.parse_args(sys.argv[1:])
    _hash = get_hash()
    if not args.no_hash:
        args.model_path = os.path.join(args.model_path, "git_" + _hash)

    args.save_iterations.append(args.iterations)
    args.checkpoint_iterations.extend(args.save_iterations)

    print("Optimizing " + args.model_path)

    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    Path(args.model_path, "config.yaml").write_text(yaml.dump(vars(args)))
    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.loss_config is not None:
        logger.warning(f"args.loss_config={args.loss_config}")

    training(
        lp.extract(args),
        op.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args=args,
    )

    # All done
    print("\nTraining complete.")

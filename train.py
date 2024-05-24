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
import yaml
from PIL import Image
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from random import randint
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import ori_render as render
from scene.dataset_readers import _preload  # noqa
from scene.scene_old import Scene, GaussianModel
from utils.debug_utils import personalized_loss
from utils.general_utils import safe_state
from utils.loss_utils import (
    l1_loss,
)
from utils.system_utils import get_hash
from utils.train_utils import training_report, prepare_output_and_logger

_preload()

TENSORBOARD_FOUND = True


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
    *,
    args,
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args.model_path, dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    if args.pcd_path is not None:
        logger.warning(f"using pcd_path = {args.pcd_path}")
    scene = Scene(dataset, gaussians, pcd_path=args.pcd_path, global_args=args)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]

    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True
    )
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(
            viewpoint_cam,
            gaussians,
            bg_color=background,
            scaling_modifier=1.0,
        )
        image, viewspace_point_tensor, visibility_filter, radii, accum_alphas = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["accum_alphas"],
        )

        image_name = viewpoint_cam.image_name

        # dump_image(image_name, image, args, viewpoint_cam)

        if args.mask_dir is not None:
            mask_path = Path(args.mask_dir) / f"{image_name}.png.png"
            with Image.open(mask_path) as fmask:
                fmask = fmask.convert("L").resize(
                    (image.shape[2], image.shape[1]), Image.NEAREST
                )

            mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
            mask_torch = torch.from_numpy(mask).cuda()[None, ...]
            #
        else:
            mask_torch = torch.ones_like(image)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image * mask_torch
        image = image * mask_torch

        Ll1 = l1_loss(
            image,
            gt_image,
        )

        loss = personalized_loss(args, opt, Ll1, image, gt_image, mask_torch)

        tb_writer.add_scalar("train/pcd_size", len(gaussians), iteration)
        tb_writer.add_scalar("train/sh_degree", gaussians.active_sh_degree, iteration)

        loss.backward()

        iter_end.record()
        if torch.isnan(loss):
            logger.error(f"loss is NaN at iteration {iteration}")
            breakpoint()

        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix(
                {
                    "pls": f"{gaussians.xyz.shape[0]:.1e}",
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                }
            )
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Log and save
        training_report(
            tb_writer,
            iteration,
            Ll1,
            loss,
            l1_loss,
            iter_start.elapsed_time(iter_end),
            testing_iterations,
            scene,
            render,
            {"bg_color": background},
            global_args=args,
        )
        if iteration in saving_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        # Densification
        if iteration < opt.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            size_threshold = 8 if iteration > opt.opacity_reset_interval else None

            if (
                iteration > opt.densify_from_iter
                and iteration % opt.densification_interval == 0
            ):
                logger.trace(f"calling densify_and_prune at iteration {iteration}")
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    0.0001,
                    scene.cameras_extent,
                    size_threshold,
                )

            if iteration % opt.opacity_reset_interval == 0:
                logger.trace("calling reset_opacity")
                gaussians.reset_opacity()
        else:
            # after having densified the pcd, we should prune the invisibile 3d gaussians.
            if iteration % 500 == 0 and args.prune_after_densification:
                opacity_mask = gaussians.opacity <= 0.005
                logger.trace(
                    f"pruned {opacity_mask.sum()} points at iteration {iteration}"
                )
                gaussians.prune_points(opacity_mask.squeeze(-1))

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        if iteration in checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save(
                (gaussians.capture(), iteration),
                scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
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
        "--force-cxcy-center", action="store_true", help="centered cxcy"
    )

    args = parser.parse_args(sys.argv[1:])
    _hash = get_hash()
    if not args.no_hash:
        args.model_path = os.path.join(args.model_path, "git_" + _hash)

    args.save_iterations.append(args.iterations)

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
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        args=args,
    )

    # All done
    print("\nTraining complete.")

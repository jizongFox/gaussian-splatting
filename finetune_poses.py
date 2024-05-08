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
import os
import random
import sys
from argparse import ArgumentParser
from functools import lru_cache
from itertools import chain
from pathlib import Path
from random import randint

import numpy as np
import torch
import yaml
from PIL import Image
from loguru import logger
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from gaussian_renderer.finetune_utils import build_rotation, initialize_quat_delta, multiply_quaternions, \
    VGGPerceptualLoss
from scene import Scene, GaussianModel
from scene.dataset_readers import _preload  # noqa
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim, l2_loss, tv_loss, Entropy, hsv_color_space_loss, yiq_color_space_loss
from utils.system_utils import get_hash
from utils.train_utils import training_report, prepare_output_and_logger

_preload()

TENSORBOARD_FOUND = True


@lru_cache()
def report_shape(image_shape):
    logger.info(f"image shape: {image_shape}")


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
        args
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
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

    # initialize poses.

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = scene.getTrainCameras().copy()
    # initialize poses.
    if args.activate_pose_grad:
        logger.info("activating pose grad")
    pose_delta = nn.ParameterList(
        nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=args.activate_pose_grad) for _ in
        range(len(viewpoint_stack)))

    percept_loss = VGGPerceptualLoss().cuda()

    quat_delta = nn.ParameterList([nn.Parameter(x, requires_grad=args.activate_pose_grad) for x in
                                   initialize_quat_delta(len(viewpoint_stack), device="cuda")])

    optimizer_poses = torch.optim.Adam(chain(pose_delta, quat_delta), lr=1e-3)
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_poses, T_max=opt.iterations // 2,

                                                                     eta_min=1e-5)

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", dynamic_ncols=True)
    first_iter += 1

    ent_criterion = Entropy()
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                (
                    custom_cam,
                    do_training,
                    pipe.convert_SHs_python,
                    pipe.compute_cov3D_python,
                    keep_alive,
                    scaling_modifer,
                ) = network_gui.receive()
                if custom_cam != None:
                    net_image = render(
                        custom_cam, gaussians, pipe, background, scaling_modifer
                    )["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and (
                        (iteration < int(opt.iterations)) or not keep_alive
                ):
                    break
            except Exception as e:
                network_gui.conn = None

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
        cur_id = torch.tensor(viewpoint_cam.uid, dtype=torch.long, device=torch.device("cuda"))
        opt_cam_rot = quat_delta[cur_id]
        opt_cam_trans = pose_delta[cur_id]

        rel_transform = torch.eye(4).cuda().float()
        rel_transform[:3, :3] = build_rotation(F.normalize(opt_cam_rot[None]))[0]
        rel_transform[:3, 3] = opt_cam_trans

        pts = gaussians.xyz
        pts_ones = torch.ones(pts.shape[0], 1).cuda().float()
        pts4 = torch.cat((pts, pts_ones), dim=1)
        transformed_pts = (rel_transform @ pts4.T).T[:, :3]

        quat = F.normalize(opt_cam_rot[None])
        _rotations = multiply_quaternions(gaussians.rotation, quat.unsqueeze(0)).squeeze(0)

        render_pkg = render(viewpoint_cam, gaussians, pipe, background, override_mean3d=transformed_pts,
                            override_quat=_rotations)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        report_shape(image.shape)
        image_name = viewpoint_cam.image_name
        if args.mask_dir is not None:
            mask_path = Path(args.mask_dir) / f"{image_name}.png.png"
            with Image.open(mask_path) as fmask:
                fmask = fmask.convert("L").resize((image.shape[2], image.shape[1]), Image.NEAREST)

            mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
            mask_torch = torch.from_numpy(mask).cuda()[None, ...]
            #
        else:
            mask_torch = torch.ones_like(image)
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_image = gt_image * mask_torch
        image = image * mask_torch

        Ll1 = l1_loss(image, gt_image, )

        if cur_id in list(range(1000)):
            tb_writer.add_images(f"train/{cur_id}/render", image[None], iteration)
            tb_writer.add_images(f"train/{cur_id}/gt", gt_image[None], iteration)
            tb_writer.add_images(f"train/{cur_id}/diff", torch.abs(image - gt_image)[None], iteration)

        # jizong test
        if args.loss_config is not None:
            if args.loss_config == "naive":
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                        1.0 - ssim(image, gt_image, )
                )
            elif args.loss_config == "l1":
                loss = Ll1
            elif args.loss_config == "l2":
                loss = l2_loss(image, gt_image, masks=mask_torch)
            elif args.loss_config == "ssim":
                loss = 1.0 - ssim(image, gt_image, )
            elif args.loss_config == "ssim_21":
                loss = 1.0 - ssim(image, gt_image, window_size=21, )
            elif args.loss_config == "ssim_5":
                loss = 1.0 - ssim(image, gt_image, window_size=5, )
            elif args.loss_config == "tv":
                loss = tv_loss(image[None, ...], gt_image[None, ...])
            elif args.loss_config == "ssim+hsv":
                loss = 0.8 * (1.0 - ssim(image, gt_image, )) + 0.2 * \
                       hsv_color_space_loss(image[None, ...], gt_image[None, ...], channel_weight=(0.5, 1, 0.1))
            elif args.loss_config == "hsv":
                loss = hsv_color_space_loss(image[None, ...], gt_image[None, ...], channel_weight=(0.5, 1, 0.1))

            elif args.loss_config == "yiq":
                loss = yiq_color_space_loss(image[None, ...], gt_image[None, ...], channel_weight=(0.1, 1, 1))

            elif args.loss_config == "ssim+yiq":
                loss = 0.8 * (1.0 - ssim(image, gt_image, )) + 0.2 * yiq_color_space_loss(
                    image[None, ...],
                    gt_image[None, ...],
                    channel_weight=(0.1, 1, 1))

            elif args.loss_config == "ssim+yiq+":
                loss = 0.5 * (1.0 - ssim(image, gt_image, )) + 0.5 * yiq_color_space_loss(
                    image[None, ...],
                    gt_image[None, ...],
                    channel_weight=(
                        0.05, 1, 1)
                )
            elif args.loss_config == "ssim-mres+yiq+":
                loss = 0.2 * (1.0 - ssim(image, gt_image, window_size=11)) + \
                       0.2 * (1.0 - ssim(image, gt_image, window_size=5)) + \
                       0.2 * (1.0 - ssim(image, gt_image, window_size=21)) + \
                       0.4 * yiq_color_space_loss(image[None, ...], gt_image[None, ...], channel_weight=(0.01, 1, 0.35))
            else:
                raise NotImplementedError(args.loss_config)
        else:
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                    1.0 - ssim(image, gt_image, )
            )

        loss = loss

        # else: entropy minimization
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

        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"pls": f"{gaussians.xyz.shape[0]:.1e}", "Loss": f"{ema_loss_for_log:.{7}f}"})
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
            (pipe, background),
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
            gaussians.add_densification_stats(
                viewspace_point_tensor, visibility_filter
            )
            size_threshold = (
                12 if iteration > opt.opacity_reset_interval else None
            )

            if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
            ):
                logger.trace(f"calling densify_and_prune at iteration {iteration}")
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    0.005,
                    scene.cameras_extent,
                    size_threshold,
                )

            if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
            ):
                logger.trace("calling reset_opacity")
                gaussians.reset_opacity()
        else:
            # after having densified the pcd, we should prune the invisibile 3d gaussians.
            if iteration % 500 == 0 and args.prune_after_densification:
                opacity_mask = gaussians.opacity <= 0.005
                logger.trace(f"pruned {opacity_mask.sum()} points at iteration {iteration}")
                gaussians.prune_points(opacity_mask.squeeze(-1))

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        optimizer_poses.step()
        optimizer_poses.zero_grad(set_to_none=True)
        optimizer_scheduler.step()

        if iteration in checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(iteration))
            torch.save(
                (gaussians.capture(), iteration),
                scene.model_path + "/chkpnt" + str(iteration) + ".pth",
            )
            pose_checkpoint = {"pose_delta": pose_delta, "quat_delta": quat_delta}
            torch.save(pose_checkpoint, scene.model_path + "/pose_ckpt" + str(iteration) + ".pth")


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
        "--test_iterations", nargs="+", type=int,
        default=[1, 2000, 5000, 7_000, 10000, 12_500, 15_000, 30_000, 100_000, 150_000] + [i for i in
                                                                                           range(2000, 150_000, 2000)]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int,
        default=[1, 2000, 5000, 7_000, 10000, 12_500, 15_000, 30_000, 100_000, 150_000] + [i for i in
                                                                                           range(2000, 150_000, 2000)]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    jizong_parser = parser.add_argument_group("jizong_test")
    jizong_parser.add_argument("--loss-config",
                               choices=["naive", "ssim", "l1", "l2", "tv", "ssim_21", "ssim_5", "ssim+hsv", "hsv",
                                        "yiq", "ssim+yiq", "ssim+yiq+", "ssim-mres+yiq+"],
                               type=str,
                               help="jizong's loss configuration")
    jizong_parser.add_argument("--ent-weight", type=float, default=1e-4, help="entropy on opacity")
    jizong_parser.add_argument("--pcd-path", type=str, default=None, help="load pcd file")
    jizong_parser.add_argument("--prune-after-densification", default=False, action="store_true",
                               help="prune after densification")
    jizong_parser.add_argument("--no-hash", default=False, action="store_true")
    jizong_parser.add_argument("--meta-file", type=Path, help="meta file for the scene")
    jizong_parser.add_argument("--image-dir", type=Path, help="images directory")
    jizong_parser.add_argument("--mask-dir", type=Path, help="mask directory, where 0 is ignored, 1 is visible")
    jizong_parser.add_argument("--activate-pose-grad", default=False, action="store_true",
                               help="activate pose grad")

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

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)

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
        args=args
    )

    # All done
    print("\nTraining complete.")

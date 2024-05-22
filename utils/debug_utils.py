import numpy as np
import torch
from PIL import Image
from argparse import Namespace
from loguru import logger
from pathlib import Path
from torch import Tensor

from scene.cameras import Camera
from utils.loss_utils import (
    ssim,
    l2_loss,
    tv_loss,
    hsv_color_space_loss,
    yiq_color_space_loss,
)

rendered_train_set = set()


def dump_image(image_name: str, image: Tensor, args: Namespace, viewpoint_cam: Camera):
    if image_name not in rendered_train_set:
        logger.trace(f"dumping image {image_name}")
        save_dir = Path(args.model_path) / "train_images" / "epoch_0"
        save_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            (image.cpu().detach().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        ).save(save_dir / f"{image_name}_pred.png")
        Image.fromarray(
            (
                viewpoint_cam.original_image.cpu().detach().numpy().transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
        ).save(save_dir / f"{image_name}_gt.png")
        Image.fromarray(
            (
                torch.abs(image - viewpoint_cam.original_image)
                .cpu()
                .detach()
                .numpy()
                .transpose(1, 2, 0)
                * 255
            ).astype(np.uint8)
        ).save(save_dir / f"{image_name}_diff.png")
        rendered_train_set.add(image_name)


def personalized_loss(
    args: Namespace,
    opt: Namespace,
    Ll1: Tensor,
    image: Tensor,
    gt_image: Tensor,
    mask_torch: Tensor,
):
    if args.loss_config is not None:
        if args.loss_config == "naive":
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0
                - ssim(
                    image,
                    gt_image,
                )
            )
        elif args.loss_config == "l1":
            loss = Ll1
        elif args.loss_config == "l2":
            loss = l2_loss(image, gt_image, masks=mask_torch)
        elif args.loss_config == "ssim":
            loss = 1.0 - ssim(
                image,
                gt_image,
            )
        elif args.loss_config == "ssim_21":
            loss = 1.0 - ssim(
                image,
                gt_image,
                window_size=21,
            )
        elif args.loss_config == "ssim_5":
            loss = 1.0 - ssim(
                image,
                gt_image,
                window_size=5,
            )
        elif args.loss_config == "tv":
            loss = tv_loss(image[None, ...], gt_image[None, ...])
        elif args.loss_config == "ssim+hsv":
            loss = 0.8 * (1.0 - ssim(image, gt_image,)) + 0.2 * hsv_color_space_loss(
                image[None, ...], gt_image[None, ...], channel_weight=(0.5, 1, 0.1)
            )
        elif args.loss_config == "hsv":
            loss = hsv_color_space_loss(
                image[None, ...], gt_image[None, ...], channel_weight=(0.5, 1, 0.1)
            )

        elif args.loss_config == "yiq":
            loss = yiq_color_space_loss(
                image[None, ...], gt_image[None, ...], channel_weight=(0.1, 1, 1)
            )

        elif args.loss_config == "ssim+yiq":
            loss = 0.8 * (1.0 - ssim(image, gt_image,)) + 0.2 * yiq_color_space_loss(
                image[None, ...], gt_image[None, ...], channel_weight=(0.1, 1, 1)
            )

        elif args.loss_config == "ssim+yiq+":
            loss = 0.5 * (1.0 - ssim(image, gt_image,)) + 0.5 * yiq_color_space_loss(
                image[None, ...], gt_image[None, ...], channel_weight=(0.05, 1, 1)
            )
        elif args.loss_config == "ssim-mres+yiq+":
            loss = (
                0.2 * (1.0 - ssim(image, gt_image, window_size=11))
                + 0.2 * (1.0 - ssim(image, gt_image, window_size=5))
                + 0.2 * (1.0 - ssim(image, gt_image, window_size=21))
                + 0.4
                * yiq_color_space_loss(
                    image[None, ...],
                    gt_image[None, ...],
                    channel_weight=(0.01, 1, 0.35),
                )
            )
        else:
            raise NotImplementedError(args.loss_config)
    else:
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
            1.0
            - ssim(
                image,
                gt_image,
            )
        )
    return loss

import imageio
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from argparse import Namespace
from loguru import logger
from pathlib import Path
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from dctoolbox.utils import run_in_thread
from nerfstudio.utils.math import normalized_depth_scale_and_shift
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


@run_in_thread()
def save_images(
    *,
    camera: Camera,
    image: Tensor,
    gt_image: Tensor,
    depth_image: Tensor = None,
    gt_depth_image: Tensor = None,
    accum_alpha_mask: Tensor = None,
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
        l1_loss = torch.abs(image - gt_image).mean()
        diff_pil_w_text.text(
            (50, 75), f"l1 loss: {l1_loss.item():.5f}", fill=(255, 255, 255)
        )
        diff_pil.save(save_path)

        if depth_image is not None:
            depth_image_np = depth_image.cpu().detach().numpy()
            save_path = (
                Path(save_dir) / f"train/{camera.image_name}/depth_{iteration:07d}.png"
            )
            normalize_depth = np.clip(
                np.clip(
                    255.0
                    / depth_image_np.max()
                    * (depth_image_np - depth_image_np.min()),
                    0,
                    255,
                ).astype(np.uint8),
                0,
                255,
            ).astype(np.uint8)
            colormap = matplotlib.colormaps.get_cmap("plasma")
            colored_pred = colormap(normalize_depth)[..., :3][0]

            imageio.imwrite(save_path, (colored_pred * 255).astype(np.uint8))

        if gt_depth_image is not None:
            gt_depth_image_np = gt_depth_image.cpu().detach().numpy()
            save_path = (
                Path(save_dir)
                / f"train/{camera.image_name}/gt_depth_{iteration:07d}.png"
            )
            normalize_depth = np.clip(
                np.clip(
                    255.0
                    / gt_depth_image_np.max()
                    * (gt_depth_image_np - gt_depth_image_np.min()),
                    0,
                    255,
                ).astype(np.uint8),
                0,
                255,
            ).astype(np.uint8)
            colormap = matplotlib.colormaps.get_cmap("plasma")
            colored_pred = colormap(normalize_depth)[..., :3][0]

            imageio.imwrite(save_path, (colored_pred * 255).astype(np.uint8))

        if (
            gt_depth_image is not None
            and depth_image is not None
            and accum_alpha_mask is not None
        ):
            save_path = (
                Path(save_dir)
                / f"train/{camera.image_name}/diff_depth_{iteration:07d}.png"
            )
            scale, shift = normalized_depth_scale_and_shift(
                depth_image, gt_depth_image, accum_alpha_mask
            )
            shifted_depth = scale.view(-1, 1, 1) * depth_image + shift.view(-1, 1, 1)

            _depth_image = (
                shifted_depth
                - gt_depth_image.min()
                / (gt_depth_image.max() - gt_depth_image.min())
                * accum_alpha_mask
            )
            _gt_depth_image = (
                gt_depth_image
                - gt_depth_image.min()
                / (gt_depth_image.max() - gt_depth_image.min())
                * accum_alpha_mask
            )
            diff_pil = Image.fromarray(
                (torch.abs(_depth_image - _gt_depth_image) * 255)
                .cpu()
                .detach()
                .numpy()
                .astype(np.uint8)
                .squeeze(0)
            )
            diff_pil_w_text = ImageDraw.Draw(diff_pil)

            # Add Text to an image
            diff_pil_w_text.text(
                (50, 75),
                f"l1 loss: {torch.abs(depth_image - gt_depth_image).mean().item():.5f}",
                fill=(255,),
            )
            diff_pil.save(save_path)

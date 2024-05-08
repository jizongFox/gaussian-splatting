import os
import uuid
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from scene import Scene
from .image_utils import psnr


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, global_args=None):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({"name": "test", "cameras": scene.getTestCameras()}, {"name": "train",
                                                                                    "cameras": [scene.getTrainCameras()[
                                                                                                    idx % len(
                                                                                                        scene.getTrainCameras())]
                                                                                                for idx in
                                                                                                range(5, 30, 5)], },)

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0, )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if global_args.mask_dir is not None:
                        image_name = viewpoint.image_name

                        mask_path = Path(global_args.mask_dir) / f"{image_name}.png.png"
                        with Image.open(mask_path) as fmask:
                            fmask = fmask.convert("L").resize((image.shape[2], image.shape[1]), Image.NEAREST)

                            mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
                        mask_torch = torch.from_numpy(mask).cuda()[None, ...]
                    else:
                        mask_torch = torch.ones_like(image)

                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config["name"] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration, )
                        tb_writer.add_images(config["name"] + "_view_{}/error".format(viewpoint.image_name),
                                             torch.abs(image[None] - viewpoint.original_image[None]),
                                             global_step=iteration, )

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config["name"] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration, )
                    l1_test += l1_loss(image * mask_torch, gt_image * mask_torch).mean().double()
                    psnr_test += psnr(image * mask_torch, gt_image * mask_torch).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config["name"], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(args.model_path)
    return tb_writer

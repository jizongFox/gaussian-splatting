import os
import random
import sys
import torch
from argparse import ArgumentParser
from loguru import logger
from random import randint
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import network_gui
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss
from utils.system_utils import searchForMaxIteration
from utils.train_utils import training_report, prepare_output_and_logger

TENSORBOARD_FOUND = True


class FinetuneColorModel(GaussianModel):
    def __init__(self, sh_degree: int):
        super().__init__(sh_degree=sh_degree)

    def training_setup(self, training_args):
        # self.percent_dense = training_args.percent_dense
        # self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [  # {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], "name": "f_dc"}, {'params': [self._features_rest], "name": "f_rest"},
            # {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            # {'params': [self._scaling], 'lr': training_args.scaling_lr * self.spatial_lr_scale, "name": "scaling"},
            # {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=1e-5, eps=1e-15)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, *, args):

    assert args is not None

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = FinetuneColorModel(dataset.sh_degree)
    load_iteration = args.load_iteration
    logger.info(f"loading iteration {load_iteration}")

    assert load_iteration == -1

    # bring this functionality outside to reuse
    if load_iteration == -1:
        load_iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))

    scene = Scene(dataset, gaussians, load_iteration=load_iteration, pcd_path=None)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # we want to finetune for opt.iteration more iterations
    logger.info(f"train iteration starting at {load_iteration + 1} and ending at {opt.iterations + load_iteration}")

    progress_bar = tqdm(range(load_iteration + 1, opt.iterations + load_iteration + 1), desc="Training progress",
                        dynamic_ncols=True)
    for iteration in range(load_iteration + 1, opt.iterations + load_iteration + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except:
                network_gui.conn = None

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
            render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = Ll1
        # loss = Ll1
        loss.backward()

        # nn.utils.clip_grad_norm_(chain(gaussians.training_params(), (viewspace_point_tensor,)), 5e-4)

        # if app_emb:
        #     amp_optimizer.step()

        iter_end.record()

        # Progress bar
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(10)
        if iteration == opt.iterations:
            progress_bar.close()

        # Log and save
        training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                        testing_iterations, scene, render, (pipe, background))
        if iteration in saving_iterations:
            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=random.randint(10000, 20000, ))
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[15_001, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[15_001, 20_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--app-emb", action="store_true")
    parser.add_argument("--load-iteration", type=int, default=-1, help="load iteration")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    with logger.catch(reraise=True, ):
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
                 args=args)

    # All done
    print("\nTraining complete.")

import numpy as np
import os
import threading
import torch
import typing as t
import uuid
from PIL import Image
from PIL.Image import Resampling
from argparse import Namespace
from jaxtyping import Float
from pathlib import Path
from queue import Queue
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs.base import DatasetConfig
from scene.cameras import Camera
from scene.creator import Scene
from .image_utils import psnr


@torch.no_grad()
def report_status(
    *,
    train_cameras: t.List[Camera],
    test_cameras: t.List[Camera],
    data_config: DatasetConfig,
    render_func: t.Callable[[Camera], t.Dict[str, str | Tensor]],
    tb_writer: t.Optional[SummaryWriter] = None,
    iteration: int,
) -> t.Tuple[t.Dict[str, float], t.Dict[str, float]]:
    tra_iterator = _iterate_over_cameras(
        cameras=train_cameras, data_conf=data_config, shuffle=False, infinite=False
    )
    test_iterator = _iterate_over_cameras(
        cameras=test_cameras, data_conf=data_config, shuffle=False, infinite=False
    )

    def inference(
        *,
        camera_iterator: t.Generator[
            t.Dict[str, t.Union[str, torch.Tensor, Camera]], None, None
        ],
        tb_tag: t.Literal["train", "test"] = "test",
        cur_iteration: int,
    ) -> t.Dict[str, float]:
        l1_list = []
        psnr_list = []
        for camera in tqdm(camera_iterator, desc=f"{tb_tag} iteration {cur_iteration}"):
            viewpoint_cam = camera["camera"]
            gt_image = camera["target"]
            mask = camera["mask"]
            image_name = camera["image_name"]

            render_pkg = render_func(viewpoint_cam)

            (
                image,
                depth,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                accum_alphas,
            ) = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["alphas"],
            )
            l1_list.append(nn.L1Loss()(image * mask, gt_image * mask).mean().item())
            psnr_list.append(psnr(image * mask, gt_image * mask).mean().item())

            if tb_writer:
                tb_writer.add_images(
                    f"{tb_tag}/{image_name}/render",
                    image[None],
                    global_step=cur_iteration,
                )
                tb_writer.add_images(
                    f"{tb_tag}/{image_name}/error",
                    torch.abs(image[None] - gt_image[None]),
                    global_step=cur_iteration,
                )
                tb_writer.add_images(
                    f"{tb_tag}/{image_name}/gt",
                    gt_image[None],
                    global_step=cur_iteration,
                )
        l1_mean = np.mean(l1_list)
        psnr_mean = np.mean(psnr_list)

        if tb_writer:
            tb_writer.add_scalar(f"{tb_tag}/l1_mean", l1_mean, cur_iteration)
            tb_writer.add_scalar(f"{tb_tag}/psnr_mean", psnr_mean, cur_iteration)
        return dict(l1_mean=l1_mean, psnr_mean=psnr_mean)

    tra_result = inference(
        camera_iterator=tra_iterator, tb_tag="train", cur_iteration=iteration
    )
    test_result = inference(
        camera_iterator=test_iterator, tb_tag="test", cur_iteration=iteration
    )
    return tra_result, test_result


def prepare_output_and_logger(model_path: str, config: Namespace):
    if not model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(model_path))
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(config))))

    # Create Tensorboard writer
    tb_writer = SummaryWriter(model_path)
    return tb_writer


def _iterate_over_cameras(
    *,
    cameras: t.Optional[t.List[Camera]],
    data_conf: DatasetConfig,
    shuffle=True,
    infinite: bool = False,
) -> t.Generator[t.Dict[str, t.Union[str, torch.Tensor, Camera]], None, None]:
    def single_epoch():
        cameras_for_cur_epoch = cameras.copy()
        if shuffle:
            shuffle_index = np.random.permutation(len(cameras_for_cur_epoch))
            cameras_for_cur_epoch = [cameras_for_cur_epoch[i] for i in shuffle_index]
        for camera in cameras_for_cur_epoch:
            image_name = camera.image_name
            gt_image = camera.original_image.cuda()
            mask_torch: Float[Tensor, "1 height width"] = torch.ones(
                1, *gt_image.shape[1:], device=gt_image.device, dtype=torch.float32
            )

            if data_conf.mask_dir is not None:
                mask_path = Path(data_conf.mask_dir) / f"{image_name}.png.png"
                with Image.open(mask_path) as fmask:
                    fmask = fmask.convert("L").resize(
                        (gt_image.shape[2], gt_image.shape[1]), Resampling.NEAREST
                    )

                mask = np.array(np.array(fmask) >= 1, dtype=np.float32)
                mask_torch = torch.from_numpy(mask).cuda()[None, ...]

            depth_torch: t.Optional[Float[Tensor, "1 height width"]] = None
            if data_conf.depth_dir is not None:
                depth_path = Path(data_conf.depth_dir) / f"{image_name}.npz"
                depth = np.load(depth_path)["pred"].astype(np.float32, copy=False)
                depth_torch = torch.from_numpy(depth).cuda()[None, ...]
                if depth_torch.shape[1:] != gt_image.shape[1:]:
                    depth_torch = torch.nn.functional.interpolate(
                        depth_torch[None, ...],
                        size=(gt_image.shape[1], gt_image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    )[0, ...]

            yield dict(
                image_name=image_name,
                target=gt_image,
                mask=mask_torch,
                depth=depth_torch,
                camera=camera,
                image_id=camera.image_id,
            )

    if infinite:
        while True:
            yield from single_epoch()
    else:
        yield from single_epoch()


def iterate_over_cameras(
    *,
    scene: t.Optional[Scene] = None,
    cameras: t.Optional[t.List[Camera]] = None,
    data_conf: DatasetConfig,
    shuffle=True,
    num_threads: int = 6,
):
    assert scene is not None or cameras is not None
    assert not (scene is not None and cameras is not None)

    if scene is not None:
        cameras = scene.getTrainCameras()
    else:
        assert cameras is not None

    queue = Queue(maxsize=num_threads)
    thread_lock = threading.Lock()
    camera_iterator = _iterate_over_cameras(
        cameras=cameras, data_conf=data_conf, shuffle=shuffle, infinite=True
    )

    def worker():
        while True:
            with thread_lock:
                cur_camera = next(camera_iterator)
            queue.put(cur_camera)

    threads = []
    for _ in range(num_threads):
        cur_thread = threading.Thread(target=worker, args=(), daemon=True)
        threads.append(cur_thread)

    for cur_thread in threads:
        cur_thread.start()

    while True:
        yield queue.get()

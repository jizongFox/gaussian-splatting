# this is to create a background dataset from a slam pcd.
# or from the slam poses and images
import numpy as np
import open3d as o3d
import torch
import typing as t
from jaxtyping import Float
from open3d.cuda.pybind.geometry import PointCloud
from torch import Tensor
from tqdm import tqdm

from configs.base import DatasetConfig
from gaussian_renderer import pose_depth_render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.train_utils import _iterate_over_cameras


class BackgroundPCDCreator:
    def __init__(
        self,
        *,
        gaussian_model: GaussianModel,
        background: Tensor,
        cameras: t.List[Camera],
        data_config: DatasetConfig,
        alpha_threshold: float = 0.7,
        num_points: int = 1e4,
    ):
        self.gaussians = gaussian_model
        self.background = background
        self.cameras = cameras.copy()
        assert (
            data_config.depth_dir is not None
        ), f"depth_dir is not set in {data_config}"

        self.camera_iter = _iterate_over_cameras(
            cameras=cameras, data_conf=data_config, shuffle=False, infinite=False
        )
        self.alpha_threshold = alpha_threshold
        self.num_points = num_points

    @torch.no_grad()
    def _get_accum_mask(
        self, camera: Camera
    ) -> t.Tuple[Float[Tensor, "1 h w"], Float[Tensor, "1 h w"]]:

        render_pkg = pose_depth_render(
            camera,
            model=self.gaussians,
            bg_color=self.background,
        )

        image, depth, viewspace_point_tensor, visibility_filter, radii, accum_alphas = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
            render_pkg["alphas"],
        )

        return depth, t.cast(
            Float[Tensor, "1 h w"], accum_alphas > self.alpha_threshold
        )

    @torch.no_grad()
    def main(self) -> PointCloud:
        batched_points = []

        for cur_camera_dict in tqdm(self.camera_iter):
            camera = cur_camera_dict["camera"]
            # rel_depth = cur_camera_dict["depth"]

            abs_depth, visibility_mask = self._get_accum_mask(camera)
            if not torch.any(visibility_mask):
                continue

            # compute the scale and shift coefficient from the rel and abs depths
            # scale, shift = normalized_depth_scale_and_shift(
            #     rel_depth, abs_depth, visibility_mask
            # )
            # rescaled_depth = scale.view(-1, 1, 1) * rel_depth + shift.view(-1, 1, 1)
            max_depth = abs_depth[visibility_mask].max() * 1.5

            intrinsic = np.array(
                [
                    [camera.focal_x, 0.0, camera.cx],
                    [0.0, camera.focal_y, camera.cy],
                    [0.0, 0.0, 1.0],
                ]
            )
            uv_grid = (
                np.mgrid[
                    0 : camera.image_width : 1, 0 : camera.image_height : 1
                ].astype(np.float32)
                + 0.5
            )
            uv_grid = uv_grid.transpose(0, -1, -2)[
                0:, ~visibility_mask.detach().cpu().numpy()[0]
            ]
            direction = np.einsum(
                "ij, kj-> ki",
                np.linalg.inv(intrinsic),
                np.concatenate(
                    [uv_grid.T, np.ones(uv_grid.T.shape[:1])[..., None]], axis=-1
                ),
            )
            direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
            direction_in_world = np.einsum(
                "ij, kj-> ki",
                camera.cam2world[:3, :3].detach().cpu().numpy(),
                direction,
            )
            origin = camera.camera_center.detach().cpu().numpy()
            points = origin + direction_in_world * max_depth.detach().cpu().numpy()
            batched_points.append(points)

        points = np.concatenate(batched_points)
        points = points[
            np.random.choice(points.shape[0], int(self.num_points), replace=False)
        ]
        colors = (
            np.ones_like(points) * self.background.detach().cpu().numpy()[None, ...]
        )
        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(points)
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
        return o3d_pcd

        # import plotly.express as pe
        #
        # bg = pe.scatter_3d(
        #     pd.DataFrame(
        #         points,
        #         columns=[*"xyz"],
        #     ),
        #     x="x",
        #     y="y",
        #     z="z",
        # )
        # bg.update_traces(marker=dict(color="blue", size=1))
        # bg.update_layout(scene=dict(aspectmode="data"))
        #
        # fg_points = self.gaussians.xyz[::100].detach().cpu().numpy()
        # fg = pe.scatter_3d(
        #     pd.DataFrame(
        #         fg_points,
        #         columns=[*"xyz"],
        #     ),
        #     x="x",
        #     y="y",
        #     z="z",
        # )
        # fg.update_traces(marker=dict(color="red", size=1))
        # fg.update_layout(scene=dict(aspectmode="data"))
        # import plotly.graph_objects as go
        #
        # big_fig = go.Figure([bg.data[0], fg.data[0]])
        # big_fig.update_layout(scene=dict(aspectmode="data"))
        # big_fig.show()
        #
        # exit()

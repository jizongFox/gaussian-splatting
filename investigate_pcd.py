# this is to investigate the pcd dataformat
import math
import numpy as np
import open3d as o3d
import plotly.express as px
import torch
from loguru import logger
from pathlib import Path
from plyfile import PlyData, PlyElement
from pytorch3d import ops as p3dops
from torch import Tensor
from torch import nn
from tqdm import tqdm

original_path = "/home/jizong/Workspace/gaussian-splatting/output/0812_with_sparsity_loss/white/baseline/point_cloud/iteration_30000/point_cloud.ply"
# modified_path = "output/jizong_meetingroom/backup/Peng/point_cloud2/point_cloud2.ply"
output_path = "/home/jizong/Workspace/gaussian-splatting/output/0812_with_sparsity_loss/white/baseline/point_cloud/iteration_130000/point_cloud.ply"

torch.set_grad_enabled(False)


def _inverse_sigmoid(value):
    if value < 1e-5:
        return -10000
    if value > 1e-5:
        return 10000
    assert 0 < value < 1, value
    return math.log(value / (1 - value))


class ReadPCD:
    def __init__(self, sh_degree):
        self.max_sh_degree = sh_degree

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                False
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def save_ply(self, path, mask=None):
        Path(path).parent.mkdir(exist_ok=True, parents=True)
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self._construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)

        attributes = np.concatenate(
            (
                xyz,
                normals,
                f_dc,
                f_rest,
                opacities,
                scale,
                rotation,
            ),
            axis=1,
        )
        if mask is not None:
            attributes = attributes[mask.astype(bool)]
            elements = elements[mask.astype(bool)]

        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def _construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def create_mask_from_modified_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        xyz = torch.from_numpy(xyz).cuda()
        mask = torch.empty((self._xyz.shape[0],), dtype=bool)
        for index_, cur_item in tqdm(enumerate(self._xyz), total=len(self._xyz)):
            mask[index_] = torch.any(torch.all(cur_item == xyz, dim=-1))
        return mask.detach().cpu().numpy()

    def change_transparency(self, mask, new_transparency):

        with torch.no_grad():
            self._opacity[~mask] = _inverse_sigmoid(new_transparency)

    def plot_3d(self, downscale_ratio: int = 1):
        xyz = self._xyz[::downscale_ratio]
        if isinstance(xyz, Tensor):
            xyz = xyz.detach().cpu()
        data = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
        fig = px.scatter_3d(
            data,
            x="x",
            y="y",
            z="z",
            size=np.ones_like(data["x"]),
            width=100,
            height=100,
        )
        fig.show()

    def create_mask_based_on_opacity(self, threshold: float):
        assert 0 <= threshold <= 1, threshold
        opacity = torch.sigmoid(self._opacity)
        min_opacity = opacity.min()
        if threshold <= min_opacity:
            logger.warning("threshold smaller than lower bound")
        mask = opacity > threshold
        mask = mask.detach().cpu().numpy().squeeze(1)

        return mask

    def create_mask_based_on_3d_scale(
            self, min_thres: float = 0, max_thres: float = 10000
    ):
        scales = torch.exp(self._scaling)
        min_scale = scales.min(dim=1)[0]
        max_scale = scales.max(dim=1)[0]
        max_min_div = max_scale / (min_scale + 1e-8)
        mask = torch.logical_and(max_min_div <= max_thres, max_min_div >= min_thres)
        mask = mask.detach().cpu().numpy()

        return mask

    def remove_outlier(self):
        # remove outliers based on nerfstudio's method
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._xyz.detach().float().cpu().numpy()
        )

        new_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=10.0)
        mask = torch.zeros(self._xyz.shape[0], device=self._xyz.device).bool()
        mask[ind] = True
        return mask.cpu().detach().numpy()

    def get_knn_distance(self):
        # using pytorch3d to get knn distance

        # here the knn distance is weighted by the distance to the center of the object so that the
        # colored ply are pixels to be removed.

        result = p3dops.knn_points(
            p1=self._xyz[None, ...], p2=self._xyz[None, ...][:, ::2], K=100
        )[0][0]
        mean_distance = result.mean(dim=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            self._xyz.detach().float().cpu().numpy()
        )
        pcd.colors = o3d.utility.Vector3dVector(
            (
                    mean_distance[..., None]
                    * torch.exp(-self._xyz.norm(dim=-1, keepdim=True) / 3)
            )
            .repeat(1, 3)
            .cpu()
            .numpy()
        )
        o3d.io.write_point_cloud("test.ply", pcd)

        # this mean distance should be colaborate with xyz range.

    def offline_change_opacity(self):
        opacity = torch.sigmoid(self._opacity)
        ranged_opacity = opacity ** 2 * 2 - 1

        def odd_pow(input, exponent):
            return input.sign() * input.abs().pow(exponent)

        rescaled_opacity = (odd_pow(ranged_opacity, 1 / 11) + 1) / 2

        def _inverse_sigmoid(value: Tensor):
            value.clip_(1e-5, 0.9999)
            return torch.log(value / (1 - value))

        self._opacity = _inverse_sigmoid(rescaled_opacity)


pcd_manager = ReadPCD(3)
mask = None
pcd_manager.load_ply(original_path)

# mask = pcd_manager.remove_outlier()
# pcd_manager.get_knn_distance()
# mask = pcd_manager.create_mask_based_on_opacity(0.05)
# mask2 = pcd_manager.create_mask_based_on_3d_scale(1, 1e4)

# pcd_manager.plot_3d(downscale_ratio=500)
# mask =pcd_manager.de
# mask = pcd_manager.create_mask_from_modified_ply(modified_path)
# pcd_manager.change_transparency(~mask, new_transparency=0)
pcd_manager.offline_change_opacity()
# mask2 = pcd_manager.create_mask_based_on_3d_scale(1, 1000)
mask = pcd_manager.create_mask_based_on_opacity(0.001)

pcd_manager.save_ply(output_path, mask)

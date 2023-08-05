# this is to investigate the pcd dataformat
import math
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
from plyfile import PlyData, PlyElement
from torch import Tensor
from torch import nn
from tqdm import tqdm

original_path = "output/50687083-5/point_cloud/iteration_30000/point_cloud.ply"
modified_path = "output/jizong_meetingroom/backup/Peng/point_cloud2/point_cloud2.ply"
output_path = (
    "output/jizong_meetingroom/backup/Peng/point_cloud2/point_cloud2_output.ply"
)


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
        def _inverse_sigmoid(value):
            if value < 1e-5:
                return -10000
            if value > 1e-5:
                return 10000
            assert 0 < value < 1, value
            return math.log(value / (1 - value))

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


pcd_manager = ReadPCD(3)
pcd_manager.load_ply(original_path)

# pcd_manager.plot_3d(downscale_ratio=500)
# mask =pcd_manager.de
# mask = pcd_manager.create_mask_from_modified_ply(modified_path)
# pcd_manager.change_transparency(~mask, new_transparency=0)

pcd_manager.save_ply(output_path, mask)

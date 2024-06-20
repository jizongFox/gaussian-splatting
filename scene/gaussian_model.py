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
from __future__ import annotations

import numpy as np
import os
import torch
import typing as t
from loguru import logger
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2  # noqa
from torch import nn, Tensor

from gaussian_renderer.finetune_utils import rotation2quat
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.graphics_utils import BasicPointCloud
from utils.sh_utils import RGB2SH
from utils.system_utils import mkdir_p, get_gpu_memory


def setup_functions(self):
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    self._scaling_activation = torch.exp
    self._scaling_inverse_activation = torch.log

    self._covariance_activation = build_covariance_from_scaling_rotation

    self._opacity_activation = torch.sigmoid
    self._inverse_opacity_activation = inverse_sigmoid

    self._rotation_activation = torch.nn.functional.normalize


def merge_gaussian_models(*model: GaussianModel):
    model = [x for x in model if x is not None]
    new_model = GaussianModel(model[0].max_sh_degree)
    new_model.active_sh_degree = model[0].active_sh_degree
    new_model._xyz = torch.cat([m._xyz for m in model], dim=0)
    new_model._features_dc = torch.cat([m._features_dc for m in model], dim=0)
    new_model._features_rest = torch.cat([m._features_rest for m in model], dim=0)
    new_model._scaling = torch.cat([m._scaling for m in model], dim=0)
    new_model._rotation = torch.cat([m._rotation for m in model], dim=0)
    new_model._opacity = torch.cat([m._opacity for m in model], dim=0)
    return new_model


class GaussianModel:
    _covariance_activation: t.Any
    _scaling_activation = torch.exp
    _scaling_inverse_activation = torch.log

    _opacity_activation = torch.sigmoid
    _inverse_opacity_activation = inverse_sigmoid

    _rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int):
        self.active_sh_degree = 0
        self.max_sh_degree: int = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer: torch.optim.Adam = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        setup_functions(self)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def scaling(self):
        return self._scaling_activation(self._scaling)

    @property
    def rotation(self):
        return self._rotation_activation(self._rotation)

    @property
    def xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def opacity(self) -> Tensor:
        opacity: Tensor = self._opacity_activation(self._opacity)
        assert torch.all((opacity <= 1) & (opacity >= 0)), (
            opacity.min(),
            opacity.max(),
        )
        return opacity

    def covariance(self, scaling_modifier=1):
        return self._covariance_activation(
            self.scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
            logger.trace(f"Active sh_degree: {self.active_sh_degree}")

    def is_max_sh(self):
        return self.active_sh_degree == self.max_sh_degree

    def create_from_pcd(
        self,
        pcd: BasicPointCloud,
        spatial_lr_scale: float,
        *,
        max_sphere: float = 1e-3,
        start_opacity: float = 0.1,
    ):
        if spatial_lr_scale == 0.0:
            spatial_lr_scale = 10.0
            logger.warning("Spatial learning rate scale is zero, setting to 10.0")
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        has_normal = False

        if (
            pcd.normals is not None
            and len(np.array(pcd.normals)) > 0
            and np.all(np.linalg.norm(np.array(pcd.normals), axis=-1) > 0.0)
        ):
            logger.warning(f"using normal")
            # breakpoint()
            # has_normal = True
            normals = np.array(pcd.normals)
            normals = normals + np.random.randn(*normals.shape) * 1e-6
            normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
            v3 = torch.from_numpy(normals).float().cuda()
            v2 = torch.cross(
                v3, torch.tensor([0.0, 0.0, 1.0], device="cuda").repeat(v3.shape[0], 1)
            )
            v2 = v2 / torch.linalg.norm(v2, dim=-1, keepdim=True)  # Normalize v2
            v1 = torch.cross(v2, v3)
            v1 = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True)  # Normalize v1
            # compute the rotation matrix
            rotation_matrix = torch.stack((v1, v2, v3), dim=1)

            # from nerfstudio.data.utils.colmap_parsing_utils import rotmat2qvec
            rots_normals = rotation2quat(rotation_matrix.transpose(-1, -2))

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
            0.0000001,
        )
        dist2 = torch.clamp(dist2, 0.0000001, max_sphere)
        scales = torch.log(torch.sqrt(dist2 / 3))[..., None].repeat(1, 3)
        if has_normal:
            scales[:, 2] = torch.log(torch.exp(scales.max(dim=-1)[0]) / 3)
            # pass
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        if has_normal:
            rots = rots_normals

        opacities = inverse_sigmoid(
            start_opacity
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def training_setup(self, training_args) -> torch.optim.Optimizer:
        assert self.spatial_lr_scale > 0, self.spatial_lr_scale
        self.percent_dense = training_args.percent_dense if hasattr(
            training_args, "percent_dense"
        ) else 0.01
        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )
        self._other_scheduler_args = {}  # noqa
        for cur_group in l:
            if cur_group["name"] == "xyz":
                self._other_scheduler_args[cur_group["name"]] = self.xyz_scheduler_args
            elif cur_group["name"] not in ["opacity", "f_dc", "f_rest"]:
                self._other_scheduler_args[cur_group["name"]] = get_expon_lr_func(
                    lr_init=cur_group["lr"],
                    lr_final=cur_group["lr"] / 2,
                    lr_delay_mult=training_args.position_lr_delay_mult,
                    max_steps=training_args.position_lr_max_steps,
                )

        return self.optimizer

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
            else:
                try:
                    lr = self._other_scheduler_args[param_group["name"]](iteration)
                    param_group["lr"] = lr
                except KeyError:
                    continue

    def construct_list_of_attributes(self):
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

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
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    @torch.no_grad()
    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.opacity, torch.ones_like(self.opacity) * 0.01)
        )
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

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
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
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
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
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
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values > self.percent_dense * scene_extent,
        )

        # scale_mask = (torch.max(self.scaling, dim=1).values / (torch.min(self.scaling, dim=1).values + 1e-9)
        #               > 2000)
        #
        # selected_pts_mask = torch.logical_and(
        #     selected_pts_mask,
        #     scale_mask
        # )

        stds = self.scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self._scaling_inverse_activation(
            self.scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        gradient_points = selected_pts_mask.sum()
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.scaling, dim=1).values <= self.percent_dense * scene_extent,
        )
        scale_points = selected_pts_mask.sum() - gradient_points
        logger.trace(
            f"densified points: {selected_pts_mask.sum()}, including gradient-based {gradient_points} "
            f"and scale-based {-scale_points}"
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    @torch.no_grad()
    def densify_and_prune(
        self,
        max_grad,
        min_opacity,
        extent,
        max_screen_size,
        opacity_percentage: float = None,
    ):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        if int(get_gpu_memory()) > 250:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)
        else:
            logger.warning("memory issue only perform prunning")
        prune_mask = (self.opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs: Tensor = self.max_radii2D > max_screen_size  # noqa
            big_points_ws = self.scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )

        # scale_mask = (torch.max(self.scaling, dim=1).values / (torch.min(self.scaling, dim=1).values + 1e-9)
        #               > 10000)
        # if scale_mask.float().sum() > 0:
        #     logger.trace(f"split based on scale range :{scale_mask.float().sum():.1e}.")
        #
        # prune_mask = torch.logical_or(
        #     prune_mask,
        #     scale_mask
        # )

        self.prune_points(prune_mask)

        # if opacity_percentage is not None:
        #     dist_weighted_opacity = self.opacity * (
        #             0.5 * torch.exp(-1 / (self.xyz.norm(dim=-1, keepdim=True) + 1)) + 0.5)
        #     threshold = torch.quantile(dist_weighted_opacity, opacity_percentage)
        #     prune_mask = (dist_weighted_opacity < threshold).squeeze(-1)  # noqa
        #     self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    @torch.no_grad()
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1

    def __len__(self) -> int:
        return len(self.xyz)

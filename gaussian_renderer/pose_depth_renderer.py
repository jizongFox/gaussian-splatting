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
import math
import torch
import typing
from torch import Tensor

from diff_gaussian_rasterization_w_pose_depth import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.sh_utils import eval_sh

if typing.TYPE_CHECKING:
    from scene import Camera
    from scene.gaussian_model import GaussianModel


def render_unified(
    viewpoint_camera: "Camera",
    model: "GaussianModel",
    *,
    background_model: typing.Union["GaussianModel", None] = None,
    bg_color: torch.Tensor,
) -> typing.Tuple[typing.Dict[str, Tensor], ...]:
    if background_model is None:
        return (
            render(
                viewpoint_camera,
                model,
                bg_color=bg_color,
            ),
        )
    else:
        return render_override(
            viewpoint_camera,
            model,
            background_model,
            bg_color=bg_color,
        )


def render(
    viewpoint_camera: "Camera",
    model: "GaussianModel",
    *,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Tensor = None,
    compute_cov3D_python: bool = False,
    convert_SHs_python: bool = False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(
        model.xyz, dtype=model.xyz.dtype, requires_grad=True, device="cuda"
    )

    screenspace_points.retain_grad()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        proj_k=viewpoint_camera.projection_matrix,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = model.xyz
    means2D = screenspace_points
    opacity = model.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    assert compute_cov3D_python is False
    if compute_cov3D_python:
        cov3D_precomp = model.covariance(scaling_modifier)
    else:
        scales = model.scaling
        rotations = model.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    assert convert_SHs_python is False

    if override_color is None:
        if convert_SHs_python:
            shs_view = model.get_features.transpose(1, 2).view(
                -1, 3, (model.max_sh_degree + 1) ** 2
            )
            dir_pp = model.xyz - viewpoint_camera.camera_center.repeat(
                model.get_features.shape[0], 1
            )
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(model.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = model.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    (color, depth, alpha, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        camera_center=viewpoint_camera.camera_center,
        camera_pose=viewpoint_camera.world_view_transform,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": color,
        "depth": depth,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "alphas": alpha,
    }


def render_override(
    viewpoint_camera: "Camera",
    model: "GaussianModel",
    background_model: "GaussianModel",
    *,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means

    # Set up rasterization configuration
    models = [model, background_model]
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=models[0].active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
        proj_k=viewpoint_camera.projection_matrix,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    xyz = torch.cat([model.xyz for model in models], dim=0)

    screenspace_points1 = torch.zeros_like(
        model.xyz, dtype=xyz.dtype, requires_grad=True, device="cuda"
    )
    screenspace_points1.retain_grad()
    screenspace_points2 = torch.zeros_like(
        background_model.xyz, dtype=xyz.dtype, requires_grad=True, device="cuda"
    )
    screenspace_points2.retain_grad()
    screenspace_points = torch.cat([screenspace_points1, screenspace_points2], dim=0)

    means3D = xyz
    means2D = screenspace_points
    opacity = torch.cat([model.opacity for model in models], dim=0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    cov3D_precomp = None

    scales = torch.cat([model.scaling for model in models], dim=0)
    rotations = torch.cat([model.rotation for model in models], dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    colors_precomp = None
    shs = torch.cat([model.get_features for model in models], dim=0)

    # Rasterize visible Gaussians to image, obtain their radii (on screen).

    (color, depth, alpha, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        camera_center=viewpoint_camera.camera_center,
        camera_pose=viewpoint_camera.world_view_transform,
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": color,
        "depth": depth,
        "viewspace_points": screenspace_points1,
        "visibility_filter": radii[: len(model.xyz)] > 0,
        "radii": radii[: len(model.xyz)],
        "alphas": alpha[: len(model.xyz)],
    }, {
        "render": color,
        "depth": depth,
        "viewspace_points": screenspace_points2,
        "visibility_filter": radii[len(model.xyz) :] > 0,
        "radii": radii[len(model.xyz) :],
        "alphas": alpha[len(model.xyz) :],
    }

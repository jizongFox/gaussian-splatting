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

import kornia
import torch
import torch.nn.functional as F
import typing as t
from functools import lru_cache
from math import exp
from torch import Tensor, nn
from torch.autograd import Variable


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


@lru_cache()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def tv_loss(img1, img2):
    tv_h_1 = (img1[:, :, 1:, :] - img1[:, :, :-1, :])
    tv_w_1 = (img1[:, :, :, 1:] - img1[:, :, :, :-1])

    tv_h_2 = (img2[:, :, 1:, :] - img2[:, :, :-1, :])
    tv_w_2 = (img2[:, :, :, 1:] - img2[:, :, :, :-1])

    return (tv_h_1 - tv_h_2).pow(2).sum(dim=0).mean() + (tv_w_1 - tv_w_2).pow(2).sum(dim=0).mean()


class Entropy(nn.Module):
    r"""General Entropy interface

    the definition of Entropy is - \sum p(xi) log (p(xi))

    reduction (string, optional): Specifies the reduction to apply to the output:
    ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
    ``'mean'``: the sum of the output will be divided by the number of
    elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean", eps=1e-16):
        super().__init__()
        self._eps = eps
        self._reduction = reduction

    def forward(self, input_: Tensor) -> Tensor:
        assert input_.shape.__len__() >= 2
        b, _, *s = input_.shape
        e = input_ * (input_ + self._eps).log()
        e = -1.0 * e.sum(1)
        assert e.shape == torch.Size([b, *s])
        if self._reduction == "mean":
            return e.mean()
        elif self._reduction == "sum":
            return e.sum()
        else:
            return e


def hsv_color_space_loss(rgb_img1: Tensor, rg2_img2: Tensor, *,
                         channel_weight: t.Tuple[float | int, float | int, float | int],
                         criterion=l1_loss):
    b, c, h, w = rgb_img1.shape
    assert rg2_img2.shape == rgb_img1.shape
    hsv1, hsv2 = kornia.color.rgb_to_hsv(rgb_img1), kornia.color.rgb_to_hsv(rg2_img2)
    weight = torch.zeros(1, 3, 1, 1, device=torch.device("cuda"), dtype=torch.float)
    weight[0, :, 0, 0] = torch.Tensor(channel_weight).cuda().type(torch.float)
    return (torch.abs(hsv1 - hsv2) * weight).mean()


def _rgb_to_yiq(rgb_img):
    """
    Convert RGB image to YIQ.
    Assumes rgb_img is a PyTorch tensor of shape (batch_size, 3, height, width) and in [0, 1].
    """
    # Define the conversion matrix
    transformation_matrix = torch.tensor([
        [0.299, 0.587, 0.114],
        [0.596, -0.274, -0.322],
        [0.211, -0.523, 0.312]
    ]).to(rgb_img.device)

    # Reshape the image tensor and perform matrix multiplication
    batch_size, channels, height, width = rgb_img.shape
    reshaped_img = rgb_img.view(batch_size, channels, -1)  # Shape: (batch_size, 3, height*width)
    yiq_img = torch.matmul(transformation_matrix, reshaped_img)

    # Reshape the output back to the original shape
    yiq_img = yiq_img.view(batch_size, channels, height, width)
    return yiq_img


def yiq_color_space_loss(rgb_img1: Tensor, rg2_img2: Tensor, *,
                         channel_weight: t.Tuple[float | int, float | int, float | int],
                         criterion=l1_loss):
    b, c, h, w = rgb_img1.shape
    assert rg2_img2.shape == rgb_img1.shape
    hsv1, hsv2 = _rgb_to_yiq(rgb_img1), _rgb_to_yiq(rg2_img2)
    weight = torch.zeros(1, 3, 1, 1, device=torch.device("cuda"), dtype=torch.float)
    weight[0, :, 0, 0] = torch.Tensor(channel_weight).cuda().type(torch.float)
    return (torch.abs(hsv1 - hsv2) * weight).mean()

import torch
import typing as t
from jaxtyping import Float, Bool
from torch import nn, Tensor
from torch.nn import functional as F

from nerfstudio.model_components.losses import (
    ScaleAndShiftInvariantLoss as _ScaleAndShiftInvariantLoss,
    depth_ranking_loss,
)
from nerfstudio.utils.math import masked_reduction


class MiDaSL1Loss(nn.Module):
    """
    data term from MiDaS paper
    """

    def __init__(self, reduction_type: t.Literal["image", "batch"] = "batch"):
        super().__init__()

        self.reduction_type: t.Literal["image", "batch"] = reduction_type
        # reduction here is different from the image/batch-based reduction. This is either "mean" or "sum"
        self.l1_loss = nn.L1Loss(reduction="none")

    def forward(
        self,
        prediction: Float[Tensor, "1 32 mult"],
        target: Float[Tensor, "1 32 mult"],
        mask: Bool[Tensor, "1 32 mult"],
    ) -> Float[Tensor, "0"]:
        """
        Args:
            prediction: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            mse loss based on reduction function
        """
        summed_mask = torch.sum(mask, (1, 2))
        image_loss = torch.sum(self.l1_loss(prediction, target) * mask, (1, 2))
        # multiply by 2 magic number?
        image_loss = masked_reduction(image_loss, 2 * summed_mask, self.reduction_type)

        return image_loss


class ScaleAndShiftInvariantLoss(_ScaleAndShiftInvariantLoss):
    def __init__(self):
        super().__init__()
        self.set_data_loss(MiDaSL1Loss())


def depth_patch_iterator(
    image: Float[Tensor, "1 C height width"],
    patch_size: int = 8,
    stride: int = 8,
) -> t.Generator[Float[Tensor, "1 C patch_size patch_size"], None, None]:
    """
    Args:
        image: image to be split into patches
        patch_size: size of the patch
        stride: stride of the patch
    Returns:
        generator of patches
    """
    _, _, height, width = image.shape
    for i in range(0, height - patch_size + 1, stride):
        for j in range(0, width - patch_size + 1, stride):
            yield image[:, :, i : i + patch_size, j : j + patch_size]


class SparseNerfDepthLoss(nn.Module):
    def __init__(self, patch_size=16, stride=8):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride

    def forward(
        self,
        pred: Float[Tensor, "1 1 height width"],
        target: Float[Tensor, "1 1 height width"],
        mask: Bool[Tensor, "1 1 height width"],
    ):
        """
        Args:
            pred: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            loss based on reduction function
        """
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
        losses = []
        for cur_image, cur_target, cur_mask in zip(
            depth_patch_iterator(pred, patch_size=self.patch_size, stride=self.stride),
            depth_patch_iterator(
                target, patch_size=self.patch_size, stride=self.stride
            ),
            depth_patch_iterator(mask, patch_size=self.patch_size, stride=self.stride),
        ):
            if not cur_mask.any():
                continue
            losses.append(self.forward_patch(cur_image, cur_target, cur_mask))

        return torch.mean(torch.stack(losses))

    def forward_patch(
        self,
        pred_patch: Float[Tensor, "1 1 height weight"],
        target_patch: Float[Tensor, "1 1 height weight"],
        mask: Bool[Tensor, "1 1 height weight"],
    ):
        """
        Args:
            pred_patch: predicted depth patch
            target_patch: ground truth depth patch
            mask: mask of valid pixels
        Returns:
            loss based on reduction function
        """
        if not mask.any():
            return torch.tensor(0.0, device=pred_patch.device, dtype=pred_patch.dtype)
        pred_patch = pred_patch[mask]
        target_patch = target_patch[mask]

        pred_patch = pred_patch.view(-1)
        target_patch = target_patch.view(-1)

        random_sample_index = torch.randperm(pred_patch.shape[0])[: pred_patch.shape[0]]
        input_pred = torch.empty(
            (pred_patch.shape[0] * 2, 1),
            device=pred_patch.device,
            dtype=pred_patch.dtype,
        )
        input_pred[::2, 0] = pred_patch
        input_pred[1::2, 0] = pred_patch[random_sample_index]

        input_target = torch.empty(
            (target_patch.shape[0] * 2, 1),
            device=target_patch.device,
            dtype=target_patch.dtype,
        )
        input_target[::2, 0] = target_patch
        input_target[1::2, 0] = target_patch[random_sample_index]

        return depth_ranking_loss(input_pred, input_target)


class SobelDepthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction="none")
        self.sobel = GradLayer()

    def forward(
        self,
        pred: Float[Tensor, "1 1 height width"],
        target: Float[Tensor, "1 1 height width"],
        mask: Bool[Tensor, "1 1 height width"],
    ):
        """
        Args:
            pred: predicted depth map
            target: ground truth depth map
            mask: mask of valid pixels
        Returns:
            loss based on reduction function
        """
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            mask = mask.unsqueeze(0)
        pred_edge = self.sobel(pred)
        target_edge = self.sobel(target)
        loss = self.loss(pred_edge, target_edge)
        loss = torch.mean(loss[mask])
        return loss


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h)[None, None, ...]
        kernel_v = torch.FloatTensor(kernel_v)[None, None, ...]
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def _get_gray(self, x):
        """
        Convert image to its gray one.
        """
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)

    def forward(self, x):
        if x.shape[1] == 3:
            x = self._get_gray(x)

        x_v = F.conv2d(x, self.weight_v, padding=1)
        x_h = F.conv2d(x, self.weight_h, padding=1)
        x = torch.sqrt(torch.pow(x_v, 2) + torch.pow(x_h, 2) + 1e-6)

        return x

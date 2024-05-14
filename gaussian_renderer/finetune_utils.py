import numpy as np
import torch
import torchvision
import typing as t
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn import functional as F

from utils.graphics_utils import getWorld2View2

if t.TYPE_CHECKING:
    from scene.cameras import Camera


def quat2rotation(r: Float[Tensor, "batch 4"]) -> Float[Tensor, "batch 3 3"]:
    assert r.ndim == 2, r.shape
    q = F.normalize(r, p=2, dim=1)
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]

    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotation2quat(R: Float[Tensor, "batch 3 3"]) -> Float[Tensor, "batch 4"]:
    q = torch.zeros(
        R.shape[0], 4, device=R.device, dtype=R.dtype
    )  # (batch_size, 4) x, y, z, w
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q0_mask = trace > 0
    q1_mask = (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2]) & ~q0_mask
    q2_mask = (R[..., 1, 1] > R[..., 2, 2]) & ~q0_mask & ~q1_mask
    q3_mask = ~q0_mask & ~q1_mask & ~q2_mask
    if q0_mask.any():
        R_for_q0 = R[q0_mask]
        S_for_q0 = 0.5 / torch.sqrt(1 + trace[q0_mask])
        q[q0_mask, 3] = 0.25 / S_for_q0
        q[q0_mask, 0] = (R_for_q0[..., 2, 1] - R_for_q0[..., 1, 2]) * S_for_q0
        q[q0_mask, 1] = (R_for_q0[..., 0, 2] - R_for_q0[..., 2, 0]) * S_for_q0
        q[q0_mask, 2] = (R_for_q0[..., 1, 0] - R_for_q0[..., 0, 1]) * S_for_q0

    if q1_mask.any():
        R_for_q1 = R[q1_mask]
        S_for_q1 = 2.0 * torch.sqrt(
            1 + R_for_q1[..., 0, 0] - R_for_q1[..., 1, 1] - R_for_q1[..., 2, 2]
        )
        q[q1_mask, 0] = 0.25 * S_for_q1
        q[q1_mask, 1] = (R_for_q1[..., 0, 1] + R_for_q1[..., 1, 0]) / S_for_q1
        q[q1_mask, 2] = (R_for_q1[..., 0, 2] + R_for_q1[..., 2, 0]) / S_for_q1
        q[q1_mask, 3] = (R_for_q1[..., 2, 1] - R_for_q1[..., 1, 2]) / S_for_q1

    if q2_mask.any():
        R_for_q2 = R[q2_mask]
        S_for_q2 = 2.0 * torch.sqrt(
            1 + R_for_q2[..., 1, 1] - R_for_q2[..., 0, 0] - R_for_q2[..., 2, 2]
        )
        q[q2_mask, 0] = (R_for_q2[..., 0, 1] + R_for_q2[..., 1, 0]) / S_for_q2
        q[q2_mask, 1] = 0.25 * S_for_q2
        q[q2_mask, 2] = (R_for_q2[..., 1, 2] + R_for_q2[..., 2, 1]) / S_for_q2
        q[q2_mask, 3] = (R_for_q2[..., 0, 2] - R_for_q2[..., 2, 0]) / S_for_q2

    if q3_mask.any():
        R_for_q3 = R[q3_mask]
        S_for_q3 = 2.0 * torch.sqrt(
            1 + R_for_q3[..., 2, 2] - R_for_q3[..., 0, 0] - R_for_q3[..., 1, 1]
        )
        q[q3_mask, 0] = (R_for_q3[..., 0, 2] + R_for_q3[..., 2, 0]) / S_for_q3
        q[q3_mask, 1] = (R_for_q3[..., 1, 2] + R_for_q3[..., 2, 1]) / S_for_q3
        q[q3_mask, 2] = 0.25 * S_for_q3
        q[q3_mask, 3] = (R_for_q3[..., 1, 0] - R_for_q3[..., 0, 1]) / S_for_q3

    q = q[:, [3, 0, 1, 2]]
    return q


def build_rotation(r):
    q = F.normalize(r, p=2, dim=1)
    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def multiply_quaternions(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Performs batch-wise quaternion multiplication.

    Given two quaternions, this function computes their product. The operation is
    vectorized and can be performed on batches of quaternions.

    Args:
        q: A tensor representing the first quaternion or a batch of quaternions.
           Expected shape is (... , 4), where the last dimension contains quaternion components (w, x, y, z).
        r: A tensor representing the second quaternion or a batch of quaternions with the same shape as q.
    Returns:
        A tensor of the same shape as the input tensors, representing the product of the input quaternions.
    """
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    y = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    z = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    return torch.stack((w, x, y, z), dim=-1)


def viewpoint2cw(viewpoint: "Camera"):
    W2C = getWorld2View2(viewpoint.R, viewpoint.T)
    C2W = np.linalg.inv(W2C)
    return C2W


def c2w2viewpoint():
    pass


def initialize_pose_delta(num_images, device):
    return torch.zeros(num_images, 3, device=device)


def initialize_quat_delta(num_images: int, device) -> Float[Tensor, "batch 4"]:
    R = torch.eye(3, device=device).unsqueeze(0).repeat(num_images, 1, 1)
    quat_result = rotation2quat(R)
    assert quat2rotation(quat_result).allclose(R)
    return quat_result


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


if __name__ == "__main__":
    quats = initialize_quat_delta(100, torch.device("cuda"))
    breakpoint()


def apply_affine(
    image: Float[Tensor, "b 3 h w"],
    cx: Float[Tensor, "b 1"],
    cy: Float[Tensor, "b 1"],
    fx_delta: Float[Tensor, "b 1"] | None = None,
    fy_delta: Float[Tensor, "b 1"] | None = None,
    padding_value: float | Tensor = 1.0,
    mode: t.Literal["bilinear", "nearest"] = "bilinear",
) -> Float[Tensor, "b 3 h w"]:
    if isinstance(padding_value, Tensor) and len(padding_value.shape) == 1:
        padding_value = padding_value[None, ..., None, None]
    affine_matrix = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        device=image.device,
        dtype=image.dtype,
    )
    affine_matrix[..., 2] = torch.stack([cx, cy], dim=-1)

    if fx_delta is not None and fy_delta is not None:
        affine_matrix[0, 0] = 1 + fx_delta
        affine_matrix[1, 1] = 1 + fy_delta

    grid = F.affine_grid(affine_matrix[None, ...], image.size(), align_corners=False)

    x = (
        F.grid_sample(image - padding_value, grid, mode=mode, align_corners=False)
        + padding_value
    )
    return x


class GradLayer(nn.Module):
    def __init__(self):
        super(GradLayer, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
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

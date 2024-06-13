import torch
import typing as t
from torch import nn
from torch.utils.tensorboard import SummaryWriter

if t.TYPE_CHECKING:
    from scene.cameras import Camera


class ExposureManager(nn.Module):

    def __init__(self, cameras: t.List["Camera"]):
        super().__init__()
        self.cameras = cameras
        self._image_names = [x.image_name for x in self.cameras]
        self.exposure_a = torch.nn.Embedding(num_embeddings=len(cameras), embedding_dim=1, )
        self.exposure_a.weight.requires_grad = False
        self.exposure_a.weight.data.fill_(0.0)
        self.exposure_b = torch.nn.Embedding(num_embeddings=len(cameras), embedding_dim=1)
        self.exposure_b.weight.requires_grad = False
        self.exposure_b.weight.data.fill_(0.0)

    def forward(self, rendered_image: torch.Tensor, image_name: str):
        cur_index = torch.tensor([self._image_names.index(image_name)], device=rendered_image.device, dtype=torch.long)
        ea = self.exposure_a(cur_index)
        eb = self.exposure_b(cur_index)
        image_ab = torch.exp(ea) * rendered_image + eb
        return image_ab

    def setup_optimizer(self, lr=1e-4, wd: float = 1e-5):
        self.exposure_b.weight.requires_grad = True
        self.exposure_a.weight.requires_grad = True
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def record_exps(self, tensorboard: SummaryWriter, iteration: int):
        tensorboard.add_histogram("exposure_a", self.exposure_a.weight.detach().clone().cpu().numpy().squeeze(),
                                  global_step=iteration)
        tensorboard.add_histogram("exposure_b", self.exposure_b.weight.detach().clone().cpu().numpy().squeeze(),
                                  global_step=iteration)


class ExposureManager2(nn.Module):

    def __init__(self, cameras: t.List["Camera"]):
        super().__init__()
        self.cameras = cameras
        self._image_names = [x.image_name for x in self.cameras]
        self.exposure_a = torch.nn.Embedding(num_embeddings=len(cameras), embedding_dim=1, )
        self.exposure_a.weight.requires_grad = False
        self.exposure_a.weight.data.fill_(1.0)
        self.exposure_b = torch.nn.Embedding(num_embeddings=len(cameras), embedding_dim=1)
        self.exposure_b.weight.requires_grad = False
        self.exposure_b.weight.data.fill_(0.0)

    def forward(self, rendered_image: torch.Tensor, image_name: str):
        cur_index = torch.tensor([self._image_names.index(image_name)], device=rendered_image.device, dtype=torch.long)
        ea = self.exposure_a(cur_index)
        eb = self.exposure_b(cur_index)
        image_ab = torch.pow(rendered_image, ea) + eb
        return image_ab

    def setup_optimizer(self, lr=1e-4, wd: float = 1e-5):
        self.exposure_b.weight.requires_grad = True
        self.exposure_a.weight.requires_grad = True
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def record_exps(self, tensorboard: SummaryWriter, iteration: int):
        tensorboard.add_histogram("exposure_a", self.exposure_a.weight.detach().clone().cpu().numpy().squeeze(),
                                  global_step=iteration)
        tensorboard.add_histogram("exposure_b", self.exposure_b.weight.detach().clone().cpu().numpy().squeeze(),
                                  global_step=iteration)

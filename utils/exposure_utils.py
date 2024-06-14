import tinycudann as tcnn
import torch
import typing as t
from functools import lru_cache
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


class ExposureGrid(nn.Module):
    def __init__(self, cameras: t.List["Camera"], ):
        super().__init__()
        self.cameras = cameras
        self._image_names = [x.image_name for x in self.cameras]

        encode_config = {"otype": "HashGrid", "n_levels": 4, "n_features_per_level": 2, "log2_hashmap_size": 15,
                         "base_resolution": 8, "per_level_scale": 1.5}
        network_config = {"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 16,
                          "n_hidden_layers": 1}

        initial_grid_network = lambda: tcnn.NetworkWithInputEncoding(2, n_output_dims=12, encoding_config=encode_config,
                                                                     network_config=network_config)

        self.encoding_dict = nn.ParameterDict({k: initial_grid_network() for k in self._image_names})

    def forward(self, rendered_image: torch.Tensor, image_name: str):
        grid = self.encoding_dict[image_name]
        width, height = rendered_image.shape[-2:]
        grid_input = self.retrieve_grid_given_image_shape(width, height)
        cur_size = grid_input.shape[:2]
        grid_output = grid(grid_input.view(-1, 2)).view(*cur_size, 12)
        affine_transform = grid_output.view(*cur_size, 3, 4)
        delta_affine = torch.eye(3, 4, device=rendered_image.device)[None, None, ...].expand(*cur_size, -1, -1)
        delta_affine = delta_affine + affine_transform
        correct_image = torch.einsum("cwh,whcl->lwh", rendered_image, delta_affine[..., :3].float())

        correct_image = torch.pow(correct_image + delta_affine[..., 3].float().permute(2, 0, 1), 1)

        self.diff = (rendered_image - correct_image).abs().mean(dim=0)

        return correct_image

    @staticmethod
    @lru_cache()
    def retrieve_grid_given_image_shape(width: int, height: int):
        meshgrid = list(torch.meshgrid([torch.arange(width), torch.arange(height)]))
        meshgrid[0] = (meshgrid[0] + 0.5) / width
        meshgrid[1] = (meshgrid[1] + 0.5) / height
        grid_input = torch.stack(meshgrid, dim=-1).to(torch.device("cuda")).float()
        return grid_input

    def setup_optimizer(self, lr=1e-4, wd: float = 1e-5):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

    def record_exps(self, tensorboard: SummaryWriter, iteration: int):
        tensorboard.add_image(tag="type3", img_tensor=self.diff[None, ...], global_step=iteration)

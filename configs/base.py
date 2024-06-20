import tyro
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
from typing import Tuple, Union, TYPE_CHECKING, List


@dataclass
class _BaseConfig:
    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


@dataclass
class ModelConfig:
    sh_degree: int = 1
    white_background: bool = True


@dataclass(kw_only=True)
class DatasetConfig:
    image_dir: Path
    """image folder path"""
    mask_dir: Path | None
    """mask folder"""
    depth_dir: Path | None = None
    """depth folder"""
    resolution: int = 1
    pcd_path: Path
    """ loading point cloud path. """
    pcd_start_opacity: float = 0.1
    """ starting opacity for the point cloud. """
    remove_pcd_color: bool = False

    max_sphere_distance: float = 1e-3
    force_centered_pp: bool = False
    """ force the principal point to be centered. """

    eval_every_n_frame: int = 8
    """ evaluate every n frame. """

    eval_mode: bool = True


@dataclass(kw_only=True)
class ColmapDatasetConfig(DatasetConfig):
    sparse_dir: Path
    """ sparse folder for colmap. """


@dataclass(kw_only=True)
class SlamDatasetConfig(DatasetConfig):
    meta_file: Path
    """ meta_file for slam. """


@dataclass(kw_only=True)
class _OptimizationConfig(_BaseConfig):
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.005


@dataclass(kw_only=True)
class OptimizerConfig(_OptimizationConfig):
    percent_dense: float = 0.01  # this is to reduce the size of the eclipse
    lambda_dssim: float = 0.2
    densification_interval: int = 500
    opacity_reset_interval: int = 500000000
    densify_from_iter: int = 4000
    densify_until_iter: int = 12_000
    densify_grad_threshold: float = 0.00001  # this is to split more

    min_opacity: float = 1e-5

    def __post_init__(self):
        if self.lambda_dssim == 0:
            logger.warning(f"dssim is disabled.")
        elif self.lambda_dssim == 1:
            raise ValueError("pixelwise loss is disabled.")


@dataclass(kw_only=True)
class BackgroundConfig(_OptimizationConfig):
    pass


@dataclass(kw_only=True)
class FinetuneOptimizerConfig(OptimizerConfig):
    position_lr_init: float = 0.0000001
    position_lr_final: float = 0.000000001
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.00001
    opacity_lr: float = 0.00001
    scaling_lr: float = 0.00001
    rotation_lr: float = 0.000001
    percent_dense: float = 0.01  # this is to reduce the size of the eclipse
    lambda_dssim: float = 0.2
    densification_interval: int = 5000000
    opacity_reset_interval: int = 500000000
    densify_from_iter: int = 4000
    densify_until_iter: int = 12_000
    densify_grad_threshold: float = 0.00001  # this is to split more


@dataclass(kw_only=True)
class ControlConfig(_BaseConfig):
    save_dir: Path
    iterations: int = 15_000
    pose_lr_init: float = 0.0

    num_evaluations: int = 10
    include_0_epoch: bool = False
    test_iterations: List[int] = field(init=False, default_factory=lambda: [])

    rig_optimization: bool = False


if not TYPE_CHECKING:
    DATACONFIG = tyro.extras.subcommand_type_from_defaults(
        {
            "colmap": ColmapDatasetConfig(
                image_dir=Path(),
                mask_dir=None,
                resolution=1,
                pcd_path=Path(),
                sparse_dir=Path(),
            ),
            "slam": SlamDatasetConfig(
                image_dir=Path(),
                mask_dir=None,
                resolution=1,
                pcd_path=Path(),
                meta_file=Path(),
            ),
        },
        prefix_names=False,
    )
    OPTCONFIG = tyro.extras.subcommand_type_from_defaults(
        {
            "optimizer": OptimizerConfig(),
            "finetune": FinetuneOptimizerConfig(),
        },
        prefix_names=False,
    )
else:
    DATACONFIG = Union[ColmapDatasetConfig, SlamDatasetConfig]
    OPTCONFIG = Union[OptimizerConfig, FinetuneOptimizerConfig]


@dataclass(kw_only=True)
class ExperimentConfig(_BaseConfig):
    model: ModelConfig

    dataset: DATACONFIG

    optimizer: OPTCONFIG

    bkg_optimizer: BackgroundConfig | None = None

    control: ControlConfig

    placeholder: str | None = None

    @property
    def save_dir(self):
        return self.control.save_dir

    @property
    def iterations(self):
        return self.control.iterations

    def __post_init__(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        test_iterations = [
            self.iterations // self.control.num_evaluations * i
            for i in range(1, self.control.num_evaluations)
        ]
        test_iterations.append(self.iterations)
        self.control.test_iterations = test_iterations
        if self.control.include_0_epoch:
            if 1 not in self.control.test_iterations:
                self.control.test_iterations.append(1)


if __name__ == "__main__":
    exp_config = tyro.cli(ExperimentConfig)
    print(exp_config)

import typing as t
import tyro
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

from configs.base import (
    ExperimentConfig,
    ModelConfig,
    ControlConfig,
    FinetuneOptimizerConfig,
    SlamDatasetConfig,
)
from visibility_helper import main as visibility_main


@dataclass
class VisibilityRunConfig:
    image_dir: Path
    meta_file: Path
    pcd_path: Path
    save_dir: Path
    image_extension: t.Literal[".jpeg", ".jpg", ".png"] = ".jpeg"
    alpha_threshold: float = 0.8

    def main(self) -> Path:
        """create visibility json and return its path."""

        slam_config = SlamDatasetConfig(
            image_dir=self.image_dir,
            mask_dir=None,
            depth_dir=None,
            meta_file=self.meta_file,
            pcd_path=self.pcd_path,
            remove_pcd_color=False,
            resolution=8,
            max_sphere_distance=0.01,
            eval_mode=False,
            pcd_start_opacity=1.0,
        )

        finetuneConfig = ExperimentConfig(
            model=ModelConfig(sh_degree=1, white_background=True),
            dataset=slam_config,
            optimizer=FinetuneOptimizerConfig(),
            control=ControlConfig(
                iterations=40000,
                save_dir=self.save_dir,
                num_evaluations=0,
            ),
        )
        num_images = len(list(self.image_dir.rglob(f"*{self.image_extension}")))
        logger.info(f"Number of images: {num_images}")
        assert num_images > 0, self.image_dir

        finetuneConfig.control.iterations = num_images
        config = tyro.cli(
            tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig})
        )
        visibility_path: Path = visibility_main(config, self.alpha_threshold)
        return visibility_path

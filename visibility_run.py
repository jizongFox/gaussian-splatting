import tyro
from pathlib import Path

from configs.base import (
    ExperimentConfig,
    ModelConfig,
    ControlConfig,
    FinetuneOptimizerConfig,
    SlamDatasetConfig,
)
from visibility_helper import main as visibility_main

# %% configuration
slam_data_dir = Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2")
save_dir = Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/visibility")

slam_config = SlamDatasetConfig(
    image_dir=Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/images"),
    mask_dir=None,
    depth_dir=None,
    meta_file=Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/meta_updated.json"),
    pcd_path=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside-opencv.ply"),
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
        save_dir=save_dir,
        num_evaluations=0,
    ),
)

finetuneConfig.optimizer.iterations = 40000

config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"ft": finetuneConfig}))
visibility_main(config)

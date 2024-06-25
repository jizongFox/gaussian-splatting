import tyro
from pathlib import Path

from configs.base import (
    ColmapDatasetConfig,
    OptimizerConfig,
    BackgroundConfig,
    ControlConfig,
    ExperimentConfig,
    ModelConfig,
)
from train_helper import main

save_dir = Path(
    "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/outputs/3dgs-learning-rate-scheduler-3"
)

colmap_config = ColmapDatasetConfig(
    image_dir=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/images"
    ),
    mask_dir=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/masks"
    ),
    # mask_dir=None,
    depth_dir=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/depths"
    ),
    resolution=2,
    pcd_path=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside-opencv.ply"
    ),
    pcd_start_opacity=0.5,
    max_sphere_distance=1e-3,
    sparse_dir=Path(
        "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/colmap-simpler-alignment/sparse/0"
    ),
    eval_every_n_frame=45,
)
optimizer_config = OptimizerConfig(
    position_lr_init=0.00002,
    position_lr_final=0.000002,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=9_000,
    feature_lr=0.002,
    opacity_lr=0.025,
    scaling_lr=0.002,
    rotation_lr=0.002,
    percent_dense=0.01,
    lambda_dssim=0.2,
    densification_interval=200,
    opacity_reset_interval=4000,
    densify_from_iter=1000,
    densify_until_iter=20_000,
    densify_grad_threshold=0.00005,
    min_opacity=2e-3,
)
bkg_optimizer_config = BackgroundConfig(
    position_lr_init=0.000016,
    position_lr_final=0.0000016,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=30_000,
    feature_lr=0.005,
    opacity_lr=0.02,
    scaling_lr=0.002,
    rotation_lr=0.002,
)
control_config = ControlConfig(
    iterations=28_000,
    save_dir=save_dir,
    num_evaluations=16,
    include_0_epoch=True,
    pose_lr_init=0e-5,
)

exp_config = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=colmap_config,
    optimizer=optimizer_config,
    bkg_optimizer=bkg_optimizer_config,
    control=control_config,
)
config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"_": exp_config}))
main(config, pose_checkpoint_path=None, use_bkg_model=True)

import tyro
from pathlib import Path

from configs.base import (
    ColmapDatasetConfig,
    OptimizerConfig,
    BackgroundConfig,
    ControlConfig,
    ExperimentConfig,
    ModelConfig,
    SlamDatasetConfig,
)
from train_helper import main

save_dir = Path(
    "/data/2024-06-19/subregion1/outputs/3dgs-slam-pcd-pose-optimize-5e-4-update-new-pose-camera-rig-6"
)

colmap_config = SlamDatasetConfig(
    image_dir=Path("/data/2024-06-19/subregion1/images"),
    mask_dir=Path("/data/2024-06-19/subregion1/masks"),
    # mask_dir=None,
    depth_dir=Path("/data/2024-06-19/undistorted/depths"),
    resolution=1,
    pcd_path=Path("/data/2024-06-19/pixel_lvl1_water2_resampled2.ply"),
    pcd_start_opacity=0.5,
    max_sphere_distance=1e-3,
    # sparse_dir=Path(
    #     "/data/2024-06-19/subregion1/colmap/BA/prior_sparse"
    # ),
    meta_file=Path("/data/2024-06-19/undistorted/meta.json"),
    eval_every_n_frame=45,
)

# slam_config = SlamDatasetConfig(
#     image_dir=Path(
#         "/data/punggol_jetty.dslam/subregion1/images"
#     ),
#
#     mask_dir=Path(
#         "/data/punggol_jetty.dslam/subregion1/masks"
#     ),
#     depth_dir=Path(
#         "/data/punggol_jetty.dslam/undistorted/depths"
#     ),
#     resolution=1,
#     pcd_path=Path(
#         "/data/punggol_jetty.dslam/punggol_jetty-subregion1-opencv.ply"
#     ),
#     pcd_start_opacity=0.5,
#     remove_pcd_color=False,
#     max_sphere_distance=1e-3,
#     eval_every_n_frame=45,
#     eval_mode=True,
#     meta_file=Path("/data/punggol_jetty.dslam/undistorted/meta.json"),
# )

optimizer_config = OptimizerConfig(
    position_lr_init=0.00001,
    position_lr_final=0.0000002,
    position_lr_delay_mult=0.01,
    position_lr_max_steps=9_000,
    feature_lr=0.002,
    opacity_lr=0.025,
    scaling_lr=0.002,
    rotation_lr=0.005,
    percent_dense=0.01,
    lambda_dssim=0.25,
    densification_interval=200,
    opacity_reset_interval=12_00000000,
    densify_from_iter=2_0000000,
    densify_until_iter=20_0000000,
    densify_grad_threshold=0.0005,
    min_opacity=1.0e-2,
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
    pose_lr_init=2e-4,
    rig_optimization=True,
)

exp_config = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=colmap_config,
    optimizer=optimizer_config,
    bkg_optimizer=bkg_optimizer_config,
    control=control_config,
)
config = tyro.cli(tyro.extras.subcommand_type_from_defaults({"_": exp_config}))
main(config, pose_checkpoint_path=None, use_bkg_model=False)

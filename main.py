from pathlib import Path

from configs.base import (
    OptimizerConfig,
    BackgroundConfig,
    ControlConfig,
    ExperimentConfig,
    ModelConfig,
    SlamDatasetConfig,
    ColmapDatasetConfig,
)
from preprocess import process_main
from train_helper import main as train_main

## data preprocessing
tar_path = Path(
    "/data/pixel_lvl1_water2_resampled.tar"
)
output_dir = Path("/data/2024-06-26/colmap")

use_colmap = True

# process_main(
#     tar_path,
#     output_dir,
#     run_colmap=use_colmap,
# )

## training
save_dir = output_dir / "outputs"

# if you use slam dataset directly!!
slam_config = SlamDatasetConfig(
    image_dir=output_dir / "subregion" / "images",
    mask_dir=output_dir / "subregion" / "masks",
    # mask_dir=None,
    depth_dir=output_dir / "subregion" / "depths",
    resolution=1,
    pcd_path=output_dir / "undistorted" / "opencv_pcd.ply",
    # attention to the filename, this is not fixed.
    pcd_start_opacity=0.5,
    max_sphere_distance=1e-3,
    meta_file=output_dir / "undistorted/meta.json",
    eval_every_n_frame=45,
)
# or if you use colmap dataset, which requires you running colmap in prior.
colmap_config = ColmapDatasetConfig(
    image_dir=output_dir / "subregion" / "images",
    mask_dir=output_dir / "subregion" / "masks",
    # mask_dir=None,
    depth_dir=output_dir / "subregion" / "depths",
    resolution=1,
    pcd_path=output_dir / "undistorted" / "opencv_pcd.ply",
    # attention to the filename, this is not fixed.
    pcd_start_opacity=0.5,
    max_sphere_distance=1e-3,
    eval_every_n_frame=45,
    sparse_dir=output_dir / "subregion" / "colmap/prior_sparse",
)

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
    rig_optimization=True if not use_colmap else False,
)

exp_config = ExperimentConfig(
    model=ModelConfig(sh_degree=1, white_background=True),
    dataset=colmap_config if use_colmap else slam_config,
    optimizer=optimizer_config,
    bkg_optimizer=bkg_optimizer_config,
    control=control_config,
)
train_main(exp_config, pose_checkpoint_path=None, use_bkg_model=False)

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

"""
usage tutorial for the pipeline.

this pipeline takes a tar file, untar it, and then run the training pipeline.
in general, one can only set `tar_path` and `output_dir` to run the pipeline.

By default, 
1. it would not run colmap and only rely on the slam data for automatic pose refinement.
2. it would not use the background model.
3. it would not densify the points or prune the points. 

It should work fine with a dense point cloud and offers good quality. 

***As it is the first deployment and it lacks of sufficient testing. 
it would not be guaranteed to work on all datasets.***

As alternative ways to use it. we reserve the option of using colmap and the background model.

* if `use_colmap = True`, it would use the colmap for pose estimation, assisted by slam coarse poses. 
This will take extra 1-2 hours. 

* if `use_background_model = True`, it would use the background model to model the background, to compensate the missing
points in the background (due to limited point range of 100-120 meters). 

* if one wants to densify the points, or the point cloud is quite noisy, please change the following parameters:

optimizer_config = OptimizerConfig(
    ...
    opacity_reset_interval=12_00000000-> 4_000,
    densify_from_iter=2_0000000 -> 3_500,
    densify_until_iter=20_0000000-> 20_000,
    ...
)

This configuration usually works well for densification, but not sure for all cases.


***Please check the outputs and the logs to make sure the pipeline works as expected.*** 


"""

#################### set up the paths #######################
tar_path = Path("/data/pixel_lvl1_water2_resampled.tar")
output_dir = Path("/data/2024-06-26/colmap")

use_colmap = False
use_background_model = False

####################### main pipeline #######################

## data preprocessing
"""
                main details of preprocessing

The preprocessing consists of 11 steps

1. untar the tar file to `output_dir/raw`
2. convert slam json data to a pre-defined format.
3. undistort all images based on distortion parameters of camera
4. processing point cloud to opencv convention
5. visibility check based on the point cloud
6. create a subset of the dataset including images seen by the point cloud
7. generate head mask
8. generate yolo object mask
9. merge masks
10. depth inference by a large model
11. optional, run colmap using slam prior.

in the end of the preprocessing, you will see the structure as the following:
.
├── raw
│   ├── DECXIN20230102350
│   ├── DECXIN2023012346
│   ├── DECXIN2023012347
│   ├── DECXIN2023012348
│   ├── LiDAR-122322001000
│   ├── pixel_lvl1_water2_resampled2.dcloud
│   └── slamMeta.json
├── subregion # optional, there would be a colmap folder.
│   ├── depths
│   ├── images
│   └── masks
│
└── undistorted
    ├── images
    ├── meta.json
    ├── opencv-pcd.ply
    └── visibility

"""

process_main(
    tar_path,
    output_dir,
    run_colmap=use_colmap,
)

############# training pipeline ############################

""" 
Training is easier as it has only a few configuration to define. 
I define the meaning of each parameter in the class definition.

the checkpoints should be saved in `output_dir/outputs`

"""

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

bkg_optimizer_config = None
if use_background_model:
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
train_main(exp_config, use_bkg_model=use_background_model)

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
from dataclasses import dataclass

# %% configuration
# slam_data_dir = Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2")
# save_dir = Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/subregion2/visibility")

# slam_config = SlamDatasetConfig(
#     image_dir=Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/images"),
#     mask_dir=None,
#     depth_dir=None,
#     meta_file=Path("/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/meta_updated.json"),
#     pcd_path=Path(
#         "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside-opencv.ply"),
#     remove_pcd_color=False,
#     resolution=8,
#     max_sphere_distance=0.01,
#     eval_mode=False,
#     pcd_start_opacity=1.0,
# )

# finetuneConfig = ExperimentConfig(
#     model=ModelConfig(sh_degree=1, white_background=True),
#     dataset=slam_config,
#     optimizer=FinetuneOptimizerConfig(),
#     control=ControlConfig(
#         iterations=40000,
#         save_dir=save_dir,
#         num_evaluations=0,
#     ),
# )

# finetuneConfig.optimizer.iterations = 40000

@dataclass
class VisibilityRunConfig:
    input_path: Path
    output_path: Path
    pcd_path: Path

    def __post_init__(self):
        assert self.input_path.exists(), f"Config path {self.input_path} does not exist"
    
    def _create_slam_config(self):
        # I want the save dir to be at /data/pixel_lv2_vaibi/undistorted/visibility
       
        slam_data_dir = Path(self.output_path.parent, "subregion2")

        slam_config = SlamDatasetConfig(
            image_dir=Path(self.output_path, "images"),
            mask_dir=None,
            depth_dir=None,
            meta_file=Path(self.output_path, "meta.json"),
            pcd_path=self.pcd_path,
            remove_pcd_color=False,
            resolution=8,
            max_sphere_distance=0.01,
            eval_mode=False,
            pcd_start_opacity=1.0,
        )

        return slam_config

    def main(self):
        
        save_dir = Path(self.output_path, "visibility")
        slam_config = self._create_slam_config()
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
    
if __name__ == "__main__":
    VisibilityRunConfig(
        input_path=Path("/data/pixel_lv2_vaibi/raw"),
        output_path=Path("/data/pixel_lv2_vaibi/undistorted"),
        pcd_path=Path("/data/pixel_lv2_vaibi/pixel_lvl1_water2_resampled2.ply")
    ).main()



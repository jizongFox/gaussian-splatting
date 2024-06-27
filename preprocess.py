import os
import rich
import shutil
from loguru import logger
from pathlib import Path

from dctoolbox.create_subset import SubregionConfig
from dctoolbox.dpt_depth import run as run_depth
from dctoolbox.mask_generator.make_head_mask import HeadMaskGeneratorConfig
from dctoolbox.mask_generator.merge_masks import MergeMasksConfig
from dctoolbox.mask_generator.yolo_detection import YoloMaskGeneratorConfig
from dctoolbox.process_pcd import ProcessPCDConfig
from dctoolbox.run_colmap import ColmapRunnerWithPointTriangulation
from dctoolbox.slam_interface_convertor import InterfaceAdaptorConfig
from dctoolbox.undistort_image import undistort_folder
from visibility_run import VisibilityRunConfig


def untar(file: Path, output_dir: Path):
    assert file.exists(), file
    output_dir.mkdir(parents=True, exist_ok=True)
    code = os.system(f"tar -xf {file} -C {output_dir}")
    if code != 0:
        raise RuntimeError(f"Failed to untar {file}")


def process_main(dataset_dir: Path, output_dir: Path, run_colmap: bool = False):
    r"""
    at the end of the program, you should have this structure
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
        ├── pixel_lvl1_water2_resampled2.ply
        └── visibility
    """

    # 0. untar dataset
    logger.info("Untaring dataset")
    untar(dataset_dir, output_dir / "raw")

    # 1. convert interface
    logger.info("Converting interface")
    InterfaceAdaptorConfig(
        slam_json_path=output_dir / "raw" / "slamMeta.json",
        output_path=output_dir / "undistorted" / "meta.json",
    ).main()

    # 2. undistorted images given key frame indicator.
    logger.info("Undistorting images")
    undistort_folder(
        input_dir=output_dir / "raw",
        output_dir=output_dir / "undistorted" / "images",
        image_extension="jpeg",
        converted_meta_json_path=output_dir / "undistorted" / "meta.json",
    )

    # 3. downsample pcd.
    logger.info("Downsampling pcd")
    dcloud_files = list((output_dir / "raw").glob("*.dcloud"))
    if len(dcloud_files) != 1:
        raise ValueError(
            "There should be only one .dcloud file in the dataset directory"
        )

    dcloud_file = dcloud_files[0]
    output_ply_path = output_dir / "undistorted" / f"{dcloud_file.stem}.ply"

    ProcessPCDConfig(
        input_path=dcloud_file,
        output_path=output_ply_path,
        voxel_size=None,
        convert_to_opencv=True,
    ).main()

    # 4. Run Visibility Check
    logger.info("Running visibility check")
    visibility_path = VisibilityRunConfig(
        image_dir=output_dir / "undistorted" / "images",
        meta_file=output_dir / "undistorted" / "meta.json",
        pcd_path=output_ply_path,
        save_dir=output_dir / "undistorted" / "visibility",
        image_extension=".jpeg",
        alpha_threshold=0.8,
    ).main()

    # 5. Create Subregion
    SubregionConfig(
        image_dir=output_dir / "undistorted" / "images",
        visibility_json=visibility_path,
        output_dir=output_dir / "subregion",
    ).main()

    # 6. generate yolo mask
    YoloMaskGeneratorConfig(
        image_dir=output_dir / "subregion" / "images",
        mask_dir=output_dir / "subregion" / "mask_yolo",
        extension=".jpeg",
    ).main()
    # 7. generate head mask
    HeadMaskGeneratorConfig(
        image_dir=output_dir / "subregion" / "images",
        mask_dir=output_dir / "subregion" / "mask_head",
        extension=".jpeg",
    ).main()

    # 8. merge mask
    MergeMasksConfig(
        mask_dirs=[
            output_dir / "subregion" / "mask_yolo",
            output_dir / "subregion" / "mask_head",
        ],
        output_dir=output_dir / "subregion" / "masks",
    ).main()

    shutil.rmtree(output_dir / "subregion" / "mask_yolo")
    shutil.rmtree(output_dir / "subregion" / "mask_head")

    # 9. Get Depth
    logger.info("Running depth model")
    run_depth(
        input_dir=output_dir / "subregion" / "images",
        output_dir=output_dir / "subregion" / "depths",
    )

    # 10. optional, run colmap
    if run_colmap:
        logger.info("Running colmap, this can take a while")
        colmap_config = ColmapRunnerWithPointTriangulation(
            data_dir=output_dir / "subregion",
            image_folder_name="images",
            mask_folder_name="masks",
            experiment_name="colmap",
            matching_type="vocab_tree",
            prior_injection=True,
            image_extension="jpeg",
            rig_bundle_adjustment=True,
            refinement_time=1,
            meta_file=output_dir / "undistorted" / "meta.json",
        )
        rich.print(colmap_config)
        colmap_config.main()


if __name__ == "__main__":
    process_main(
        Path("/home/jizong/Workspace/dConstruct/data/pixel_lvl1_water2_resampled.tar"),
        Path("/home/jizong/Workspace/dConstruct/data/2024-06-25"),
        run_colmap=True,
    )

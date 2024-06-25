import numpy as np
import open3d as o3d
import os
import shutil
from loguru import logger
from pathlib import Path

from dctoolbox.create_subset import SubregionConfig
from dctoolbox.dpt_depth import run as run_depth
from dctoolbox.mask_generator.make_head_mask import HeadMaskGeneratorConfig
from dctoolbox.mask_generator.merge_masks import MergeMasksConfig
from dctoolbox.mask_generator.yolo_detection import YoloMaskGeneratorConfig
from dctoolbox.process_pcd import ProcessPCDConfig
from dctoolbox.slam_interface_convertor import InterfaceAdaptorConfig
from dctoolbox.undistort_image import undistort_folder
from visibility_run import VisibilityRunConfig


def untar(file: Path, output_dir: Path):
    assert file.exists(), file
    output_dir.mkdir(parents=True, exist_ok=True)
    os.system(f"tar -xf {file} -C {output_dir}")


def downsample_pcd(pcd_path: Path, output_path: Path, downsample_rate: int):
    pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
    points = np.array(pcd.points)[::downsample_rate]
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array(pcd.colors)[::downsample_rate]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path.as_posix(), pcd)


def main(dataset_dir: Path, output_dir: Path):
    # 0. untar
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

    # 6. downsample pcd.
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

    # 7. Run Visibility Check
    logger.info("Running visibility check")
    visibility_path = VisibilityRunConfig(
        image_dir=output_dir / "undistorted" / "images",
        meta_file=output_dir / "undistorted" / "meta.json",
        pcd_path=output_ply_path,
        save_dir=output_dir / "undistorted" / "visibility",
        image_extension=".jpeg",
        alpha_threshold=0.8,
    ).main()

    # 8. Create Subregion
    SubregionConfig(
        image_dir=output_dir / "undistorted" / "images",
        visibility_json=visibility_path,
        output_dir=output_dir / "subregion",
    ).main()

    # 3. generate yolo mask
    YoloMaskGeneratorConfig(
        image_dir=output_dir / "subregion" / "images",
        mask_dir=output_dir / "subregion" / "mask_yolo",
        extension=".jpeg",
    ).main()
    # 4. generate head mask
    HeadMaskGeneratorConfig(
        image_dir=output_dir / "subregion" / "images",
        mask_dir=output_dir / "subregion" / "mask_head",
        extension=".jpeg",
    ).main()

    # 4. merge mask
    MergeMasksConfig(
        mask_dirs=[
            output_dir / "subregion" / "mask_yolo",
            output_dir / "subregion" / "mask_head",
        ],
        output_dir=output_dir / "subregion" / "masks",
    ).main()

    shutil.rmtree(output_dir / "subregion" / "mask_yolo")
    shutil.rmtree(output_dir / "subregion" / "mask_head")

    # 5. Get Depth
    run_depth(
        input_dir=output_dir / "subregion" / "images",
        output_dir=output_dir / "subregion" / "depths",
    )


main(
    Path("/home/jizong/Workspace/dConstruct/data/pixel_lvl1_water2_resampled.tar"),
    Path("/home/jizong/Workspace/dConstruct/data/2024-06-25"),
)

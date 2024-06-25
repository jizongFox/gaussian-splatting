import numpy as np
import open3d as o3d
import shutil
from loguru import logger
from pathlib import Path

from dctoolbox.dpt_depth import run as run_depth
from dctoolbox.mask_generator.make_head_mask import HeadMaskGeneratorConfig
from dctoolbox.mask_generator.merge_masks import MergeMasksConfig
from dctoolbox.mask_generator.yolo_detection import YoloMaskGeneratorConfig
from dctoolbox.slam_interface_convertor import InterfaceAdaptorConfig
from dctoolbox.undistort_image import undistort_folder
from dctoolbox.process_pcd import ProcessPCDConfig
from dctoolbox.create_subset import SubregionConfig
from visibility_run import VisibilityRunConfig
from utils.system_utils import get_hash


def downsample_pcd(pcd_path: Path, output_path: Path, downsample_rate: int):
    pcd = o3d.io.read_point_cloud(pcd_path.as_posix())
    points = np.array(pcd.points)[::downsample_rate]
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.array(pcd.colors)[::downsample_rate]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(output_path.as_posix(), pcd)


def main(dataset_dir: Path, output_dir: Path):
    # 1. convert interface
    logger.info("Converting interface")
    InterfaceAdaptorConfig(
        slam_json_path=dataset_dir / "slamMeta.json",
        output_path=output_dir / "meta.json",
    ).main()

    # 2. undistorted images given key frame indicator.
    logger.info("Undistorting images")
    undistort_folder(input_dir=dataset_dir, output_dir=output_dir / "images", image_extension="jpeg",
                     converted_meta_json_path=output_dir / "meta.json")

    # 3. generate yolo mask
    YoloMaskGeneratorConfig(image_dir=output_dir / "images", mask_dir=output_dir / "mask_yolo",
                            extension=".jpeg").main()
    # 4. generate head mask
    HeadMaskGeneratorConfig(image_dir=output_dir / "images", mask_dir=output_dir / "mask_head",
                            extension=".jpeg").main()

    # 4. merge mask
    MergeMasksConfig(mask_dirs=[output_dir / "mask_yolo", output_dir / "mask_head"],
                     output_dir=output_dir / "masks").main()

    shutil.rmtree(output_dir / "mask_yolo")
    shutil.rmtree(output_dir / "mask_head")

    # 5. Get Depth
    run_depth(input_dir=output_dir / "images", output_dir=output_dir / "depths")

    # 6. downsample pcd.
    dcloud_files = list(dataset_dir.glob("*.dcloud"))
    if len(dcloud_files) != 1:
        raise ValueError("There should be only one .dcloud file in the dataset directory")
    
    dcloud_file = dcloud_files[0]
    output_ply_path = output_dir.parent / f"{dcloud_file.stem}.ply"
    
    ProcessPCDConfig(
        input_path=Path(dcloud_file),
        output_path=Path(output_ply_path),
        voxel_size=0.01, convert_to_opencv=True
    ).main()

    # 7. Run Visibility Check
    VisibilityRunConfig(input_path=dataset_dir, 
                        output_path=output_dir, 
                        pcd_path=output_ply_path).main()
    
    # 8. Create Subregion
    SubregionConfig(input_path=dataset_dir, 
                    output_path=output_dir, 
                    git_hash=get_hash()).main()


main(Path("/data/pixel_lv2_vaibi/raw/"), Path("/data/pixel_lv2_vaibi/undistorted"))



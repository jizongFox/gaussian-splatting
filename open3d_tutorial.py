# this file is to investigate the pytorch3d package.
import json
import numpy as np
import open3d as o3d
from pathlib import Path
from pytorch3d.vis import plot_scene

from dctoolbox.utils import quat2rotation

# S from opencv to slam convention
S = np.array([[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)


def load_meta_poses(meta_json: Path | str):
    with open(meta_json) as f:
        data = json.load(f)

    data = data["data"]
    image_poses = {}

    for i, d in enumerate(data):
        image_path_ = d["imgName"]
        for k, v in image_path_.items():
            px = d["worldTcam"][k]["px"]
            py = d["worldTcam"][k]["py"]
            pz = d["worldTcam"][k]["pz"]
            qw = d["worldTcam"][k]["qw"]
            qx = d["worldTcam"][k]["qx"]
            qy = d["worldTcam"][k]["qy"]
            qz = d["worldTcam"][k]["qz"]
            R = np.zeros((4, 4))
            qvec = np.array([qw, qx, qy, qz])
            norm = np.linalg.norm(qvec)
            qvec /= norm
            R[:3, :3] = quat2rotation(torch.from_numpy(qvec).float()[None, ...])[0]
            R[:3, 3] = np.array([px, py, pz])
            R[3, 3] = 1.0
            # assert np.linalg.det(R[:3, :3]) == 1
            # todo: check if normalized. If not, normalize it.
            # Q, T = SE3_to_quaternion_and_translation_torch(torch.from_numpy(R).unsqueeze(0).double())
            # assert torch.allclose(Q.float(), torch.from_numpy(qvec).float(), rtol=1e-3, atol=1e-3), (
            #     Q.float(), torch.from_numpy(qvec).float())
            # assert torch.allclose(T.float(), torch.from_numpy([px, py, pz]).float(), rtol=1e-3, atol=1e-3)
            # here the world coordinate is defined in robotics space, where z is up, x is left and y is right.
            # the camera coordinate is defined in opencv convention, where the camera is looking down the z axis,
            # y is down and x is right.

            # convert the world coordinate to camera coordinate.
            R = S.T.dot(R)  # this is the c2w in opencv convention.

            image_poses[v] = R

    return image_poses


def from_opencv2slam(points: np.ndarray) -> np.ndarray:

    return np.einsum("ij,bj->bi", S, points)


def from_slam2opencv(points: np.ndarray) -> np.ndarray:
    return np.einsum("ij,bj->bi", S.T, points)


import torch
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

DATA_DIR = "./data"
obj_filename = "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside.ply"
device = torch.device("cuda:0")
# Load point cloud
pointcloud = o3d.io.read_point_cloud(obj_filename)
verts = torch.from_numpy(np.array(pointcloud.points).astype(np.float32)).to(device)

rgb = torch.from_numpy(np.array(pointcloud.colors).astype(np.float32)).to(device)

point_cloud = Pointclouds(points=[verts], features=[rgb])


R, T = look_at_view_transform(
    5,
    10,
    10,
)
cameras = PerspectiveCameras(
    device=device,
    R=R,
    T=T,
)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
# and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters.
raster_settings = PointsRasterizationSettings(
    image_size=512, radius=0.07, points_per_pixel=1
)


# Create a points renderer by compositing points using an alpha compositor (nearer points
# are weighted more heavily). See [1] for an explanation.
rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())

images = renderer(point_cloud)
plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()

fig = plot_scene({"Pointcloud": {"person": point_cloud}})
fig.show()

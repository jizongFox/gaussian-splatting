import json
import math
import matplotlib.pyplot as plt
import numpy as np
import open3d
import torch
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

from nerfstudio.scripts.dctoolbox.utils import quat2rotation

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
    S = np.array(
        [[-1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
    )

    return np.einsum("ij,bj->bi", S.T, points)


def clip_near_far(
    camera_coordinates: np.ndarray, near: float, far: float
) -> np.ndarray:
    assert len(camera_coordinates.shape) == 2, camera_coordinates.shape
    if camera_coordinates.shape[1] == 4:
        assert np.allclose(
            camera_coordinates[:, 3], np.ones_like(camera_coordinates[:, 3])
        )
    z = camera_coordinates[:, 2]
    return camera_coordinates[(z >= near) & (z <= far)]


class Intrinsic:
    def __init__(
        self,
        *,
        w: int,
        h: int,
        cx: float | None = None,
        cy: float | None = None,
        fov_degree: float,
        near: float,
        far: float,
    ):
        self.w = w
        self.h = h
        self.__cx = cx
        self.__cy = cy
        self.fov_degree = fov_degree
        self.near = near
        self.far = far

    @property
    def fovy(self):
        """make degrees to radians."""
        return self.fov_degree / 360.0 * 2.0 * np.pi  # 45° in radians

    @property
    def f(self) -> float:
        return 0.5 * self.h / math.tan(self.fovy / 2)

    @property
    def cx(self) -> float:
        if self.__cx is None:
            return self.w / 2
        return self.__cx

    @cx.setter
    def cx(self, value):
        self.__cx = value

    @property
    def cy(self):
        if self.__cy is None:
            return self.h / 2
        return self.__cy

    @property
    def pinhole_camera_matrix(self) -> np.ndarray:
        """
        pinhole camera matrix under opencv convention.
        """
        camera_mtx = np.array(
            [[self.f, 0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        full_camera_mtx = np.hstack([camera_mtx, np.array([0, 0, 0])[..., None]])
        return full_camera_mtx

    @property
    def opengl_projection_matrix(self) -> np.ndarray:
        """opengl projection matrix under opengl convention."""
        opengl_projection_mtx = np.array(
            [
                [2 * self.f / self.w, 0.0, (self.w - 2 * self.cx) / self.w, 0.0],
                [
                    0.0,
                    2 * self.f / self.h,
                    (self.h - 2 * (self.h - self.cy))
                    / self.h,  # convention changes for cx and cy
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    (-self.far - self.near) / (self.far - self.near),
                    -2.0 * self.far * self.near / (self.far - self.near),
                ],
                [0.0, 0.0, -1.0, 0.0],
            ]
        )
        return opengl_projection_mtx

    @staticmethod
    def to_opengl_coordinate(opencv_coordinate: np.ndarray) -> np.ndarray:
        """
        to convert opencv coordinate to opengl coordinate
        """
        # OpenGL projection
        cv2gl = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        return np.einsum("ij,bj->bi", cv2gl, opencv_coordinate)


@dataclass
class Camera2World:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        assert self.R.shape == (3, 3), self.R.shape
        assert self.t.squeeze().shape == (3,), self.t.shape
        self.t = self.t.squeeze()

    @property
    def extrinsic(self) -> np.ndarray:
        full_matrix = np.zeros((4, 4), dtype=float)
        full_matrix[:3, :3] = self.R
        full_matrix[:3, 3] = self.t
        full_matrix[3, 3] = 1.0
        return full_matrix

    def to_world2camera(self) -> "World2Camera":
        extrinsic = self.extrinsic
        extrinsic_inv = np.linalg.inv(extrinsic)
        return World2Camera(extrinsic_inv[:3, :3], extrinsic_inv[:3, 3])

    @property
    def camera_center(self) -> np.ndarray:
        return self.extrinsic[:3, 3]


@dataclass
class World2Camera:
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        assert self.R.shape == (3, 3), self.R.shape
        assert self.t.squeeze().shape == (3,), self.t.shape
        self.t = self.t.squeeze()

    @property
    def extrinsic(self) -> np.ndarray:
        full_matrix = np.zeros((4, 4), dtype=float)
        full_matrix[:3, :3] = self.R
        full_matrix[:3, 3] = self.t
        full_matrix[3, 3] = 1.0
        return full_matrix

    def to_camera2world(self) -> "Camera2World":
        extrinsic = self.extrinsic
        extrinsic_inv = np.linalg.inv(extrinsic)
        return Camera2World(extrinsic_inv[:3, :3], extrinsic_inv[:3, 3])

    @property
    def camera_center(self) -> np.ndarray:
        return self.to_camera2world().camera_center


def plot3D_pointcloud(pcd: np.ndarray):
    import plotly.express as px

    pcd = pcd[::20]
    batch_size = pcd.shape[0]
    data = {
        "x": pcd[:, 0],
        "y": pcd[:, 1],
        "z": pcd[:, 2],
        "size": np.ones_like(pcd[:, 0]) * 1,
        "c": ["red"] * batch_size,
    }
    fig = px.scatter_3d(data, x="x", y="y", z="z", size="size", color="c")
    fig.update_layout(scene=dict(aspectmode="data"))
    fig.show()


def load_pointcloud(filename: str) -> np.ndarray:
    pointcloud = open3d.io.read_point_cloud(filename)

    pointcloud_np = np.array(pointcloud.points)
    pointcloud_np = pointcloud_np[np.linalg.norm(pointcloud_np, axis=-1) < 200]
    return np.hstack(
        [
            pointcloud_np,
            np.array([1])[None, ...].repeat(
                pointcloud_np.shape[0],
                axis=0,
            ),
        ]
    )


# we compute the corresponding opengl projection matrix
# cf https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
# NOTE: K00 = K11 = f, K10 = 0.0, K02 = cx, K12 = cy, K22 = 1.0


# point is in opencv camera space (along Oz axis)
# point = np.array([1.0, 2.0, 15.0])  # Note: coords must be floats

# OpenCV projection
# screen_point, _ = cv2.projectPoints(
#     np.array([point]), np.zeros(3), np.zeros(3), camera_mtx, np.zeros(5)
# )
# logger.info("OpenCV projection")
# logger.info(screen_point)

# Note: we obtain the same result with this:
# (that's what cv2.projectPoints basically does: multiply points with camera matrix and then divide result by z coord)
cloud_name = "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/korea_accoms_outside.ply"
meta_file = "/home/jizong/Workspace/dConstruct/data/bundleAdjustment_korea_scene2/meta_updated.json"

logger.info("Also computed from matrix multiplication")
pointcloud_np_homo = load_pointcloud(filename=cloud_name)

pointcloud_np_homo = from_slam2opencv(pointcloud_np_homo)
# plot3D_pointcloud(pointcloud_np_homo)

poses = load_meta_poses(meta_file)

camera_poses = np.stack([x[:3, 3] for x in poses.values()])
#
# pcd = pointcloud_np_homo[::20]
# batch_size = pcd.shape[0]
# data = {
#     "x": pcd[:, 0],
#     "y": pcd[:, 1],
#     "z": pcd[:, 2],
#     "size": np.ones_like(pcd[:, 0]) * 1,
#     "c": ["red"] * batch_size,
# }
# fig = px.scatter_3d(pd.DataFrame(data), x="x", y="y", z="z", size="size", color="c")
# fig.update_layout(scene=dict(aspectmode="data"))
# fig2 = px.scatter_3d(pd.DataFrame(camera_poses, columns=[*"xyz"]), x="x", y="y", z="z")
# fig2.update_layout(scene=dict(aspectmode="data"))
# import plotly.graph_objects as go
#
# big_fig = go.Figure([fig.data[0], fig2.data[0]])
# big_fig.show()
poses = load_meta_poses(meta_file)
camera = Intrinsic(h=800, w=800, near=1, far=500, fov_degree=90, cx=500, cy=500)

cur_pose = poses["img_DECXIN2023012347_1117.png"]
c2w = Camera2World(R=cur_pose[:3, :3], t=cur_pose[:3, 3])
print(c2w)

xyz_camera = np.einsum("ij,bj->bi", c2w.to_world2camera().extrinsic, pointcloud_np_homo)
xyz_camera = clip_near_far(xyz_camera, near=camera.near, far=camera.far)
plot3D_pointcloud(xyz_camera)
pinhole_uvw = np.einsum("ij,nj->ni", camera.pinhole_camera_matrix, xyz_camera)

pinhole_uv = pinhole_uvw[:, :2] / pinhole_uvw[:, -1:]
pinhole_uv = pinhole_uv[
    (pinhole_uv[:, 0] < camera.w)
    & (pinhole_uv[:, 0] >= 0)
    & (pinhole_uv[:, 1] < camera.h)
    & (pinhole_uv[:, 1] >= 0)
]

pinhole_uv = pinhole_uv.astype(np.uint32)

screen_pinhole = np.zeros((camera.h, camera.w), dtype=np.uint8)
for cur_w, cur_h in pinhole_uv:
    screen_pinhole[cur_h, cur_w] = +1
    screen_pinhole[min(cur_h + 1, camera.h - 1), cur_w] = +1
    screen_pinhole[cur_h, min(cur_w + 1, camera.w - 1)] = +1
    screen_pinhole[min(cur_h + 1, camera.h - 1), min(cur_w + 1, camera.w - 1)] = +1

plt.imshow(screen_pinhole)
plt.show(block=False)

# OpenGL projection
xyz_camera_opengl = camera.to_opengl_coordinate(xyz_camera)

# # we flip the point z coord, because in opengl camera is oriented along -Oz axis
# point[2] = -point[2]
# point2 = np.hstack(
#     [point, 1.0]
# )  # we add vertex w coord (usually done in vertex shader before multiplying by projection matrix)

# we get the point in clip space
# clip_point = opengl_mtx.dot(point2)
clip_point = np.einsum("ij,bj->bi", camera.opengl_projection_matrix, xyz_camera_opengl)
# NOTE: what follows "simulates" what happens in OpenGL after the vertex shader.
# This is necessary so that we can make sure our projection matrix will yield the correct result when used in OpenGL
# we get the point in NDC
ndc_point = clip_point / clip_point[:, -1:]
ndc_point = ndc_point[
    (ndc_point[:, 0] >= -1)
    & (ndc_point[:, 0] <= 1)
    & (ndc_point[:, 1] >= -1)
    & (ndc_point[:, 1] <= 1)
    & (ndc_point[:, 2] >= -1)
    & (ndc_point[:, 2] <= 1)
]

plot3D_pointcloud(ndc_point)
# we get the screen coordinates
viewport_point = (ndc_point + 1.0) / 2.0 * np.array([camera.w, camera.h, 1.0, 1.0])

# opencv Oy convention is opposite of OpenGL so we reverse y coord
viewport_point[:, 1] = camera.h - viewport_point[:, 1]

viewport_point = viewport_point.astype(np.uint32)

screen_opengl = np.zeros((camera.h, camera.w), dtype=np.uint8)
for cur_w, cur_h, *_ in viewport_point:
    screen_opengl[cur_h, cur_w] = +1
    screen_opengl[min(cur_h + 1, camera.h - 1), cur_w] = +1
    screen_opengl[cur_h, min(cur_w + 1, camera.w - 1)] = +1
    screen_opengl[min(cur_h + 1, camera.h - 1), min(cur_w + 1, camera.w - 1)] = +1

plt.figure()
plt.imshow(screen_opengl)
# assert np.allclose(screen_opengl, screen_pinhole)
plt.figure()
plt.imshow(
    np.abs((screen_opengl > 0).astype(float) - (screen_pinhole > 0).astype(float))
)
plt.title("diff")
plt.show(block=True)

# Now you can see that viewport_point and screen_point have the same x/y coordinates!
# This means you can now, from OpenCv camera matrix, use OpenGl to render stuff on top of the image,
# thanks to the opengl projection matrix, computed from opencv camera matrix


# NOTE: when near plane is small (a few units) and when focal length is small (ex: 10-12),
# both results tend to diverge. I'm not sure why the formula starts falling apart at extreme values.

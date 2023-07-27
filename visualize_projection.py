import math
import matplotlib.pyplot as plt
import numpy as np
import open3d
from dataclasses import dataclass
from loguru import logger

cloud_name = "/home/jizong/Workspace/nerfstudio/data/flower_undist3/sparse_points.ply"


def clip_near_far(camera_coordinates: np.ndarray, near: float, far: float) -> np.ndarray:
    assert len(camera_coordinates.shape) == 2, camera_coordinates.shape
    if camera_coordinates.shape[1] == 4:
        assert np.allclose(camera_coordinates[:, 3], np.ones_like(camera_coordinates[:, 3]))
    z = camera_coordinates[:, 2]
    return camera_coordinates[(z >= near) & (z <= far)]


@dataclass
class Intrinsic:
    w: int
    h: int
    fov_degree: float = 90

    near: float = 0.5  # near plane
    far: float = 0.9  # far plane

    @property
    def fovy(self):
        return self.fov_degree / 360.0 * 2.0 * np.pi  # 45° in radians

    @property
    def f(self) -> float:
        return 0.5 * self.h / math.tan(self.fovy / 2)

    @property
    def cx(self):
        return self.w / 2

    @property
    def cy(self):
        return self.h / 2

    @property
    def pinhole_camera_matrix(self) -> np.ndarray:
        """
        pinhole camera matrix under opencv convention.
        """
        camera_mtx = np.array([[self.f, 0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        full_camera_mtx = np.hstack([camera_mtx, np.array([0, 0, 0])[..., None]])
        return full_camera_mtx

    @property
    def opengl_projection_matrix(self) -> np.ndarray:
        """opengl projection matrix under opengl convention."""
        opengl_projection_mtx = np.array(
            [
                [2 * self.f / self.w, 0.0, (self.w - 2 * self.cx) / self.w, 0.0],
                [0.0, 2 * self.f / self.h, (self.h - 2 * self.cy) / self.h, 0.0],
                [0.0, 0.0, (-self.far - self.near) / (self.far - self.near),
                 -2.0 * self.far * self.near / (self.far - self.near)],
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
    data = {"x": pcd[:, 0], "y": pcd[:, 1], "z": pcd[:, 2], "size": np.ones_like(pcd[:, 0]) * 1,
            "c": ["red"] * batch_size}
    fig = px.scatter_3d(data, x="x", y="y", z="z", size="size", color="c")
    fig.show()


def load_pointcloud(filename: str) -> np.ndarray:
    pointcloud = open3d.io.read_point_cloud(filename)

    pointcloud_np = np.array(pointcloud.points)
    pointcloud_np = pointcloud_np[np.linalg.norm(pointcloud_np, axis=-1) < 2]
    pointcloud_np_homo = np.hstack(
        [pointcloud_np, np.array([1])[None, ...].repeat(pointcloud_np.shape[0], axis=0, )]
    )
    return pointcloud_np_homo


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
logger.info("Also computed from matrix multiplication")
pointcloud_np_homo = load_pointcloud(filename=cloud_name)
camera = Intrinsic(h=800, w=800, near=1, far=5, fov_degree=90)
c2w = Camera2World(R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), t=np.array([0, 0, -2]))

xyz_camera = np.einsum("ij,bj->bi", c2w.to_world2camera().extrinsic, pointcloud_np_homo)
xyz_camera = clip_near_far(xyz_camera, near=camera.near, far=camera.far)
plot3D_pointcloud(xyz_camera)

pinhole_uvw = np.einsum("ij,nj->ni", camera.pinhole_camera_matrix, xyz_camera)

pinhole_uv = pinhole_uvw[:, :2] / pinhole_uvw[:, -1:]
pinhole_uv = pinhole_uv[
    (pinhole_uv[:, 0] < camera.w) & (pinhole_uv[:, 0] >= 0) & (pinhole_uv[:, 1] < camera.h) & (pinhole_uv[:, 1] >= 0)]

pinhole_uv = pinhole_uv.astype(np.uint32)

screen_pinhole = np.zeros((camera.h, camera.w), dtype=np.uint8)
for (cur_w, cur_h) in pinhole_uv:
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
    (ndc_point[:, 0] >= -1) & (ndc_point[:, 0] <= 1) & (ndc_point[:, 1] >= -1) & (ndc_point[:, 1] <= 1) & \
    (ndc_point[:, 2] >= -1) & (ndc_point[:, 2] <= 1)]

plot3D_pointcloud(ndc_point)
# we get the screen coordinates
viewport_point = (ndc_point + 1.0) / 2.0 * np.array([camera.w, camera.h, 1.0, 1.0])

# opencv Oy convention is opposite of OpenGL so we reverse y coord
viewport_point[:, 1] = camera.h - viewport_point[:, 1]

viewport_point = viewport_point.astype(np.uint32)

screen_opengl = np.zeros((camera.h, camera.w), dtype=np.uint8)
for (cur_w, cur_h, *_) in viewport_point:
    screen_opengl[cur_h, cur_w] = +1
    screen_opengl[min(cur_h + 1, camera.h - 1), cur_w] = +1
    screen_opengl[cur_h, min(cur_w + 1, camera.w - 1)] = +1
    screen_opengl[min(cur_h + 1, camera.h - 1), min(cur_w + 1, camera.w - 1)] = +1

plt.figure()
plt.imshow(screen_opengl)
# assert np.allclose(screen_opengl, screen_pinhole)
plt.figure()
plt.imshow(np.abs((screen_opengl > 0).astype(float) - (screen_pinhole > 0).astype(float)))
plt.title("diff")
plt.show(block=True)

# Now you can see that viewport_point and screen_point have the same x/y coordinates!
# This means you can now, from OpenCv camera matrix, use OpenGl to render stuff on top of the image,
# thanks to the opengl projection matrix, computed from opencv camera matrix


# NOTE: when near plane is small (a few units) and when focal length is small (ex: 10-12),
# both results tend to diverge. I'm not sure why the formula starts falling apart at extreme values.

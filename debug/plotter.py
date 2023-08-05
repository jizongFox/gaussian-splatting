import plotly.express as px
from torch import Tensor


def plot_3d_scatter(xyz):
    if isinstance(xyz, Tensor):
        xyz = xyz.detach().cpu()

    xyz = xyz[::10]

    data = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    fig = px.scatter_3d(data, x="x", y="y", z="z")
    fig.show()

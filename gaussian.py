import numpy as np
from torch import Tensor
import plotly.express as px

cov = np.array([[1, 0.5], [0.5, 1]])


def plot_3d_scatter(xyz):
    if isinstance(xyz, Tensor):
        xyz = xyz.detach().cpu()
    xyz = xyz[::10]
    data = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    fig = px.scatter_3d(data, x="x", y="y", z="z")
    fig.show()


def get_gaussian_2d(mean, cov):
    xs = np.linspace(-3, 3, 100)
    ys = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(
        xs,
        ys,
    )
    xy = np.stack([x.flatten(), y.flatten()], axis=-1)

    full_matrix: np.ndarray = np.exp(
        -0.5 * (xy - mean) @ np.linalg.inv(cov) @ (xy - mean).transpose(-1, -2)
    )

    return x.flatten(), y.flatten(), full_matrix.diagonal()


def pdf_multivariate_gauss(x, mu, cov):
    """
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    """
    assert mu.shape[0] > mu.shape[1], "mu must be a row vector"
    assert x.shape[0] > x.shape[1], "x must be a row vector"
    assert cov.shape[0] == cov.shape[1], "covariance matrix must be square"
    assert (
        mu.shape[0] == cov.shape[0]
    ), "cov_mat and mu_vec must have the same dimensions"
    assert mu.shape[0] == x.shape[0], "mu and x must have the same dimensions"
    part1 = 1 / (((2 * np.pi) ** (len(mu) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
    part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
    return float(part1 * np.exp(part2))


def test_gauss_pdf():
    x = np.array([[0], [0]])
    mu = np.array([[0], [0]])
    cov = np.eye(2)

    print(pdf_multivariate_gauss(x, mu, cov))


if __name__ == "__main__":
    mean = np.array([0, 0])
    cov = cov
    x, y, z = get_gaussian_2d(mean, cov)
    plot_3d_scatter(np.stack([x, y, z], axis=-1))
    test_gauss_pdf()
    # points = get_gaussian_2d(mean, cov)
    # breakpoint()

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def plot_point_clouds(pcs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for pc in pcs:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    plt.show()


def plot_planes(planes: npt.NDArray):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for plane in planes:
        x = np.linspace(-10, 10, 10)
        y = np.linspace(-10, 10, 10)
        x, y = np.meshgrid(x, y)
        z = (-plane[0] * x - plane[1] * y - plane[3]) / plane[2]
        ax.plot_surface(x, y, z)
    plt.show()


def generate_planar_cloud(
        points_number: int,
        plane_coefficients: tuple,
        voxel_corner: np.ndarray,
        edge_length: float,
        sigma: float,
        outlier_ratio: float,
):
    voxel_points = (
            np.random.rand(points_number, 3) * np.array([edge_length - 6 * sigma] * 3)
            + voxel_corner
            + 3 * sigma
    )
    noise = np.random.normal(0, sigma, (points_number,))
    plane_points_z = (
                             -plane_coefficients[0] * voxel_points[:, 0]
                             - plane_coefficients[1] * voxel_points[:, 1]
                             - plane_coefficients[3]
                     ) / plane_coefficients[2]
    noisy_plane_points_z = plane_points_z + noise
    oulier_noise = np.random.normal(0, 10 * sigma, (int(points_number * outlier_ratio),))
    noisy_plane_points_z[: int(points_number * outlier_ratio)] += oulier_noise
    return np.column_stack((voxel_points[:, :2], noisy_plane_points_z))

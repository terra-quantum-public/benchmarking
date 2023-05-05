import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def create_nshells(ndimen, npoints, scale, noise):
    """
    Sample ``npoints`` from an ``ndimen`` dimensional noisy shell of radius ``scale``

    Args
    ____
    ndimen : int
        the dimension of the Euclidean space in which to generate shell
    npoints : int
        the number of points in the shell
    scale : float
        the average radius of the shell
    noise : float
        the level of normally distributed noise to apply to the shell

    Returns
    _______
    points : np.ndarray
        a numpy array with shape ``(npoints, ndimen)`` containing the generated points
    """
    points = np.zeros((npoints, ndimen))
    for j in range(npoints):
        point = np.zeros(ndimen)
        norm = 0
        for i in range(ndimen):
            points[j, i] = np.random.normal(0, 1)
            norm += points[j, i] ** 2
        points[j] = points[j] * scale / norm ** 0.5 + np.random.normal(scale=noise, size=ndimen)
    return points


def nshells(ndimen, npoints, scale=1.0, noise=0.0):
    """
    Create two nested, normally distributed, n-dimensional shells and their labels.

    Args
    ____
    ndimen : int
        the dimension of the Euclidean space in which to generate shells
    npoints : int
        the number of points in each shell
    scale : float
        the average radius of the inner shell.
        Default: 1.0
    noise : float
        the level of normally distributed noise to apply to both shells.
        Default: 0.0

    Returns
    _______
    x : np.ndarray
        a numpy array with shape ``(npoints, ndimen)`` containing the generated points
    y : np.ndarray
        a numpy array with shape ``(npoints)`` containg the labels 1 and 0 for the inner and outer shells, respectively
    """
    x0 = create_nshells(ndimen, npoints, 1.0, noise)
    x1 = create_nshells(ndimen, npoints, scale, noise)
    y0, y1 = np.zeros(npoints), np.ones(npoints)
    x, y = np.vstack((x0, x1)), np.hstack((y0, y1))
    p = np.random.permutation(y.shape[0])
    return x[p], y[p]


def visualize_shells(npoints, noise, scale):
    """
    Visualize shells.

    Args
    ____
    npoints : int
        the number of points in each shell
    scale : float
        the average radius of the inner shell
    noise : float
        the level of normally distributed noise to apply to both shells
    """
    x, y = nshells(2, npoints, noise=noise, scale=scale)
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    ax.axis('equal')
    plt.scatter(x[:, 0], x[:, 1], c=y)

    x, y = nshells(3, npoints, noise=noise, scale=scale)
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
    plt.show()


def gen_sets(dims: int, npoints: int = 1, noise: float = 0.2,
             scale: float = 0.3, random_seed: int = 42, ):
    """
    Generates a dataset of 2 concentric hyper-spherical shells

    Args
    ____
    dims : int
        the number of dimensions of the space
    npoints : int
        number of data points to be generated.
        Default: 1
    noise : float
        the amount of random noise in dataset.
        Default: 0.2
    scale : float
        scale of the dataset.
        Default: 0.3
    random_seed : int
        random seed to fix the data generation.
        Default: 42

    Returns
    _______
    features : np.ndarray
        in the shape ``(npoints, dims)``
    labels : nd.ndarray
        in the shape ``(npoints, 1)`` indicating which class they belong to
    """
    val_cutoff = 1  # size of the evaluation dataset

    features, labels = nshells(dims, npoints, noise, scale)
    labels = labels.reshape(-1, 1)

    # taking only the first point
    return features[:val_cutoff], labels[:val_cutoff]
"""
Kernel functions
"""

# Authors: Junki Ishikawa

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel


def rbf_kernel(X, Y, sigma=None):
    """
    RBF kernel. this is a wrapper of sklearn.metrics.pairwise.rbf_kernel.
    K(x, y) = exp(- (1/2) * (||x - y||/sigma)^2)

    Parameters
    ----------
    X : array of shape (n_dims, n_samples_X)

    Y : array of shape (n_dims, n_samples_Y)

    sigma : float, default None
        If None, defaults to sqrt(n_dims / 2)

    Returns:
    --------
    K : array-like, shape: (n_samples_X, n_samples_Y)
        Grammian matrix
    """

    n_dims = X.shape[0]
    if sigma is None:
        sigma = np.sqrt(n_dims / 2)

    # subtraction, (n_dims, n_samples_X, n_samples_Y)
    x = X.reshape(n_dims, -1, 1) - Y.reshape(n_dims, 1, -1)
    # l2 distance, (n_samples_X, n_samples_Y)
    x = np.sum(x**2, axis=0)
    # gausiann kernel, (n_samples_X, n_samples_Y)
    x = np.exp(-0.5 * x / (sigma**2))
    return x


def linear_kernel(X, Y):
    """
    Linear kernel. this calculates simple inner product.
    K(x, y) = x @ y

    Parameters
    ----------
    X : array of shape (n_dims, n_samples_X)

    Y : array of shape (n_dims, n_samples_Y)

    Returns:
    --------
    K : array-like, shape: (n_samples_X, n_samples_Y)
        Grammian matrix
    """
    return X.T @ Y
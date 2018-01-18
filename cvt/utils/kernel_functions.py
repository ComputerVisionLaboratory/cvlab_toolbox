"""
Kernel functions
"""

# Authors: Junki Ishikawa

from sklearn.metrics.pairwise import rbf_kernel as _rbf_kernel


def rbf_kernel(X, Y, sigma=None):
    """
    RBF kernel. this is a wrapper of sklearn.metrics.pairwise.rbf_kernel.
    K(x, y) = exp(- (1/2) * (||x - y||/sigma)^2)

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    sigma : float, default None
        If None, defaults to sqrt(n_features / 2)

    Returns:
    --------
    K : array-like, shape: (n_samples_X, n_samples_Y)
        Grammian matrix
    """

    gamma = 1 / (2 * sigma**2) if sigma is not None else None
    return _rbf_kernel(X, Y, gamma=gamma)


def linear_kernel(X, Y):
    """
    Linear kernel. this calculates simple inner product.
    K(x, y) = x @ y

    Parameters
    ----------
    X : array of shape (n_samples_X, n_features)

    Y : array of shape (n_samples_Y, n_features)

    Returns:
    --------
    K : array-like, shape: (n_samples_X, n_samples_Y)
        Grammian matrix
    """
    return X @ Y.T
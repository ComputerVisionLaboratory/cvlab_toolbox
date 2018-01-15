"""
Mathematical utilities
"""

# Authors: Junki Ishikawa
#          Takahiro Inagaki

import numpy as np


def canonical_angle(X, Y):
    """
    Calculate cannonical angles beween subspaces

    Parameters
    ----------
    X: basis matrix, array-like, shape: (n_subdim_X, n_dim)
    Y: basis matrix, array-like, shape: (n_subdim_Y, n_dim)

    Returns
    -------
    c: float, similarity of X, Y

    """

    _, s, _ = np.linalg.svd(Y @ X.T)
    c = (s ** 2).mean()
    return c


def canonical_angle_matrix(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """

    n_set_X, n_set_Y = len(X), len(Y)
    C = np.zeros((n_set_X, n_set_Y))
    for x in range(n_set_X):
        for y in range(n_set_Y):
            C[x, y] = canonical_angle(X[x], Y[y])

    return C

# faster method
def canonical_angle_matrix_f(X, Y):
    """Calculate canonical angles between subspaces
    example     similarity = MathUtils.calc_basis_vector(X, Y)

    Parameters
    ----------
    X: set of basis matrix, array-like, shape: (n_set_X, n_subdim, n_dim)
        n_subdim can be variable on each subspaces
    Y: set of basis matrix, array-like, shape: (n_set_Y, n_subdim, n_dim)
        n_set can be variable from n_set of X
        n_subdim can be variable on each subspaces

    Returns:
        C: similarity matrix, array-like, shape: (n_set_X, n_set_Y)

    """
    X = np.transpose(X, (0, 2, 1))
    D = Y.dot(X)
    _D = np.transpose(D, (0, 2, 1, 3))
    _, C, _ = np.linalg.svd(_D)
    sim = C**2
    return sim.mean(2)


def subspace_bases(X, n_subdims):
    """
    Simple PCA and

    Parameters
    ----------
    X: array-like, shape (n_vectors, n_features)
        data matrix
    n_subdims: integer
        number of subspace dimension

    Returns
    -------
    V: array-like, shape (n_subdims, n_features)
        bases matrix
    """
    _, v = np.linalg.eigh(X.T @ X)
    v = v.T[::-1]
    V = v[:n_subdims]
    return V

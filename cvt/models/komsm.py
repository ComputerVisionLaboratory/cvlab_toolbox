"""
Kernel Orthogonal Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from ..utils import rbf_kernel, dual_vectors, mean_square_singular_values


class KernelOMSM(BaseEstimator, ClassifierMixin):
    """
    Discriminant analysis on Grassmann maniforld.

    Parameters
    ----------
    n_subdims : int, optional (default=3)
        A dimension of subspace. it must be smaller than the dimension of original space.

    normalize : boolean, optional (default=True)
        If this is True, all vectors are normalized as |v| = 1

    """

    def __init__(self, n_subdims=3, normalize=True, sigma=None, kernel_func=rbf_kernel):
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.sigma = sigma
        self.kernel_func = kernel_func
        self.le = LabelEncoder()
        self.train_X = None
        self.labels = None
        self.mappings = None
        self.W = None

    def get_params(self, deep=True):
        return {'n_subdims': self.n_subdims, 'sigma': self.sigma, 'kernel_func': self.kernel_func}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fit the model according to the given traininig data and parameters

        Parameters
        ----------
        X: array-like, shape = [n_samples, n_vectors, n_features]
            Training vectors, where n_samples is number of samples, n_vectors is number of vectors on each samples
            and n_features is number of features. Since n_vectors may be variable on each samples, X can be lists
            containing n_sample matrices: [array(n_vectors{1}, n_features),..., array(n_vectors{n_samples}, n_features)]

        y: integer array, shape = [n_samples]
            Target values
        """

        # converted labels
        y = self.le.fit_transform(y)
        # numbers of vectors in each class
        n_vectors = [len(_X) for _X in X]
        # number of classes
        n_classes = self.le.classes_.size
        # number of subspace dimension
        n_subdims = self.n_subdims

        mappings = np.array([y[i] for i, n in enumerate(n_vectors) for _ in range(n)])

        # Data matrix, shape: (sum of n_vectors, n_dims)
        X = np.vstack(X)

        if self.normalize:
            X = _normalize(X)

        self.train_X = X
        self.labels = y
        self.mappings = mappings

        # K is a Grammian matrix of all vectors, shape: (sum of n_vectors, sum of n_vectors)
        K = self.kernel_func(X, X, self.sigma)

        # A has dual vectors of each class in its diagonal
        A = np.zeros((n_classes * n_subdims, sum(n_vectors)))
        s = 0
        for i in range(n_classes):
            p = (mappings == y[i])
            # clipping K(X[y==c], X[y==c])
            _K = K[p][:, p]
            _A, _ = dual_vectors(_K)
            A[i*n_subdims: (i+1)*n_subdims, s: s+n_vectors[i]] = _A[:n_subdims]
            s = s + n_vectors[i]

        # D is a matrix, shape: (n_classes *0 n_subdims, n_classes * n_subdims)
        D = A @ K @ A.T

        # Dual vectors over all classes
        B, e = dual_vectors(D)
        self.W = D @ B.T @ np.diag(1/e) @ B @ A

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: array-like, shape: (n_samples, n_vectors, n_features)
            Data Matrix

        Returns:
        --------
        pred: array-like, shape: (n_sample)
            Predictions

        """

        n_subdims = self.n_subdims
        pred = []
        for _X in X:
            if self.normalize:
                _X = _normalize(_X)

            K = self.kernel_func(_X, _X)
            A, _ = dual_vectors(K)
            A = A[: self.n_subdims]

            K = self.kernel_func(self.train_X, _X)
            S = self.W @ K @ A.T

            i = np.argmax([
                mean_square_singular_values(S[i*n_subdims: (i+1)*n_subdims]) for i, _ in enumerate(self.labels)
            ])
            pred.append(self.labels[i])

        return self.le.inverse_transform(pred)

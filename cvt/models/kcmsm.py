"""
Kernel Constrained Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values, subspace_bases


class KernelCMSM(BaseEstimator, ClassifierMixin):
    """
    Kernel Constrained Mutual Subspace Method

    Parameters
    ----------
    n_subdims : int
        A dimension of subspace. it must be smaller than the dimension of original space.

    n_kgds : int
        A dimension of Kernel GDS

    normalize : boolean, optional (default=True)
        If this is True, all vectors are normalized as |v| = 1

    kernel_func : function, optional (default=rbf_kernel)
        A function which takes 2 matrix and returns gram matrix

    sigma : float, optional (default=None)
        A parameter of rbf_kernel
    """

    def __init__(self, n_subdims, n_kgds, normalize=True, kernel_func=rbf_kernel, sigma=None):
        self.n_subdims = n_subdims
        self.n_kgds = n_kgds
        self.normalize = normalize
        self.sigma = sigma
        self.kernel_func = kernel_func
        self.le = LabelEncoder()
        self.train_X = None
        self.labels = None
        self.mappings = None
        self.W = None
        self.refs = None

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
        X: array-like, shape = (n_samples, n_vectors, n_features)
            Training vectors, where n_samples is number of samples, n_vectors is number of vectors on each samples
            and n_features is number of features. Since n_vectors may be variable on each samples, X can be lists
            containing n_sample matrices: [array(n_vectors{1}, n_features),..., array(n_vectors{n_samples}, n_features)]

        y: integer array, shape = (n_samples, )
            Target values
        """

        # converted labels
        y = self.le.fit_transform(y)
        # number of classes
        n_classes = self.le.classes_.size
        # number of subspace dimension
        n_subdims = self.n_subdims
        # an array which maps a vector to its label
        mappings = np.array([y[i] for i, _X in enumerate(X) for _ in range(len(_X))])

        if self.normalize:
            X = [_normalize(_X) for _X in X]

        # Data matrix, shape: (sum of n_vectors, n_features)
        X = np.vstack(X)

        # (sum of n_vectors, n_features) -> (n_features, sum of n_vectors)
        X = X.T

        self.train_X = X
        self.labels = y
        self.mappings = mappings

        # K is a Grammian matrix of all vectors, shape: (sum of n_vectors, sum of n_vectors)
        K = self.kernel_func(X.T, X.T, self.sigma)

        # A has dual vectors of each class in its diagonal
        E = []
        for i in range(n_classes):
            p = (mappings == y[i])
            # clipping K(X[y==c], X[y==c])
            _K = K[p][:, p]
            _A, _ = dual_vectors(_K, n_subdims)
            E.append(_A)
        E = block_diag(*E)

        # D is a matrix, shape: (n_classes * n_subdims, n_classes * n_subdims)
        D = E.T @ K @ E

        # Dual vectors over all classes
        B, _ = dual_vectors(D)
        B = B[:, len(B) - self.n_kgds:]
        W = B.T @ E.T

        X_kgds = W @ K
        X_kgds = [X_kgds[:, mappings == y[i]] for i in range(n_classes)]

        # reference subspaces
        refs = [subspace_bases(_X, self.n_subdims) for _X in X_kgds]

        self.X_kgds = X_kgds
        self.W = W
        self.refs = refs

    def predict(self, X):
        """
        Predict each classes

        Parameters:
        -----------
        X: array-like, shape: (n_samples, n_vectors, n_features)
            Data Matrix

        Returns:
        --------
        pred: array-like, shape: (n_samples, )
            Predictions

        """

        n_data = len(X)
        n_subdims = self.n_subdims

        mappings = np.array([i for i, _X in enumerate(X) for _ in range(len(_X))])

        if self.normalize:
            X = [_normalize(_X) for _X in X]

        X = np.vstack(X).T

        K = self.kernel_func(self.train_X.T, X.T)
        X_kgds = self.W @ K
        X_kgds = [X_kgds[:, mappings == i] for i in range(n_data)]

        # input subspaces
        inputs = [subspace_bases(_X, self.n_subdims) for _X in X_kgds]

        # similarities per references
        similarities = np.array([[mean_square_singular_values(ref.T @ _input) for ref in self.refs] for _input in inputs])

        pred = self.labels[np.argmax(similarities, axis=1)]

        return self.le.inverse_transform(pred)
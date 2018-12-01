"""
Subspace Method Interface
"""

# Authors: Junki Ishikawa
import itertools
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from cvt.utils import subspace_bases
from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values


class SMBase(BaseEstimator, ClassifierMixin):
    """
    Base class of Subspace Method
    """
    param_names = {'normalize', 'n_subdims'}

    def __init__(self, n_subdims, normalize=False, faster_mode=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1
        """
        self.n_subdims = n_subdims
        self.normalize = normalize
        self.faster_mode = faster_mode
        self.le = LabelEncoder()
        self.dic = None
        self.labels = None
        self.n_classes = None
        self.params = ()

    def get_params(self, deep=True):
        return {name: getattr(self, name) for name in self.param_names}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _prepare_X(self, X):
        """
        preprocessing data matricies X.
        normalize and transpose

        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
        """
        # normalize each vectors
        if self.normalize:
            X = [_normalize(_X) for _X in X]

        # transpose to make feature vectors as column vectors
        # this makes it easy to implement refering to formula
        X = [_X.T for _X in X]

        return X

    def _prepare_y(self, y):
        # converted labels
        y = self.le.fit_transform(y)
        self.labels = y

        # number of classes
        self.n_classes = self.le.classes_.size

        # number of data
        self.n_data = len(y)

        return y

    def fit(self, X, y):
        """
        Fit the model using the given data and parameters

        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_samples, n_dims)
            Training vectors. n_classes is count of classes.
            n_samples is number of vectors of samples, this is variable across each classes.
            n_dims is number of dimentions of vectors.

        y: integer array, (n_classes)
            Class labels of training vectors. 
        """

        # preprocessing data matricies
        # ! X[i] will transposed for conventional
        X = self._prepare_X(X)
        y = self._prepare_y(y)

        self._fit(X, y)

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        self.dic = dic

    def predict(self, X):
        """
        Predict classes

        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.

        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array

        """
        
        if self.faster_mode and hasattr(self, 'fast_predict_proba'):
            proba = self.fast_predict_proba(X)
        else:
            proba = self.predict_proba(X)
        return self.proba2class(proba)

    def proba2class(self, proba):
        pred = self.labels[np.argmax(proba, axis=1)]
        return self.le.inverse_transform(pred)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: 2d-array, (n_samples, n_dims)
            Matrix of input vectors.

        Returns:
        --------
        pred: array-like, shape: (n_samples)
            Prediction array

        """

        # preprocessing data matricies
        X = self._prepare_X([X])[0]
        pred = self._predict_proba(X)
        return pred

    def _predict_proba(self, X):
        """
        Parameters
        ----------
        X: arrays, (n_dims, n_samples)
        """
        raise NotImplementedError('_predict is not implemented')


class KernelSMBase(SMBase):
    """
    Base class of Kernel Subspace Method
    """
    param_names = {'normalize', 'n_subdims', 'sigma'}

    def __init__(self, n_subdims, normalize=False, sigma=None, faster_mode=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1

        sigma : int or str, optional (default=None)
            a parameter of rbf kernel. if sigma is None, sqrt(n_dims / 2) will be used.
        """
        super(KernelSMBase, self).__init__(n_subdims, normalize, faster_mode)
        self.sigma = sigma

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """
        coeff = []
        for _X in X:
            K = rbf_kernel(_X, _X, self.sigma)
            _coeff, _ = dual_vectors(K, self.n_subdims)
            coeff.append(_coeff)

        self.dic = list(zip(X, coeff))


class ConstrainedSMBase(SMBase):
    """
    Base class of Constrained Subspace Method
    """
    param_names = {'normalize', 'n_subdims', 'n_gds_dims'}

    def __init__(self, n_subdims, n_gds_dims, normalize=False):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        n_gds_dims : int
            The dimension of Generalized Difference Subspace.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1.
        """
        super(ConstrainedSMBase, self).__init__(n_subdims, normalize)
        self.n_gds_dims = n_gds_dims
        self.gds = None

    def _gds_projection(self, bases):
        """
        GDS projection.
        Projected bases will be normalized and orthogonalized.

        Parameters
        ----------
        bases: arrays, (n_dims, n_subdims)

        Returns
        -------
        bases: arrays, (n_gds_dims, n_subdims)
        """

        # bases_proj, (n_gds_dims, n_subdims)
        bases_proj = np.matmul(self.gds.T, bases)
        qr = np.vectorize(np.linalg.qr, signature='(n,m)->(n,m),(m,m)')
        bases, _ = qr(bases_proj)
        return bases

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        dic = [subspace_bases(_X, self.n_subdims) for _X in X]
        # dic,  (n_classes, n_dims, n_subdims)
        dic = np.array(dic)
        # all_bases, (n_dims, n_classes * n_subdims)
        all_bases = np.hstack(dic)

        # n_gds_dims
        if 0.0 < self.n_gds_dims <= 1.0:
            n_gds_dims = int(all_bases.shape[1] * self.n_gds_dims)
        else:
            n_gds_dims = self.n_gds_dims

        # gds, (n_dims, n_gds_dims)
        self.gds = subspace_bases(all_bases, n_gds_dims, higher=False)

        dic = self._gds_projection(dic)
        self.dic = dic


class KernelCSMBase(SMBase):
    """
    Base class of Kernel Constrained Subspace Method
    """
    param_names = {'normalize', 'n_subdims', 'sigma', 'n_gds_dims'}

    def __init__(self, n_subdims, n_gds_dims, normalize=False, sigma=None):
        """
        Parameters
        ----------
        n_subdims : int
            The dimension of subspace. it must be smaller than the dimension of original space.

        n_gds_dims : int
            The dimension of Generalized Difference Subspace.

        normalize : boolean, optional (default=True)
            If this is True, all vectors are normalized as |v| = 1.
        """
        super(KernelCSMBase, self).__init__(n_subdims, normalize)

        self.sigma = sigma
        self.n_gds_dims = n_gds_dims
        self.gds = None

    def _fit(self, X, y):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_classes, n_dims, n_samples)
        y: array, (n_classes)
        """

        # mapings, (n_classes * n_samples)
        mappings = np.array([
            self.labels[i] for i, _X in enumerate(X)
            for _ in range(_X.shape[1])
        ])

        # stack_X, (n_dims, n_classes * n_samples)
        stack_X = np.hstack(X)

        # K, (n_classes * n_samples, n_classes * n_samples)
        K = rbf_kernel(stack_X, stack_X, self.sigma)

        coeff = []
        for i, _ in enumerate(X):
            p = (mappings == y[i])
            # clipping K(X[y==c], X[y==c])
            # _K, (n_samples_i, n_samples_i)
            _K = K[p][:, p]
            _coeff, _ = dual_vectors(_K, self.n_subdims)
            coeff.append(_coeff)
        # coeff, (n_classes * n_samples, n_classes * n_subdims)
        coeff = block_diag(*coeff)

        # gramian, (n_classes * n_subdims, n_classes * n_subdims)
        gramian = coeff.T @ K @ coeff

        # n_gds_dims
        if 0.0 < self.n_gds_dims <= 1.0:
            n_gds_dims = int(gramian.shape[0] * self.n_gds_dims)
        else:
            n_gds_dims = self.n_gds_dims

        # coefficients of `bases in feature space` to get GDS bases in feature space
        # gds_coeff, (n_classes * n_subdims, n_gds_dims)
        gds_coeff, _ = dual_vectors(gramian, n_gds_dims, higher=False)

        # coefficients of `given data X in feature space` to get GDS bases in features space
        # gds_coeff, (n_classes * n_samples, n_gds_dims)
        gds_coeff = np.dot(coeff, gds_coeff)

        # X_gds = (GDS bases in feature space).T @ (X_stacked in feature space)
        # projection coefficients of GDS in feature space.
        # this vectors have finite dimension, such as n_gds_dims,
        # and these vectors can treat as linear feature.
        # X_gds, (n_gds_dims, n_classes * n_samples)
        X_gds = np.dot(gds_coeff.T, K)
        # X_gds, (n_classes, n_gds_dims, n_samples)
        X_gds = [X_gds[:, mappings == y[i]] for i, _ in enumerate(X)]

        # dic, (n_classes, n_dims, n_subdims)
        dic = [subspace_bases(_X, self.n_subdims) for _X in X_gds]
        dic = np.array(dic)

        self.train_stack_X = stack_X
        self.mappings = mappings
        self.gds_coeff = gds_coeff
        self.dic = dic

class MSMInterface(object):
    """
    Prediction interface of Mutual Subspace Method
    """

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: list of 2d-arrays, (n_vector_sets, n_samples, n_dims)
            List of input vector sets.

        Returns:
        --------
        pred: array, (n_vector_sets)
            Prediction array

        """

        # preprocessing data matricies
        X = self._prepare_X(X)

        pred = []
        for _X in X:
            # gramians, (n_classes, n_subdims, n_subdims)
            gramians = self._get_gramians(_X)

            # i_th singular value of grammian of subspace bases is
            # square root of cosine of i_th cannonical angles
            # average of square of them is caonnonical angle between subspaces
            c = [mean_square_singular_values(g) for g in gramians]
            pred.append(c)
        return np.array(pred)

    def _get_gramians(self, X):
        """
        Parameters
        ----------
        X: array, (n_dims, n_samples)

        Returns
        -------
        G: array, (n_class, n_subdims, n_subdims)
            gramian matricies of references of each class
        """
        raise NotImplementedError('_get_gramians is not implemented')

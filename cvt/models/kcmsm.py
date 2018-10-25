"""
Kernel Constrained Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np
from scipy.linalg import block_diag

from .base_class import KernelCSMBase, MSMInterface
from cvt.utils import rbf_kernel
from cvt.utils import subspace_bases


class KernelCMSM(MSMInterface, KernelCSMBase):
    """
    Kernel Constrained Mutual Subspace Method
    """

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

        # K, (n_class * n_samples_train, n_samples)
        K = rbf_kernel(self.train_stack_X, X)
        # X_gds, (n_gds_dims, n_samples)
        X_gds = np.dot(self.gds_coeff.T, K)

        # input subspace bases
        bases = subspace_bases(X_gds, self.n_subdims)

        # grammians, (n_classes, n_subdims, n_subdims)
        gramians = np.dot(self.dic.transpose(0, 2, 1), bases)
        return gramians

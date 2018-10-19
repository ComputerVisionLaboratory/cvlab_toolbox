"""
Kernel Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np

from .base_class import KernelSMBase, MSMInterface
from cvt.utils import rbf_kernel, dual_vectors


class KernelMSM(MSMInterface, KernelSMBase):
    """
    Kernel Mutual Subspace Method
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

        # _X, (n_dims, n_samples)
        # K, (n_samples, n_samples)
        K = rbf_kernel(X, X, self.sigma)
        # in_coeff, (n_samles, n_subdims)
        in_coeff, _ = dual_vectors(K, self.n_subdims)

        gramians = []
        for i in range(self.n_classes):
            # ref_X, (n_dims, n_samples_ref_X)
            # ref_coeff, (n_samples_ref_X, n_subdims)
            ref_X, ref_coeff = self.dic[i]

            # _K, (n_samples_ref_X, n_samples)
            _K = rbf_kernel(ref_X, X)
            # S, (n_subdims, n_subdims)
            S = ref_coeff.T.dot(_K.dot(in_coeff))

            gramians.append(S)

        return np.array(gramians)

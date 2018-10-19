"""
Kernel Mutual Subspace Method
"""

# Authors: Junki Ishikawa

from sklearn.preprocessing import normalize as _normalize, LabelEncoder
import numpy as np

from .base_class import KernelSMBase, MSMInterface
from cvt.utils import rbf_kernel, dual_vectors, mean_square_singular_values


class KernelMSM(MSMInterface, KernelSMBase):
    """
    Kernel Mutual Subspace Method
    """

    def _predict(self, X):
        """
        Parameters
        ----------
        X: list of 2d-arrays, (n_vector_sets, n_dims, n_samples)
        """

        pred = []
        for _X in X:
            K = rbf_kernel(_X, _X, self.sigma)
            in_coeff, _ = dual_vectors(K, self.n_subdims)

            c = []
            for i in range(self.n_classes):
                ref_X, ref_coeff = self.dict[i]

                _K = rbf_kernel(ref_X, _X)
                S = ref_coeff.T.dot(_K).dot(in_coeff)
                _c = mean_square_singular_values(S)
                c.append(_c)
            pred.append(self.labels[np.argmax(c)])
        pred = np.array(pred)
        return pred

import numpy as np
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

class SM(BaseEstimator):
    """Subspace Method (SM)
    Classification method using Subspace.
    
    Parameters
    ----------
    n_dimension : int
        Number of dimension of subspace.

    Attributes
    ----------
    subspaces_ : array, shape (n_classes, n_features, n_features)
        Subspace of N classes.
        
    classes_ : array, shape (n_classes,)
        Unique labels of Classes.
    """
    
    def __init__(self, n_dimension):
        self.n_dimension = n_dimension
    
    def fit(self, X, y):
        """Fit the model with X.
        X should be normalized because this method uses PCA.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.
        y : array_like, shape (n_samples,)
            Label data, where n_samples in the number of samples
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.classes_ = np.unique(y)
        self.subspaces_ = []
        for class_name in self.classes_:
            idx = np.argwhere(y == class_name).squeeze()
            X_class_i = X[idx, :]

            pca = PCA(n_components=self.n_dimension)
            pca.fit(X_class_i)

            # define projection matrix (equal to subspace)
            P = pca.components_.T @ pca.components_
            self.subspaces_.append(P)
            
        self.subspaces_ = np.array(self.subspaces_)
         
        return self

    def predict(self, X):
        """
        Perform classification on samples in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Class labels for samples in X.
        """

        #===<normal version>===#
#         n, d = X.shape
#         preds = []
#         for x in X:
#             similarities = []
#             for P in self.subspaces_:
#                 similarities.append(x.T @ P @ x)
#             preds.append(np.argmax(similarities))
#         return self.classes_[preds]

        #===<faster version (using broadcast)>===#
        # n: n_samples, d: n_features
        n, d = X.shape

        similarities = X.reshape((n, 1, 1, d)) @ \
            np.expand_dims(self.subspaces_, axis=0) @ \
            X.reshape((n, 1, d, 1))
        similarities = similarities.squeeze()
        
        return self.classes_[np.argmax(similarities, axis=1)]
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_params(self, deep=True):
        return {'n_dimension': self.n_dimension}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self,parameter, value)
        return self

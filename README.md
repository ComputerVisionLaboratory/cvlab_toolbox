# cvlab_toolbox
This is the repository of CVLAB toolbox


## Usage
- Scikit-learn API
```python
import numpy as np
from numpy.random import randint, rand
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from cvt.models import KernelMSM

dim = 100
n_class = 4
n_train, n_test = 20, 5

# input data X is list of vector sets (list of 2d-arrays)
X_train = [rand(randint(10, 20), dim) for i in range(n_train)]
X_test = [rand(randint(10, 20), dim) for i in range(n_test)]

# labels y is 1d-array
y_train = randint(0, n_class, n_train)
y_test = randint(0, n_class, n_test)

model = KernelMSM(n_subdims=3, sigma=0.01)
# fit
model.fit(X_train, y_train)
# predict
pred = model.predict(X_test)

print(accuracy_score(pred, y_test))

```

## Install
- pip
```bash
pip install -U git+https://github.com/ComputerVisionLaboratory/cvlab_toolbox
```

## Coding styles
- Follow `PEP8` as much as possible
  - [English](https://www.python.org/dev/peps/pep-0008/)
  - [日本語](http://pep8-ja.readthedocs.io/ja/latest/)
- Write a description as **docstring**
  ```python
  def PCA(X, whiten = False):
    '''
      apply PCA
      components, explained_variance = PCA(X)

      Parameters
      ----------
      X: ndarray, shape (n_samples, n_features)
        matrix of input vectors

      whiten: boolean
        if it is True, the data is treated as whitened
        on each dimensions (average is 0 and variance is 1)

      Returns
      -------
      components: ndarray, shape (n_features, n_features)
        the normalized component vectors

      explained_variance: ndarray, shape (n_features)
        the variance of each vectors
    '''

    ...
  ```

## Contribution rules
1. Make a pull request
2. Ask some lab members to review the code
3. when all agreements are taken, ask any admin member to merge it

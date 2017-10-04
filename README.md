# cvlab_toolbox
This is the repository of CVLAB toolbox

## Coding styles
- Follow `PEP8`
  - [English](https://www.python.org/dev/peps/pep-0008/)
  - [日本語](http://pep8-ja.readthedocs.io/ja/latest/)
- Write a description as docstring
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

import numpy as np


def calc_er(X, y, labels=None, data_type='S'):
    '''
    calculate Error Rate (ER)

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_elements)
        data matrix
    y: ndarray, shape (n_samples)
        true label (integer)
    labels(optional): ndarray, shape (n_elements)
        labels that represents what class each element(row) belongs to
        if it's not given, each rows are treated as independent class
    data_type(optional): string
        'S': Similarity (default)
        'D': Distance

    Returns
    -------
    er: float
        error rate
    '''

    _, _pred_class, mappings = _calc_preparations(X, labels, data_type)

    if mappings is not None:
        _y = np.array([mappings[v] for v in y])
    else:
        _y = y

    er = (_pred_class == _y).mean()

    return er


def calc_eer(X, labels=None, data_type='S'):
    '''
    calculate Error Rate (ER)

    Parameters
    ----------
    X: ndarray, shape (n_samples, n_elements)
        data matrix
    labels(optional): ndarray, shape (n_elements)
        labels that represents what class each element(row) belongs to
        if it's not given, each rows are treated as independent class
    data_type(optional): string
        'S': Similarity (default)
        'D': Distance

    Returns
    -------
    eer: float
        equal error rate
    thresh: float
        threashold
    '''

    _labels, _pred_class, _ = _calc_preparations(X, labels, data_type)

    # B has same shape with X
    # each vector elements are 1 if its class is predected True and 0 if False
    n_classes = len(np.unique(_labels))
    print(_labels)
    B = np.eye(n_classes)[_labels][:, _pred_class]

    # make them 1D
    _X = X.reshape(-1)
    _B = B.reshape(-1)

    # get sorted index regarding to data_type
    if data_type == 'S':
        sort_idx = np.argsort(_X)
    else:
        sort_idx = np.argsort(_X)[:, :, -1]

    # number of positive/negative prediction
    n_pos = _B[_B == 1].size
    n_neg = _B.size - n_pos

    # make _B sorted
    _B = _B[sort_idx]

    # False Acceptance Rate
    far = 1 - np.cumsum(_B == 0) / n_neg
    # False Rejection Rate
    frr = np.cumsum(_B == 1) / n_pos

    thresh_idx = np.abs(far - frr).argmin()
    eer = (far[thresh_idx] + frr[thresh_idx]) / 2
    thresh = _X[sort_idx][thresh_idx]

    return eer, thresh


def _calc_preparations(X, labels, data_type):
    # check data_type
    if data_type not in {'S', 'D'}:
        raise ValueError('`data_type` must be \'S\' or \'D\'')

    # check shape
    if len(X.shape) != 2:
        raise ValueError('`X` must be 2 dimensional matrix')

    # when `labels` is not given,
    # all elements belongs to each unique class
    if labels is None:
        _labels = np.arange(X.shape[1])
        mappings = None
    else:
        classes = np.unique(labels)
        mappings = np.stack((classes, np.arange(len(classes))), axis=1)
        mappings = dict(mappings)
        _labels = np.array([mappings[v] for v in labels])

    # when data type is similarity
    if data_type == 'S':
        idx = X.argmax(axis=1)
    else:
        idx = X.argmin(axis=1)

    _pred_class = _labels[idx]

    return _labels, _pred_class, mappings


if __name__ == '__main__':
    X = np.array([
        [.9, .1, .1, .1, .1, .1, .1, .1, .1, .1],
        [.1, .9, .1, .1, .1, .1, .1, .1, .1, .1],
        [.1, .1, .9, .1, .1, .1, .1, .1, .1, .1],
        [.1, .1, .1, .9, .1, .1, .1, .1, .1, .1],
        [.1, .1, .1, .1, .9, .1, .1, .1, .1, .1],
        [.1, .1, .1, .1, .1, .9, .1, .1, .1, .1],
        [.1, .1, .1, .1, .1, .1, .9, .1, .1, .1],
        [.1, .1, .1, .1, .1, .1, .1, .9, .1, .1],
        [.1, .1, .1, .1, .1, .1, .1, .1, .9, .1],
        [.1, .1, .1, .1, .1, .1, .1, .1, .1, .9],
    ])

    y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])

    print(calc_er(X, y))
    print(calc_eer(X))
    print(calc_er(X, y, labels))
    print(calc_eer(X, labels))

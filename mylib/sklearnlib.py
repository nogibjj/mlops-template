"""
sk-learn hello world demo
"""
from sklearn import linear_model

def linearRegression(X, y, sample_weight=None):
    """
    Lineaer Regresssion based on sklearn

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.

    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values. Will be cast to X's dtype if necessary.

    sample_weight : array-like of shape (n_samples,), default=None
        Individual weights for each sample.

        .. versionadded:: 0.17
            parameter *sample_weight* support to LinearRegression.

    Returns
    -------
    self : object
        Fitted Estimator.
    """
    reg = linear_model.LinearRegression()
    reg.fit(X, y, sample_weight)
    return reg
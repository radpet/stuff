import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_X_y


class NBWeights(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y=None):
        x, y = check_X_y(X, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))

        return self

    def transform(self, X, y=None):
        return X.multiply(self._r)

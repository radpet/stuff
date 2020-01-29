import pickle

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HateModel(BaseEstimator, TransformerMixin):
    def __init__(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('Feature extraction via Hate model')
        print(self.model.classes_)
        return self.model.predict(X)


if __name__ == '__main__':
    hate = HateModel('../data/hate_model.pkl')

    print(hate.transform(np.array(["I love you","Hello", "this is hate, I hate you"])))

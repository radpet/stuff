import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator

from util import make_batch



class USEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, model_ver = 'https://tfhub.dev/google/universal-sentence-encoder/4', batch_size=5000):
        self.model = None
        self.model_ver = model_ver
        self.batch_size = batch_size

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        import tensorflow_hub as hub
        if self.model is None:
            print("model %s loaded" % self.model_ver)
            self.model = hub.load(self.model_ver)
        else:
            print("model %s has already been loaded" % self.model_ver)
        embedded = None
        for i, batch in enumerate(make_batch(X, self.batch_size)):
            print("Processing batch ", i)

            batch_embeddings = self.model(batch)
            if embedded is None:
                embedded = batch_embeddings.numpy()
            else:
                embedded = np.append(embedded, batch_embeddings, axis=0)
            print("Shape ", embedded.shape)

        return embedded

    def __getstate__(self):
        return self.model_ver

    def __setstate__(self, state):
        self.model_ver = state


if __name__ == '__main__':
    encoder = USEEncoder()

    print(encoder.transform(["Hello world"]))